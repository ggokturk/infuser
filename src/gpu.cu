#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include "common.h"
#include "sketch.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
//bool verbose = false;
template <class T> thrust::device_ptr<T> p(T* ptr) { return thrust::device_ptr <T>(ptr); }
inline void assert_gpu(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        cerr << "CUDAERR:" << cudaGetErrorString(code) << file << ":" << line << endl;
        abort();
    }
}
#define cuchk(ans) { assert_gpu((ans), __FILE__, __LINE__); }
template <class T>
T* get_dev(size_t size) {
    T* t;
    cuchk(cudaMalloc((void**)&t, sizeof(T) * size));
    cuchk(cudaMemset(t, 0, sizeof(T) * size));
    return t;
}
template <class T>
T* devcpy(T* host_t, size_t size) {
    T* dev_t;
    cuchk(cudaMalloc((void**)&dev_t, sizeof(T) * size));
    cuchk(cudaMemcpy(dev_t, host_t, size * sizeof(T), cudaMemcpyHostToDevice));
    return dev_t;
}

__device__
inline uint32_t dev_hash(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53L;
    h ^= h >> 33;
    return h & HASHMASK;//FFFF;
}
void read_to_gpu(char* dst, ifstream& rf, size_t bytes) {
    size_t buf_size = 1024 * 1024 * 1024;
    size_t remaining_bytes = bytes;
    auto buf = get_aligned<char>(buf_size);
    size_t pos = 0;
    while (remaining_bytes > 0 && rf.read(buf.get(), std::min(buf_size, remaining_bytes))) {
        cuchk(cudaMemcpy(dst + pos, buf.get(), rf.gcount(), cudaMemcpyHostToDevice));
        pos += std::min(buf_size, remaining_bytes);
        remaining_bytes -= std::min(buf_size, remaining_bytes);
    }
}
graph_t read_bin_gpu(string filename) {
    graph_t g;
    ifstream rf(filename, std::ios::out | std::ios::binary);
    if (!rf) {
        cerr << "Cannot open file!" << endl;
        abort();
    }
    int mode;
    rf.read((char*)&g.n, sizeof(g.n));
    rf.read((char*)&g.m, sizeof(g.m));
    g.xadj = get_dev<size_t>(g.n + 1); //new size_t[g.n + 1];
    g.adj = get_dev<edge_t>(g.m);//new edge_t[g.m];
    read_to_gpu((char*)g.xadj, rf, size_t(g.n + 1) * sizeof(g.xadj[0]));
    read_to_gpu((char*)g.adj, rf, size_t(g.m) * sizeof(edge_t));
    return g;
}

__device__
inline uint64_t dev_hash64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53L;
    h ^= h >> 33;
    return h;
}
__device__
inline uint32_t dev_edge_hash(uint32_t a, uint32_t b) {
    return dev_hash64((((uint64_t)a) << 32) | b) & INT32_MAX;
}

__global__
void fill_hypersx(char* hypers, size_t n)
{
    size_t R = blockDim.x;
    int j = threadIdx.x;
    for (size_t i = blockIdx.x; i < n; i += (gridDim.x))
        if (hypers[i * R + j] == -1) continue;
        else hypers[i * R + j] = __clzll(dev_hash64(~(i * R + j + R)));
}
const auto ITEMS_PER_REG = 4;
__global__
void fill_hypersx(float* hypers, size_t n)
{
    size_t R = blockDim.x;
    int j = threadIdx.x;

    for (size_t i = blockIdx.x; i < n; i += (gridDim.x))
    {
        if (hypers[i * R + j] == -1) continue;
        float reg = 0;
        uint64_t nonce = ~(i * R + j + R);
        for (int k = 0; k < ITEMS_PER_REG; k++) {
            nonce = dev_hash64(nonce);
            reg += 1.f / (float)__clzll(nonce);
        }
        hypers[i * R + j] = ITEMS_PER_REG / reg;
    }
}
__global__
void fill_hypersx(char* hypers, const size_t n, const int c, const int offset) {
    size_t R = blockDim.x;
    int j = threadIdx.x;
    for (size_t i = blockIdx.x; i < n; i += (gridDim.x))
        if (hypers[i * R + j] == -1) continue;
        else hypers[i * R + j] = __clzll(dev_hash64(~(i * c + j + R + offset)));
}
__global__
void fill_hypersx(float* hypers, const size_t n, const int c, const int offset) {
    size_t R = blockDim.x;
    int j = threadIdx.x;

    for (size_t i = blockIdx.x; i < n; i += (gridDim.x))
    {
        if (hypers[i * R + j] == -1) continue;
        float reg = 0;
        uint64_t nonce = ~(i * c + j + R + offset);
        for (int k = 0; k < ITEMS_PER_REG; k++) {
            nonce = dev_hash64(nonce);
            reg += 1.f / (float)__clzll(nonce);
        }
        hypers[i * R + j] = ITEMS_PER_REG / reg;
    }
}
template<typename T>
__global__
void simulate_kernelng(graph_t g, size_t R, T* hypers, int* rands) {
    int j = threadIdx.x;
    const auto r = rands[j];
    for (size_t i = blockIdx.x; i < g.n; i += (gridDim.x)) {
        auto reg = hypers[i * R + j];
        if (reg < 0) continue;
        for (size_t x = g.xadj[i]; x < g.xadj[i + 1]; x++) {
            const edge_t e = g.adj[x];
            auto hash = dev_edge_hash(i, e.v);
            if (((hash ^ r) <= e.w))
                reg = max(reg, hypers[e.v * R + j]);
        }
        hypers[i * R + j] = reg;
    }
}

template <typename T>
__inline__ __device__
T reduce_warp(T val) {
    for (int offset = (warpSize >> 1); offset > 0; offset = (offset >> 1)) // div 2 -> shift 1
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
template <typename T>
__inline__ __device__
T reduce_block(T val) {
    static __shared__ T s[32];
    int lane = threadIdx.x & 0x1f; // last 5 bits== mod 32
    int warp = threadIdx.x >> 5; // shift 5 -> div 32
    val = reduce_warp(val);
    if (lane == 0) s[warp] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? s[lane] : 0;
    if (warp == 0) val = reduce_warp(val);
    return val;
}

template<typename T, typename T2>
__global__
void maxsum_gpu(T* hypers, int R, size_t N, T* mask, T2* estimates) {
    int j = threadIdx.x;
    const auto reg = mask[j];
    for (size_t i = blockIdx.x; i < N; i += (gridDim.x)) {
        T2 m = max(hypers[i * R + j], reg);
        m = reduce_block(m);
        if (j == 0)  estimates[i] = m;
    }
}

template<typename T, typename T2>
__global__
void harmonicmean_kernel(T* hypers, int R, size_t N, T* mask, T2* estimates) {
    int j = threadIdx.x;
    T reg = mask[j];
    for (size_t i = blockIdx.x; i < N; i += (gridDim.x)) {
        T2 m = max(hypers[i * R + j], reg);
        int b = m >= 0;
        if (b) m = 1.0f / float(m);
        else m = 0;
        b = reduce_block(b);
        m = reduce_block(m);
        if (j == 0)
            if (m == 0) estimates[i] = 0;
            else estimates[i] = (float(b)) / m;
    }
}

auto N_BLOCKS = 2048 * 2;
auto N_THREADS = 1024;

#ifdef _MSC_VER
#define __builtin_clzll __lzcnt64
#endif

template<typename T>
__global__
void process_queueng(
    const int* q, const int* q_end,
    int* q_next, int* q_next_end, char* in_q,
    const graph_t g, const size_t R, const int* rands, T* hypers, const size_t QUEUE_LIMIT) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    int stride = gridDim.x;
    int rnd = rands[j];
    int score = 0;
    const size_t end = *q_end;
    for (size_t q_i = i; q_i < end; q_i += stride) {
        int vertex = q[q_i];
        in_q[vertex] = false;
        if (hypers[vertex * R + j] != -1)
            continue;
        for (size_t it = g.xadj[vertex]; it < g.xadj[vertex + 1]; it++) {
            int flag = 0;
            const edge_t e = g.adj[it];
            auto hash = dev_edge_hash(vertex, e.v);
            if (((rnd ^ hash) <= e.w) && hypers[e.v * R + j] != -1) {
                hypers[e.v * R + j] = -1;
                flag = 1;
            }
            //flag = reduce_block(flag);
            //flag = __any(flag);
            //flag = __any_sync(0xffffffff,flag);
            //flag = any_block(flag);
            if (
                //(j%32)==0 &&
                //j == 0 &&
                flag &&
                !in_q[e.v]
                ) {
                if (*q_next_end >= g.n) return;
                int pos = atomicAdd(q_next_end, 1);
                q_next[pos] = e.v;
                in_q[e.v] = 1;
            }
        }
    }

}
#include <numeric>

void syncall(int num_streams) {
#pragma omp parallel for
    for (int sid = 0; sid < num_streams; sid++) {
        cudaSetDevice(sid % num_streams);
        cudaDeviceSynchronize();
    }
}

size_t get_free_vram() {
    size_t free_byte, total_byte;
    cuchk(cudaMemGetInfo(&free_byte, &total_byte));
    return free_byte;
}
template <typename T>
void hyperfuser_gpufx(const graph_t& g, const int K, const size_t R, const float eps, const  float tr, const float trc) {
    int device_id;
    cuchk(cudaGetDevice(&device_id));
    size_t free_before = get_free_vram();

    T* dev_hypers = get_dev<T>(R * g.n),
        * dev_mask = get_dev<T>(R);
    char* dev_active = get_dev<char>(g.n);
    float* dev_fbuffer = get_dev<float>(N_BLOCKS);
    float* dev_estimates = get_dev<float>(g.n);

    int* dev_q = get_dev<int>(g.n);//QUEUE_LIMIT*2);
    int* dev_q_size = get_dev<int>(1);
    int* dev_q_next = get_dev<int>(g.n);//QUEUE_LIMIT*2);
    int* dev_q_next_size = get_dev<int>(1);
    char* dev_in_q = get_dev<char>(g.n);

    auto rand_seeds = get_rands(R);
    std::sort(rand_seeds.get(), rand_seeds.get() + R);
    auto dev_rand_seeds = devcpy(rand_seeds.get(), R);

    fill_hypersx << < N_BLOCKS, R >> > (dev_hypers, g.n);

    graph_t dev_g = g;
    int* dev_buffer = get_dev<int>(N_BLOCKS);
    t.reset();
    int bfs_iter_limit = 20;
    int simulation_limit = 10;
    float old = 0, cnt = 0;

    for (int i = 0; i < simulation_limit; i++) {
        simulate_kernelng << <N_BLOCKS, R >> > (dev_g, R, dev_hypers, dev_rand_seeds);
    }
    bool dont_rebuild = 0, dont_cascade = 0;
    vector<uint64_t> S;
    vector<float> time;
    float score = 0, base = 0, old_score = 0;
    int next_size = 1;
    int i = 0;
    vector<float> estimate, scores;
    while (S.size() < K) {

        maxsum_gpu << <N_BLOCKS, R, R >> > (dev_hypers, R, g.n, dev_mask, dev_estimates);

        auto max_elem = thrust::max_element(p(dev_estimates), p(dev_estimates) + g.n);
        size_t s = thrust::distance(p(dev_estimates), max_elem);
        S.push_back(s);
        time.push_back(t.elapsed());
        float max_val = 0;
        cuchk(cudaMemcpy(&max_val, thrust::raw_pointer_cast(max_elem), sizeof(max_val), cudaMemcpyDeviceToHost));
        thrust::transform(p(dev_hypers + (R * s)), p(dev_hypers + (R * s) + R), p(dev_mask), p(dev_mask), thrust::maximum<float>());
        thrust::fill(p(dev_hypers) + s * R, p(dev_hypers) + s * R + R, -1.f);

        if (score / old_score < 1.001) {
            scores.push_back(score);
            continue;
        }

        thrust::fill(p(dev_q), p(dev_q) + 1, s);
        thrust::fill(p(dev_q_size), p(dev_q_size) + 1, 1);
        int next_size = 1;
        float processed = 0;
        for (int bfs_iter = 0; bfs_iter < bfs_iter_limit; bfs_iter++) {
            cudaMemset(dev_q_next_size, 0, sizeof(int));
            process_queueng << <min(next_size, N_BLOCKS), R >> > (
                dev_q, dev_q_size, dev_q_next, dev_q_next_size, dev_in_q,
                dev_g, R, dev_rand_seeds, dev_hypers, g.n);
            cudaMemcpy(&next_size, dev_q_next_size, sizeof(int), cudaMemcpyDeviceToHost);
            processed += next_size;
            if (next_size == 0) break;
            std::swap(dev_q, dev_q_next);
            std::swap(dev_q_size, dev_q_next_size);
        }
        old_score = score;
        score = thrust::count(p(dev_hypers), p(dev_hypers) + g.n * R, -1.0f) / R;
        scores.push_back(score);

        float est = powf(2.0f, max_val / R);
        float mg = score - base;
        float err = (est - mg) / mg;
        //if (!(err > eps)) continue;

        fill_hypersx << < N_BLOCKS, R >> > (dev_hypers, g.n);
        for (int i = 0; i < simulation_limit; i++)
            simulate_kernelng << <N_BLOCKS, R >> > (dev_g, R, dev_hypers, dev_rand_seeds);
        thrust::fill(p(dev_mask), p(dev_mask + R), 0);
        base = score;

    }
    for (int i = 0; i < K; i++)
        cout << S[i] << "\t" << scores[i] << "\t" << time[i] << "\t" << "0" << endl;
    size_t free_after = get_free_vram();
    cerr << std::fixed << std::setprecision(2) << "ALGO_VRAM_USE:" << float(free_before - free_after) / 1024 / 1024 << "MB" << endl;
    std::cout << std::fixed << std::setprecision(2);
    cudaFree(dev_hypers); cudaFree(dev_mask);
    cudaFree(dev_active); cudaFree(dev_fbuffer);
    cudaFree(dev_estimates); cudaFree(dev_q);
    cudaFree(dev_q_size); cudaFree(dev_q_next);
    cudaFree(dev_q_next_size); cudaFree(dev_in_q);
}
//template <class T>
//T** get_dev(size_t size, size_t devs) {
//    T** t;
//    t = new * T[devs];
//    for (int i = 0; i < devs; i++) {
//        cudaSetDevice(i);
//        t[i] = get_dev<T>(size);
//    }
//    return t;
//}
float* estimates[16], __score[16];
size_t __s;
template <typename T>
void superfuser_gpufx2(const graph_t& g, const int K, const size_t R, int* rand_seeds, int id, int n_devices) {
    T* dev_hypers = get_dev<T>(R * g.n),
        * dev_mask = get_dev<T>(R);
    char* dev_active = get_dev<char>(g.n);
    float* dev_fbuffer;
    dev_fbuffer = get_dev<float>(g.n);
    float* dev_estimates = get_dev<float>(g.n);
    estimates[id] = dev_estimates;
    int* dev_q = get_dev<int>(g.n);//QUEUE_LIMIT*2);
    int* dev_q_size = get_dev<int>(1);
    int* dev_q_next = get_dev<int>(g.n);//QUEUE_LIMIT*2);
    int* dev_q_next_size = get_dev<int>(1);
    char* dev_in_q = get_dev<char>(g.n);

    auto dev_rand_seeds = devcpy(rand_seeds, R);

    fill_hypersx << < N_BLOCKS, R >> > (dev_hypers, g.n, n_devices * R, id * R);

    graph_t dev_g = g;
    //int* dev_buffer = get_dev<int>(N_BLOCKS);
    if (id == 0) t.reset();
    int bfs_iter_limit = 20;
    int simulation_limit = 10;
    float old = 0, cnt = 0;

    for (int i = 0; i < simulation_limit; i++) {
        simulate_kernelng << <N_BLOCKS, R >> > (dev_g, R, dev_hypers, dev_rand_seeds);
    }
    bool dont_rebuild = 0, dont_cascade = 0;
    vector<uint64_t> S;
    vector<float> time;
    float score = 0, base = 0, old_score = 0;
    size_t s;
    int next_size = 1;
    int i = 0;
    vector<float> estimate, scores;
    while (S.size() < K) {

        maxsum_gpu << <N_BLOCKS, R, R >> > (dev_hypers, R, g.n, dev_mask, dev_estimates);

        cudaDeviceSynchronize();
        //auto max_elem = thrust::max_element(p(dev_estimates), p(dev_estimates) + g.n);
        //size_t s = thrust::distance(p(dev_estimates), max_elem);
        //float max_val = 0;
        //cuchk(cudaMemcpy(&max_val, thrust::raw_pointer_cast(max_elem), sizeof(max_val), cudaMemcpyDeviceToHost));
        //
        //#pragma omp critical
        //        cout << id << " selects " << s << " with "<< max_val << endl;

        for (int i = 1; i < n_devices; i *= 2) {
#pragma omp barrier
            if (id % (1 << i) != 0) continue;
            //cerr << id << " reduce " << (id + i) << endl;
            cuchk(cudaMemcpy(dev_fbuffer, estimates[id + i], g.n * sizeof(float), cudaMemcpyDefault));
            thrust::transform(p(dev_fbuffer), p(dev_fbuffer + g.n), p(dev_estimates), p(dev_estimates), thrust::plus<float>());
        }

        if (id == 0) {

            //for (int i = 1; i < n_devices; i++) {
            //    cuchk(cudaMemcpy(dev_fbuffer, estimates[i], g.n * sizeof(float), cudaMemcpyDefault));
            //    thrust::transform(p(dev_fbuffer), p(dev_fbuffer + g.n), p(dev_estimates), p(dev_estimates), thrust::plus<float>());
            //}

            auto max_elem = thrust::max_element(p(dev_estimates), p(dev_estimates) + g.n);
            size_t s = thrust::distance(p(dev_estimates), max_elem);
            S.push_back(s);
            time.push_back(t.elapsed());
            //float max_val = 0;
            //cuchk(cudaMemcpy(&max_val, thrust::raw_pointer_cast(max_elem), sizeof(max_val), cudaMemcpyDeviceToHost));
            __s = s;
        }
#pragma omp barrier
        s = __s;
        thrust::transform(p(dev_hypers + (R * s)), p(dev_hypers + (R * s) + R), p(dev_mask), p(dev_mask), thrust::maximum<float>());
        thrust::fill(p(dev_hypers) + s * R, p(dev_hypers) + s * R + R, -1.f);


        if (score / old_score < 1.001) {
            if (id == 0)
                scores.push_back(score);
            continue;
        }

        thrust::fill(p(dev_q), p(dev_q) + 1, s);
        thrust::fill(p(dev_q_size), p(dev_q_size) + 1, 1);
        int next_size = 1;
        float processed = 0;
        for (int bfs_iter = 0; bfs_iter < bfs_iter_limit; bfs_iter++) {
            cuchk(cudaMemset(dev_q_next_size, 0, sizeof(int)));
            process_queueng << <min(next_size, N_BLOCKS), R >> > (
                dev_q, dev_q_size, dev_q_next, dev_q_next_size, dev_in_q,
                dev_g, R, dev_rand_seeds, dev_hypers, g.n);
            cuchk(cudaMemcpy(&next_size, dev_q_next_size, sizeof(int), cudaMemcpyDeviceToHost));
            processed += next_size;
            if (next_size == 0) break;
            std::swap(dev_q, dev_q_next);
            std::swap(dev_q_size, dev_q_next_size);
        }
        old_score = score;
        score = thrust::count(p(dev_hypers), p(dev_hypers) + g.n * R, -1.0f) / double(R);
        __score[id] = score;

        //#ifdef __sync_scores
#pragma omp barrier
        if (id == 0) {
            score = 0;
            for (int i = 0; i < n_devices; i++)
                score += __score[i] / n_devices;
            scores.push_back(score);
            __score[0] = score;
        }
#pragma omp barrier
        score = __score[0];
        //#else
        //        scores.push_back(score);
        //#endif


                //float est = powf(2.0f, max_val / R);
        float mg = score - base;
        //float err = (est - mg) / mg;
        //if (!(err > eps)) continue;

        fill_hypersx << < N_BLOCKS, R >> > (dev_hypers, g.n, n_devices * R, id * R);
        for (int i = 0; i < simulation_limit; i++)
            simulate_kernelng << <N_BLOCKS, R >> > (dev_g, R, dev_hypers, dev_rand_seeds);
        thrust::fill(p(dev_mask), p(dev_mask + R), 0);
        base = score;
    }
    if (id == 0)
        for (int i = 0; i < K; i++)
            cout << S[i] << "\t" << scores[i] << "\t" << time[i] << endl;
    //size_t free_after = get_free_vram();
    //cerr << std::fixed << std::setprecision(2) << "ALGO_VRAM_USE:" << float(free_before - free_after)*n_devices / 1024 / 1024 << "MB" << endl;
    exit(0);//CLEAN UP TAKES TOO LONG!
    cudaFree(dev_hypers); cudaFree(dev_mask);
    cudaFree(dev_active); if (id == 0) cudaFree(dev_fbuffer);
    cudaFree(dev_estimates); cudaFree(dev_q);
    cudaFree(dev_q_size); cudaFree(dev_q_next);
    cudaFree(dev_q_next_size); cudaFree(dev_in_q);
}

vector<graph_t> split_gpu(const graph_t& g, size_t R, int* rand_seeds, int n_splits) {
    int step = (R / n_splits);
    vector<graph_t> samples(n_splits);
#pragma omp parallel for
    for (int r = 0; r < R; r += step) {
        graph_t __g;
        __g.n = g.n;
        __g.m = 0;
        int* rands = rand_seeds + r;
        //cout << r << endl;
        // first pass to detect # of edges sampled
        for (int s = 0; s < g.n; s++) {
            //const auto hash_s = _mm_crc32_u32(0, s);
            for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
                const edge_t e = g.adj[it];
                //const auto hash_sv = _mm_crc32_u32(s, e.v) >> 1;
                const auto hash_sv = edge_hash(s, e.v);
                //for (int b = r; b < r + step; b++) {
                for (int b = 0; b < step; b++) {
                    int rnd = rands[b];
                    if (((rnd ^ hash_sv) < e.w)) {
                        __g.m++;
                        break;
                    }
                }
            }
        }
        __g.xadj = new size_t[g.n + 1];
        __g.adj = new edge_t[__g.m];
        int j = 0; __g.xadj[0] = 0;
        for (int s = 0; s < g.n; s++) {
            //const auto hash_s = _mm_crc32_u32(0, s);
            for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
                const edge_t e = g.adj[it];
                const auto hash_sv = edge_hash(s, e.v);
                //const auto hash_sv = _mm_crc32_u32(s, e.v) >> 1;
                //for (int b = r; b < r + step; b++) {
                for (int b = 0; b < step; b++) {
                    int rnd = rands[b];
                    if (((rnd ^ hash_sv) < e.w)) {
                        __g.adj[j++] = g.adj[it];
                        break;
                    }
                }
            }
            __g.xadj[s + 1] = j;
        }
        __g.m = j;
        samples[r / step] = __g;
    }
    return samples;
}
void meta(string filename, size_t R, size_t K, int n_devices) {
    //cout << "super" << endl;
    graph_t g = read_txt(filename);
    auto rands = get_rands(R);
    std::sort(rands.get(), rands.get() + R);
    auto gs = split_gpu(g, R, rands.get(), n_devices);
    vector<graph_t> dev_gs(n_devices);
    int device_count;
    cudaGetDeviceCount(&device_count);
#pragma omp parallel for
    for (int i = 0; i < n_devices; i++) {
        cudaSetDevice(i % device_count);
        dev_gs[i] = gs[i];
        dev_gs[i].xadj = devcpy(gs[i].xadj, gs[i].n + 1);
        dev_gs[i].adj = devcpy(gs[i].adj, gs[i].m);
        cudaDeviceSynchronize();
    }
#pragma omp parallel num_threads(n_devices)
    {
        int i = omp_get_thread_num();
        cudaSetDevice(i % device_count);
        superfuser_gpufx2<float>(dev_gs[i], K, R / n_devices, rands.get() + i * (R / n_devices), i, n_devices);
    }
}
int main(int argc, char* argv[]) {
    int K = 50, R = 64, c, blocksize = 32, n_devices = 1;
    bool directed = false;
    float p = 0.01, eps = 0.3, tr = 0.01, trc = 0.01;
    string method = "float", filename;
    ofstream out;
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <-M [float/char/super]> <-R [#MC=64]> <-e [threshold=0.01]> <-o [output='STDOUT']>";
        return -1;
    }
    for (int i = 1; i < argc; i++) {
        string s(argv[i]);
        if (s == "-K") K = atoi(argv[++i]);
        else if (s == "-R") R = atoi(argv[++i]);
        //else if (s == "-v") verbose = true;
        else if (s == "-g") n_devices = atoi(argv[++i]);
        else if (s == "-M") method = string(argv[++i]);
        else if (s == "-e") eps = atof(argv[++i]);
        else if (s == "-t") tr = atof(argv[++i]);
        else if (s == "-c") trc = atof(argv[++i]);
        else if (s == "-B") blocksize = atoi(argv[++i]);
        else if (s == "-o") { out.open(argv[++i]); std::cout.rdbuf(out.rdbuf()); }
        else filename = s;
    }
    meta(filename, R, K, n_devices);

    return 0;
}

