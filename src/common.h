#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <memory>
#include <algorithm>
#include <random>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <climits>
#include <cstring>
#if defined (_MSC_VER)
#include <nmmintrin.h>
#endif
#include <algorithm>
#if _MSC_VER >= 1910
#include <execution>
#else
#include <parallel/algorithm>
#include <parallel/numeric>
#include <stdlib.h> 
void* _aligned_malloc(size_t size, size_t alignment) {
	void* p;
	if (posix_memalign(&p, alignment, size))
		return NULL;
	return p;
}
void _aligned_free(void* p) { free(p); }

#endif

using std::vector;
using std::ifstream;
using std::ofstream;
using std::cin;
using std::cout;
using std::unique_ptr;
using std::string;
using std::pair;
using std::stringstream;
using std::make_pair;
using std::endl;
using std::cerr;
using std::tuple;
using std::get;


const uint32_t HASHMASK = INT32_MAX;

struct Timer{
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	std::chrono::time_point<clock_> beg_;
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	double elapsed() const { return std::chrono::duration_cast<second_> (clock_::now() - beg_).count(); }
} t;

struct edge_t { unsigned s,v; signed w, h; };
struct graph_t {
	unsigned* xadj, * t; size_t n = 0, m = 0;
	edge_t* adj;
};
//uint64_t popcnt(const uint64_t* data, const size_t n) {
//	uint64_t result = 0;
//#pragma unroll
//	for (int i = 0; i < n; i++) {
//		result += _mm_popcnt_u64(data[i]);
//	}
//	return result;
//}
uint64_t popcnt(const char* ptr, const size_t size) {

	uint64_t i = 0;
	uint64_t cnt = 0;

	for (; i < size - size % 8; i += 8)
		cnt += _mm_popcnt_u64(*(const uint64_t*)(ptr + i));
	for (; i < size; i++)
		cnt += _mm_popcnt_u64(ptr[i]);

	return cnt;
}
uint64_t parpopcnt(const char* ptr, const size_t size) {
	uint64_t cnt = 0;
#pragma omp parallel for reduction(+:cnt)
	for (long long i = 0; i < size - size % 8; i += 8)
		cnt += _mm_popcnt_u64(*(const uint64_t*)(ptr + i));
	for (long long i = size - size % 8; i < size; i++)
		cnt += _mm_popcnt_u64(ptr[i]);
	return cnt;
}


inline uint32_t __hash(uint64_t h) {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdL;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53L;
	h ^= h >> 33;
	return h & HASHMASK;//FFFF;
}
inline uint64_t __hash64(uint64_t h) {
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdL;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53L;
	h ^= h >> 33;
	return h;//FFFF;
}

inline uint32_t __hash(const uint32_t x, const uint32_t y) {
	uint64_t h = (x > y) ? (((uint64_t)y) << 32) | x : (((uint64_t)x) << 32) | y;
	return __hash(h);
}
template <typename T>
void cpy(T* buf, T* end, T* out) {
	const int len = end - buf;
#pragma omp parallel for
	for (int i = 0; i < len; i++)
		out[i] = buf[i];
}

template <typename T>
void parfill(T* buf, T* end, T pattern) {
	const int len = end - buf;
#pragma omp parallel for
	for (int i = 0; i < len; i++)
		buf[i] = pattern;
}

template <typename T>
void parfill(vector<T> & buf, T pattern) {
	const int len = buf.size();
#pragma omp parallel for
	for (int i = 0; i < len; i++)
		buf[i] = pattern;
}



template<class T>
unique_ptr<T[], decltype(&_aligned_free)> get_aligned(size_t elems, size_t align = 64){
	auto* ptr = static_cast<T*>(_aligned_malloc(sizeof(T) * elems, align));
	std::fill(ptr, ptr + elems, 0);
	return unique_ptr<T[], decltype(&_aligned_free)>(ptr, &_aligned_free);
}


graph_t read_file(std::string filename, bool directed, float p) {
  graph_t g;
	ifstream in(filename);
	uint32_t s, v, i = 0, j = 0;
	string line;
	getline(in,line);
	vector<pair<pair<uint32_t, uint32_t>, float> > pairs;
	while (getline(in,line)) {
		stringstream ss(line);
		float my_p = p;
		ss >> s >> v >> my_p;
		pairs.push_back(make_pair(make_pair(s, v),my_p));
		if (!directed)
			pairs.push_back(make_pair(make_pair(v, s),my_p) );
		if (s >= g.n) g.n = s + 1;
		if (v >= g.n) g.n = v + 1;
	}
#if _MSC_VER >= 1910
	sort(pairs.begin(), pairs.end());
#else
	__gnu_parallel::sort(pairs.begin(), pairs.end());
#endif
	for (auto it = cbegin(pairs), last = cend(pairs); it != last; g.m++)
		it = std::upper_bound(it, last, *it);
	g.xadj = new unsigned[g.n + 1];
	g.adj = new edge_t[g.m];
	uint32_t ind = 0;
	for (int i = 0; i < pairs.size();) {
		auto e = get<0>(pairs[i]);
		auto w = get<1>(pairs[i]);
		auto a = get<0>(e),b= get<1>(e);
		for (; ind <= a; ind++)
			g.xadj[ind] = j;
		int count = 1;
		for (i = i + 1; i < pairs.size() && get<0>(pairs[i - 1]) == get<0>(pairs[i]); i++){
			w = w + (1.0-w)*(get<1>(pairs[i]));
		}
		float prob = std::max(0.0f, std::min(w, 1.0f)); //CLAMP
		int hash = directed? (int)__hash((uint64_t(a)<<32)|b) : (int)__hash(a, b);		
		g.adj[j++] = { ind, b,  int(prob * double(HASHMASK)), hash };
	}
	for (; ind <= g.n; ind++)
		g.xadj[ind] = j;
	return g;
}

graph_t read_bin(string filename) {
	graph_t g;
	ifstream rf(filename, std::ios::out | std::ios::binary);
	if (!rf) {
		cerr << "Cannot open file!" << endl;
		return g;
	}
	int mode;
	rf.read((char*)&g.n, sizeof(g.n));
	rf.read((char*)&g.m, sizeof(g.m));
	g.xadj = new unsigned[g.n + 1];
	g.adj = new edge_t[g.m];
	rf.read((char*)g.xadj, size_t(g.n + 1) * sizeof(uint32_t));
	rf.read((char*)g.adj, size_t(g.m) * sizeof(edge_t));
	return g;
}
#include <assert.h>
#define isbitset(x,i) ((x[i>>3] & (1<<(i&7)))!=0)
#define setbit(x,i) x[i>>3]|=(1<<(i&7));
#define CLEARBIT(x,i) x[i>>3]&=(1<<(i&7))^0xFF;
template <class T>
vector<graph_t> split(const graph_t& g, size_t batch_size, T& rand_seeds) {
	vector<graph_t> samples;
#pragma omp parallel for
	for (int sid = 0; sid < rand_seeds.size(); sid++) {
		graph_t __g = g;
		__g.m = 0;
		// first pass to detect # of edges sampled
		for (int s = 0; s < g.n; s++) {
			for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
				const edge_t e = g.adj[it];
				for (int r = 0; r < batch_size; r++) {
					unsigned rnd = rand_seeds[sid][r];
					if (((rnd ^ e.h) <= e.w)) 
					{
						__g.m++;
						break;
					}
				}
			}
		}
		__g.xadj = new unsigned[g.n + 1];
		__g.adj = new edge_t[__g.m];
		int j = 0; __g.xadj[0] = 0;
		for (int s = 0; s < g.n; s++) {
			for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
				const edge_t e = g.adj[it];
				for (int r = 0; r < batch_size; r++) {
					unsigned rnd = rand_seeds[sid][r];
					if (((rnd ^ e.h) <= e.w)) 
					{
						__g.adj[j++] = g.adj[it];
						break;
					}
				}
			}
			__g.xadj[s + 1] = j;

		}
		__g.m = j;
#pragma omp critical
		samples.push_back(__g);
		/* */
		cout << float(__g.m) / (g.m) << endl;
		//cout << memcmp(g.adj, __g.adj, sizeof(edge_t) * g.m) << endl;
		//cout << memcmp(g.xadj, __g.xadj, sizeof(int) * g.n) << endl;
		//for (int i = 0; i < 10; i++)
		//	cout << g.xadj[i] << " " << __g.xadj[i] << endl;
	}
	return samples;
}
//
vector<graph_t> split(const graph_t& g, size_t R, int* rand_seeds, int n_splits) {
	int step = (R / n_splits);
	vector<graph_t> samples;
//#pragma omp parallel for
	for (int r = 0; r < R; r += step) {
		graph_t __g;
		__g.n = g.n;
		__g.m = 0;
		// first pass to detect # of edges sampled
		for (int s = 0; s < g.n; s++) {
			for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
				const edge_t e = g.adj[it];
				for (int b = r; b < r + step; b++) {
					int rnd = rand_seeds[b];
					if (((rnd ^ e.h) < e.w)) {
						__g.m++;
						break;
					}
				}
			}
		}
		__g.xadj = new unsigned[g.n + 1];
		__g.adj = new edge_t[__g.m];
		int j = 0; __g.xadj[0] = 0;
		for (int s = 0; s < g.n; s++) {
			for (int it = g.xadj[s]; it < g.xadj[s + 1]; it++) {
				const edge_t e = g.adj[it];
				for (int b = r; b < r + step; b++) {
					int rnd = rand_seeds[b];
					if (((rnd ^ e.h) < e.w)) {
						__g.adj[j++] = g.adj[it];
						break;
					}
				}
			}
			__g.xadj[s + 1] = j;
		}
		__g.m = j;
#pragma omp critical
		samples.push_back(__g);
	}
	return samples;
}
unique_ptr<int[], decltype(&_aligned_free)> get_rands(size_t size) {
	std::default_random_engine e1(42);
	std::uniform_int_distribution<int> uniform_dist(0, INT_MAX);

	auto rand_seeds = get_aligned<int>(size);
	for (int i = 0; i < size; i++)
		rand_seeds[i] = uniform_dist(e1);
	return move(rand_seeds);
}