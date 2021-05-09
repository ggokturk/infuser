#pragma once
#include "common.h"
double run_ic_64_s1_c(const graph_t& g, const uint32_t S, const size_t R, uint64_t* cache);

void scc(float scores[], const graph_t& g, const int R, const size_t BLOCKSIZE, float counts_ptr[], uint32_t labels_ptr[], int* rand_seeds) {
	for (int r = 0; r < R; r += BLOCKSIZE)
	{
		float* counts = &(counts_ptr[r * g.n]);
		uint32_t* labels = &(labels_ptr[r * g.n]);

		for (int i = 0; i < g.n; i++)
			for (int b = 0; b < BLOCKSIZE; b++)
				labels[i * BLOCKSIZE + b] = i;
		//labels[i] = _mm256_set1_epi32(i);
		bool cont = true;
		//vector<char> active(g.n, 1), nactive(g.n, 0);
		auto active = get_aligned<bool>(g.n), nactive = get_aligned<bool>(g.n);
		parfill(active.get(), active.get() + g.n, true);
		parfill(nactive.get(), nactive.get() + g.n, false);
		while (cont) {
			cont = false;
#pragma omp parallel for schedule(dynamic)
			for (int n = 0; n < g.n; n++) {
				if (!active[n]) continue;
				const uint32_t begin_p = g.xadj[n], end_p = g.xadj[n + 1];
				const uint32_t source = n;
				const auto label_s = labels[source];
				bool localflag = false;
				for (uint32_t i = begin_p; i < end_p; i++) {
					const edge_t e = g.adj[i];
					const auto target = e.v;
					const auto hash = (e.h);
					const auto w = (e.w);

					//if (!active[target]) continue;
					bool blockflag = false;

					for (int b = 0; b < BLOCKSIZE; b += 8) {
						const auto label_target = *(__m256i*) & (labels[target * BLOCKSIZE + b]);
						const auto label_source = *(__m256i*) & (labels[source * BLOCKSIZE + b]);
						const auto colormask = _mm256_cmpgt_epi32(label_target, label_source);
						const auto colors = _mm256_blendv_epi8(label_target, label_source, colormask);
						const auto hashvec = _mm256_set1_epi32(hash);
						const auto simhashvec = _mm256_xor_si256(hashvec, *(__m256i*) & (rand_seeds[b]));
						const auto sel = _mm256_cmpgt_epi32(_mm256_set1_epi32(w), simhashvec);
						const auto s = _mm256_blendv_epi8(label_source, colors, sel);
						const int flag = _mm256_movemask_epi8(_mm256_and_si256(sel, colormask));
						if (flag) {
							const auto t = _mm256_blendv_epi8(label_target, colors, sel);
							*(__m256i*)& (labels[target * BLOCKSIZE + b]) = t;

							//const auto s = _mm256_blendv_epi8(label_source, colors, sel);
							//*(__m256i*)& (labels[source * BLOCKSIZE + b]) = s;


							//nactive[target] = 1;
							blockflag = true;
						}
						
					}
					if (blockflag) {
						nactive[target] = true;
						localflag = true;
					}
				}
				if (localflag) {
					cont = true;
				}
			}
			swap(active, nactive);
			parfill(nactive.get(), nactive.get() + g.n, false);
			//__gnu_parallel::fill(nactive.begin(), nactive.end(), false);
		}
#pragma omp parallel for 
		for (int i = 0; i < g.n; i++) {
			for (int b = 0; b < BLOCKSIZE; b++) {
				const auto label = ((uint32_t*)labels)[(BLOCKSIZE * i) + b];
				counts[label * BLOCKSIZE + b]++;
			}
		}
#pragma omp parallel for 
		for (int i = 0; i < g.n; i++) {
			for (int b = 0; b < BLOCKSIZE; b++) {
				const auto label = ((uint32_t*)labels)[i * BLOCKSIZE + b];
				const auto count = counts[label * BLOCKSIZE + b];
				scores[i] += count;
			}
		}
	}
}

void get_s_cc(unsigned* s_cc, const graph_t& g, const std::vector<unsigned>& S, const size_t k, const int R, uint32_t labels_ptr[], float counts_ptr[], const int BLOCKSIZE) {
	if (k < 2) return;
	for (int r = 0; r < R; r += BLOCKSIZE) {
		float* counts = &(counts_ptr[r * g.n]);
		uint32_t* labels = &(labels_ptr[r * g.n]);

		int i = k - 2;
#pragma omp parallel for 
		//for (int i = 0; i < k - 1; i++) {
		for (int j = 0; j < BLOCKSIZE; j++) {
			s_cc[i * BLOCKSIZE + j] = labels[S[i] * BLOCKSIZE + j];
		}
		//}
	}
}

double run(const graph_t& g, const vector<unsigned>& S, const size_t k, const int R, const unsigned* s_cc, uint32_t labels_ptr[], float counts_ptr[], const int BLOCKSIZE) {
	//vector<unsigned> s_cc((k - 1) * R);
	double score = 0;
	for (int r = 0; r < R; r += BLOCKSIZE) {
		float* counts = &(counts_ptr[r * g.n]);
		uint32_t* labels = &(labels_ptr[r * g.n]);
		const uint32_t u = S[k - 1];
#pragma omp parallel for reduction(+:score)
		for (int b = 0; b < BLOCKSIZE; b++) {
			const auto label = labels[u * BLOCKSIZE + b];
			bool flag = true;
			for (int i = 0; i < k - 1; i++) {
				if (label == s_cc[i * BLOCKSIZE + b]) {
					flag = false;
				}
			}
			if (flag) {
				const auto count = counts[label * BLOCKSIZE + b];
				score += count;
			}
		}
	}

	return score / R;
}
void scc2(float* scores, const graph_t& g, const int R, const size_t BLOCKSIZE,
	float counts_ptr[], uint32_t labels_ptr[], int* rand_seeds) {

#pragma omp parallel for
	for (int r = 0; r < R; r += BLOCKSIZE)
	{
		float* counts = counts_ptr + (r * g.n);
		uint32_t* labels = labels_ptr + (r * g.n);
		int* rands = rand_seeds + r;
		for (int i = 0; i < g.n; i++)
			for (int b = 0; b < BLOCKSIZE; b++)
				labels[i * BLOCKSIZE + b] = i;

		bool cont = true;
		auto active = get_aligned<bool>(g.n), nactive = get_aligned<bool>(g.n);
		std::fill(active.get(), active.get() + g.n, true);
		std::fill(nactive.get(), nactive.get() + g.n, false);
		while (true) {
			//cont = false;
			//#pragma omp parallel for schedule(dynamic)
			for (int n = 0; n < g.n; n++) {
				if (!active[n]) continue;
				const uint32_t begin_p = g.xadj[n], end_p = g.xadj[n + 1];
				const uint32_t source = n;
				bool localflag = false;
				for (uint32_t i = begin_p; i < end_p; i++) {
					const edge_t e = g.adj[i];
					const auto target = e.v;
					const auto hash = (e.h);
					const auto w = (e.w);

					//if (!active[target]) continue;
					bool blockflag = false;
					//#define __AVX2__
#if !defined __AVX2__ && !defined __AVX512BW__ || defined NOVEC
					for (int b = 0; b < BLOCKSIZE; b++) {
						if ((hash ^ rands[b]) < w &&
							(labels[source * BLOCKSIZE + b] < labels[target * BLOCKSIZE + b])) {
							labels[target * BLOCKSIZE + b] = labels[source * BLOCKSIZE + b];
							blockflag = true;
						}
					}
#elif defined __AVX512BW__ && !defined NOAVX512
					const int vecsize = 16;
					typedef __m512i vec_t;
					for (int b = 0; b < BLOCKSIZE; b += vecsize) {
						const auto hashvec = _mm512_set1_epi32(hash);
						const auto simhashvec = _mm512_xor_si512(hashvec, *(vec_t*)&(rands[b]));
						const auto sel = _mm512_cmpgt_epu32_mask(_mm512_set1_epi32(w), simhashvec);
						if (!sel) continue;

						const auto label_target = *(vec_t*)&(labels[target * BLOCKSIZE + b]);
						const auto label_source = *(vec_t*)&(labels[source * BLOCKSIZE + b]);
						const auto colormask = _mm512_cmpgt_epu32_mask(label_target, label_source);
						const auto colors = _mm512_mask_blend_epi32(colormask, label_target, label_source);

						const auto s = _mm512_mask_blend_epi32(sel, label_source, colors);
						const int flag = sel & colormask;
						if (flag) {
							const auto t = _mm512_mask_blend_epi32(sel, label_target, colors);
							*(vec_t*)&(labels[target * BLOCKSIZE + b]) = t;
							blockflag = true;
						}
					}
#elif defined __AVX2__ 
					const int vecsize = 8;
					typedef __m256i vec_t;
					for (int b = 0; b < BLOCKSIZE; b += vecsize) {
						const auto hashvec = _mm256_set1_epi32(hash);
						const auto simhashvec = _mm256_xor_si256(hashvec, *(vec_t*)&(rands[b]));
						const auto sel = _mm256_cmpgt_epi32(_mm256_set1_epi32(w), simhashvec);
						if (_mm256_testz_si256(sel, sel)) continue;
						//int cmp = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(sel), _CMP_LT_OQ));
						//if (cmp)
						//	continue;

						const auto label_target = *(vec_t*)&(labels[target * BLOCKSIZE + b]);
						const auto label_source = *(vec_t*)&(labels[source * BLOCKSIZE + b]);
						const auto colormask = _mm256_cmpgt_epi32(label_target, label_source);
						const auto colors = _mm256_blendv_epi8(label_target, label_source, colormask);

						const auto s = _mm256_blendv_epi8(label_source, colors, sel);
						const int flag = !_mm256_testz_si256(sel, colormask);
						if (flag) {
							const auto t = _mm256_blendv_epi8(label_target, colors, sel);
							*(vec_t*)&(labels[target * BLOCKSIZE + b]) = t;
							blockflag = true;
						}
					}
#endif
					if (blockflag)
						nactive[target] = true;

				}
			}



			//if (blockflag) cont = true;
			bool flag = false;
			//#pragma omp parallel for
			for (int i = 0; i < g.n; i++)
				if (active[i]) {
					flag = true;
					break;
				}
			if (!flag)
				break;
			swap(active, nactive);
			parfill(nactive.get(), nactive.get() + g.n, false);
		}
		//#pragma omp parallel for
		for (int i = 0; i < g.n; i++) {
			for (int b = 0; b < BLOCKSIZE; b++) {
				const auto label = labels[BLOCKSIZE * i + b];
				counts[label * BLOCKSIZE + b]++;
			}
		}
	}
}

float get_scores(const size_t i, const size_t n, const size_t R, const size_t blocksize, uint32_t* labels_ptr, float* counts_ptr) {
	float acc = 0;
	//#pragma omp parallel for
	for (int r = 0; r < R; r += blocksize) {
		float* counts = counts_ptr + (r * n);
		uint32_t* labels = labels_ptr + (r * n);
		for (int b = 0; b < blocksize; b++) {
			const auto label = labels[i * blocksize + b];
			const auto count = counts[label * blocksize + b];
			acc += count;
		}
	}
	return acc;
}
void reset(const size_t i, const size_t n, const size_t R, const size_t blocksize, uint32_t* labels_ptr, float* counts_ptr) {
	//#pragma omp parallel for
	for (int r = 0; r < R; r += blocksize) {
		float* counts = counts_ptr + (r * n);
		uint32_t* labels = labels_ptr + (r * n);
		for (int b = 0; b < blocksize; b++) {
			const auto label = labels[i * blocksize + b];
			counts[label * blocksize + b] = 0;
		}
	}
}
void newgreedy2(const graph_t& g, const int K, const int R, const size_t blocksize) {
	t.reset();
	vector<unsigned> S, iteration(g.n, 0);
	vector<float> marginal_gain(g.n, 0);
	size_t size = R * g.n;
	float score = 0.0;
	auto scores = get_aligned<float>(g.n);
	auto counts = get_aligned<float>(size);
	parfill(counts.get(), counts.get() + size, 0.0f);
	parfill(scores.get(), scores.get() + g.n, 0.0f);
	auto labels = get_aligned<unsigned>(size);
	auto rand_seeds = get_aligned<int>(R);
	for (int i = 0; i < R; i++)
		rand_seeds[i] = __hash(g.n + i);
	std::sort(rand_seeds.get(), rand_seeds.get() + R);
	scc2(scores.get(), g, R, blocksize, counts.get(), labels.get(), rand_seeds.get());
#pragma omp parallel for 
	for (int i = 0; i < g.n; i++) {
		scores[i] = get_scores(i, g.n, R, blocksize, labels.get(), counts.get());
	}
	auto cmp = [&](uint32_t left, uint32_t right) {
		return (marginal_gain[left] < marginal_gain[right]); };
	std::priority_queue<uint32_t, vector<unsigned>, decltype(cmp)> q(cmp);

	for (int i = 0; i < g.n; i++) {
		marginal_gain[i] = scores[i] / (double)R;
		q.push(i);
	}
	int tries = 0;
	size_t level1 = omp_get_max_threads();
	vector <int> vertices(level1);
	while (S.size() < K) {
		const auto u = q.top();
		if (iteration[u] == S.size()) {
			S.push_back(u);
			score += marginal_gain[u];
			reset(u, g.n, R, blocksize, labels.get(), counts.get());
			printf("%d\t%.2f\t%.2f\t%d\n", u, score, t.elapsed(), tries);
			fflush(stdout);
			q.pop();
		}
		else {
			//q.pop()
			//marginal_gain[u] = get_scores(u, g.n, R, blocksize, labels.get(), counts.get())/R;
			//tries++;
			//iteration[u] = S.size(); // recently scored
			//q.pop()
			//q.push(u);
			int elements = std::min((int)q.size(), (int)level1);
			for (int i = 0; i < elements; i++) {
				tries++;
				int u = vertices[i] = q.top();
				q.pop();
			}
#pragma omp parallel for num_threads(level1)
			for (int i = 0; i < elements; i++) {
				const int u = vertices[i];
				marginal_gain[u] = get_scores(u, g.n, R, blocksize, labels.get(), counts.get()) / R;
				iteration[u] = S.size();
#pragma omp critical
				{
					q.push(u);
				}
			}
		}
	}
}

void newgreedy(const graph_t& g, const int K, const int R) {
	t.reset();
	vector<unsigned> S(K), iteration(g.n, 0);
	vector<float> marginal_gain(g.n, 0);
	uint32_t elems = (R / 64 * g.n);
	size_t size = R * g.n;
	float score = 0.0;
	auto scores = get_aligned<float>(g.n);
	auto counts = get_aligned<float>(size);
	parfill(counts.get(), counts.get() + size, 0.0f);
	parfill(scores.get(), scores.get() + g.n, 0.0f);
	auto labels = get_aligned<unsigned>(size);
	auto rand_seeds = get_aligned<int>(R);
	for (int i = 0; i < R; i++)
		rand_seeds[i] = __hash(-i - 1);
	std::sort(rand_seeds.get(), rand_seeds.get() + R);
	int blocksize = R;
	scc(scores.get(), g, R, blocksize, counts.get(), labels.get(), rand_seeds.get());
	auto cmp = [&](uint32_t left, uint32_t right) {
		return (marginal_gain[left] < marginal_gain[right]); };
	std::priority_queue<uint32_t, vector<unsigned>, decltype(cmp)> q(cmp);

	for (int i = 0; i < g.n; i++) {
		marginal_gain[i] = scores[i] / (double)R;
		q.push(i);
	}

	float max_for_k = -1;
	int tries = 0;
	//vector<unsigned> s_cc(K*R,-1);
	auto s_cc = get_aligned<unsigned>(K * R);
	for (size_t k = 0; k < K;) {
		const auto u = q.top();
		q.pop();
		S[k] = u;
		if (iteration[u] == k) {
			k++; // commit candidate
			//run_ic_64_s1_c(g, uvec{ u }, 1, R, cache);
			score += marginal_gain[u];

			get_s_cc(s_cc.get(), g, S, k + 1, R, labels.get(), counts.get(), blocksize);
			printf("%d\t%.2f\t%.2f\t%d\n", u, score, t.elapsed(), tries);
			fflush(stdout);
		}
		else {
			tries++;
			//std::copy(cache, cache + elems, visited);

			marginal_gain[u] = run(g, S, k + 1, R, s_cc.get(), labels.get(), counts.get(), blocksize);//run_ic_64_s1_c(g, uvec{ u }, 1, R, visited) - score;
			iteration[u] = k; // recently scored
			q.push(u);
		}
	}
}


