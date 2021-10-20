#include "common.h"
using namespace std;

struct FastSet {
	vector<int> items;
	vector<bool> membership;
	FastSet(size_t n) {
		items.reserve(n);
		membership.resize(n);
	}
	void add(int e) {
		if (!membership[e]) {
			items.push_back(e);
			membership[e] = true;
		}
	}
	void clear() {
		for (auto e : items)
			membership[e] = false;
		items.clear();
	}
	vector<int>& get() {
		return items;
	}
};

void sample_ic_wide_forward(float* scores, const graph_t& g, const size_t R) {
	fill(scores, scores + g.n, 0);
	const size_t BLOCKSIZE = 8, m = g.xadj[g.n];
#pragma omp parallel for schedule(guided)
	for (int r = 0; r < R; r += BLOCKSIZE) {
		__m256i rand_seeds;
		auto counts = get_aligned<float>(g.n * BLOCKSIZE);
		fill(counts.get(), counts.get() + g.n * BLOCKSIZE, 0);
		auto labels_int = get_aligned<unsigned>(g.n * BLOCKSIZE);
		fill(labels_int.get(), labels_int.get() + g.n * BLOCKSIZE, 0);
		__m256i* labels = (__m256i*) labels_int.get();
		for (int b = 0; b < BLOCKSIZE; b++) {
			((uint32_t*)(&rand_seeds))[b] = __hash(-(r + b));
		}
		for (int i = 0; i < g.n; i++)
			labels[i] = _mm256_set1_epi32(i);
		bool cont = true;
		int tries;
		vector<bool> active(g.n, 1);
		vector<bool> nactive(g.n, 0);

		for (int iter = 0; cont && iter < 40; iter++) {
			cont = false;
			for (uint32_t n = 0; n < g.n; n++) {
				if (!active[n]) continue;
				active[n] = 0;
				const uint32_t begin_p = g.xadj[n], end_p = g.xadj[n + 1];
				const uint32_t source = n;
				auto label_s = labels[source];
				bool localflag = false;
				const auto hash_n = _mm_crc32_u32(0, n);
				for (uint32_t i = begin_p; i < end_p; i++) {
					const uint32_t target = g.adj[i].v;
					// if (target<source) break;
					const auto colormask = _mm256_cmpgt_epi32(labels[target], label_s);
					const auto colors = _mm256_blendv_epi8(labels[target], label_s, colormask);
					const auto hashvec = _mm256_set1_epi32(_mm_crc32_u32(hash_n, g.adj[i].v)>>1);
					const auto simhashvec = _mm256_xor_si256(hashvec, rand_seeds);
					const auto sel = _mm256_cmpgt_epi32(_mm256_set1_epi32(g.adj[i].w), simhashvec);
					const int flag = _mm256_movemask_epi8(_mm256_and_si256(sel, colormask));
					if (flag) {
						const auto t = _mm256_blendv_epi8(labels[target], colors, sel);
						labels[target] = t;
						const auto s = _mm256_blendv_epi8(labels[source], colors, sel);
						labels[source] = s;
						nactive[target] = 1;
						// nactive[source] = 1;
						localflag = true;
					}
				}
				if (localflag) {
					cont = true;
				}

			}
			if (cont)
				swap(active, nactive);
		}
		for (int i = 0; i < g.n; i++) {
			//__m256i label = labels[i];
			for (int j = 0; j < BLOCKSIZE; j++) {
				auto label = labels_int[i * BLOCKSIZE + j];
				counts[BLOCKSIZE * label + j]++;
				//				counts[BLOCKSIZE * (((int*)&label)[j]) + j]++;
			}
		}
#pragma omp critical
		for (int i = 0; i < g.n; i++) {
			//const auto ccl = labels[i];
			for (int b = 0; b < BLOCKSIZE; b++) {
				auto label = labels_int[i * BLOCKSIZE + b];
				scores[i] += counts[BLOCKSIZE * label + b];
			}
			//scores[i] += counts[BLOCKSIZE * ((uint32_t*)(&ccl))[b] + b];
		}
	}
}


double run_ic_64_s1_c(const graph_t& g, const uint32_t S, const size_t R, uint64_t* cache) {
	double score = 0;
	const uint32_t n = g.n, BLOCKSIZE = 64, blockiter = BLOCKSIZE / 8;
#pragma omp parallel for  reduction(+:score)
	for (int r = 0; r < R; r += BLOCKSIZE) {
		__m256i rand_seeds[blockiter]; /*__attribute__((aligned(64)))*/;
		auto visited = &cache[size_t(r / BLOCKSIZE) * g.n];
		auto q_ptr = make_unique<uint32_t[]>(n);
		uint32_t* q = q_ptr.get();
		auto in_q_ptr = make_unique<bool[]>(n);
		bool* in_q = in_q_ptr.get();
		for (int b = 0; b < BLOCKSIZE; b++)
			((uint32_t*)(&rand_seeds))[b] = __hash(-(r + b + 1));
		uint32_t q_begin = 0, q_end = 1;
		visited[q[0] = S] = UINT64_MAX;//blocksize bits
		in_q[S] = true;

		while (q_begin != q_end) {
			const uint32_t s = q[q_begin++], begin_pos = g.xadj[s], end_pos = g.xadj[s + 1];
			if (q_begin == g.n) q_begin = 0;
			in_q[s] = false;
			const auto hash_s = _mm_crc32_u32(0, s);
			const uint64_t curr_sims = visited[s];
			for (uint32_t j = begin_pos; j < end_pos; j++) {
				const uint32_t v = g.adj[j].v;
				const auto edge_hash8 = _mm256_set1_epi32(_mm_crc32_u32(hash_s,g.adj[j].v)>>1),
					threshold = _mm256_set1_epi32(g.adj[j].w);
				uint64_t packed = 0;
				for (int b = 0; b < blockiter; b++) {
					packed <<= 8;
					const auto prob_hash = _mm256_xor_si256(rand_seeds[b], edge_hash8);
					packed |= _mm256_movemask_ps(_mm256_castsi256_ps(
						_mm256_cmpgt_epi32(threshold, prob_hash)));
				}
				const uint64_t will_visit = packed & curr_sims & (~visited[v]);
				if (will_visit) {
					visited[v] |= will_visit;
					if (!in_q[v]) {
						q[q_end++] = v;
						if (q_end == g.n) q_end = 0;
						in_q[v] = true;
					}
				}
			}
		}
		score += popcnt((char*)visited, g.n);
	}
	return score / R;
}
//
//double run_ic_64_s1_c(const graph_t& g, const uint32_t S, const size_t R,
//	unsigned* qs, bool* in_qs, const uint64_t* visiteds, uint64_t* cache, vector<FastSet>& sets) {
//	double score = 0;
//	const uint32_t n = g.n, BLOCKSIZE = 64, blockiter = BLOCKSIZE / 8;
//#pragma omp parallel for  reduction(+:score)
//	for (int r = 0; r < R; r += BLOCKSIZE) {
//		int tid = omp_get_thread_num();
//		FastSet& set = sets.at(tid);
//		__m256i rand_seeds[blockiter]; /*__attribute__((aligned(64)))*/;
//		const auto visited = &visiteds[size_t(r / BLOCKSIZE) * g.n];
//		auto nvisited = &cache[size_t(r / BLOCKSIZE) * g.n];
//		unsigned* q = qs + (tid * g.n);
//		bool* in_q = in_qs + (tid * g.n);
//		for (int b = 0; b < BLOCKSIZE; b++)
//			((uint32_t*)(&rand_seeds))[b] = __hash(-(r + b + 1));
//		uint32_t q_begin = 0, q_end = 1;
//		nvisited[q[0] = S] = UINT64_MAX;//blocksize bits
//		in_q[S] = true;
//
//		while (q_begin != q_end) {
//			const uint32_t s = q[q_begin++];
//			const uint32_t begin_pos = g.xadj[s];
//			const uint32_t end_pos = g.xadj[s + 1];
//			if (q_begin == g.n) q_begin = 0;
//			in_q[s] = false;
//			set.add(s);
//			const uint64_t curr_sims = visited[s] | nvisited[s];
//			for (uint32_t j = begin_pos; j < end_pos; j++) {
//				const uint32_t v = g.adj[j].v;
//				const auto edge_hash8 = _mm256_set1_epi32(g.adj[j].h),
//					threshold = _mm256_set1_epi32(g.adj[j].w);
//				uint64_t packed = 0;
//				for (int b = 0; b < blockiter; b++) {
//					packed <<= 8;
//					const auto prob_hash = _mm256_xor_si256(rand_seeds[b], edge_hash8);
//					packed |= _mm256_movemask_ps(_mm256_castsi256_ps(
//						_mm256_cmpgt_epi32(threshold, prob_hash)));
//				}
//				const uint64_t will_visit = packed & curr_sims & (~(visited[v] | nvisited[v]));
//				if (will_visit) {
//					//score += _mm_popcnt_u64(will_visit);
//					nvisited[v] |= will_visit;
//					if (!in_q[v]) {
//						q[q_end++] = v;
//						if (q_end == g.n) q_end = 0;
//						in_q[v] = true;
//					}
//				}
//			}
//		}
//		// score += popcnt(nvisited, g.n);
//		for (auto& s : set.items) {
//			score += _mm_popcnt_u64(nvisited[s]);
//			nvisited[s] = 0l;
//		}
//		set.clear();
//		// fill(nvisited,nvisited+g.n,0);
//	}
//	return score / R;
//}
void mixgreedy(const graph_t& g, const int K, const int R) {
	t.reset();
	vector<signed> S(K), iteration(g.n, -1);
	vector<float> marginal_gain(g.n, 0);
	uint32_t elems = (R / 64 * g.n);

	auto scores = get_aligned<float>(g.n);
	float score = 0.0;
	// FIRST CANDIDATE IS CALCULATED HERE!
	sample_ic_wide_forward(scores.get(), g, R);

	auto cmp = [&](uint32_t left, uint32_t right) {
		return (marginal_gain[left] < marginal_gain[right]); };
	priority_queue<uint32_t, vector<unsigned>, decltype(cmp)> q(cmp);
	for (int i = 0; i < g.n; i++) {
		marginal_gain[i] = scores[i] / (double)R;
		q.push(i);
	}
	float max_for_k = 0;
	int max_u = 0;
	int tries = 0;
	int tries_for_k = 0;
	auto threads = omp_get_max_threads();
	auto qs = get_aligned<unsigned>(threads * g.n);
	auto in_qs = get_aligned<bool>(threads * g.n);
	auto cache = get_aligned<uint64_t>(elems);
	auto visiteds = get_aligned<uint64_t>(elems);
	vector<FastSet> sets(threads, FastSet(g.n));
	for (int k = 0; k < K;) {
		int u = q.top();
		q.pop();
		S[k] = u;
		if (iteration[u] == k) {
			k++; // commit candidate
			tries_for_k = 0;
			max_for_k = 0;
			marginal_gain[u] = run_ic_64_s1_c(g, u, R, visiteds.get()) - score;
			score += marginal_gain[u];
			printf("%d\t%.2f\t%.2f\t%d\n", u, score, t.elapsed(), tries);
			fflush(stdout);
		}
		else {
			tries++;
			tries_for_k++;
			auto prev = marginal_gain[u];
			marginal_gain[u] = run_ic_64_s1_c(g,  u, R, visiteds.get()) - score;
			//marginal_gain[u] = run_ic_64_s1_c(g, u, R, qs.get(), in_qs.get(), visiteds.get(), cache.get());
			iteration[u] = k; // recently scored
			q.push(u);
		}
	}
}
//void mixgreedy_nested(const graph_t& g, const int K, const int R) {
//	vector<signed> S(K), iteration(g.n, -1);
//	vector<float> marginal_gain(g.n, 0);
//	uint32_t elems = (R / 64 * g.n);
//	int level1, level2;
//	string threadstr = std::getenv("OMP_NUM_THREADS");
//	if (threadstr.find(',') == string::npos) {
//		cout << "please set an nested OMP_NUM_THREADS env parameter such as OMP_NUM_THREADS=4,2"; exit(-1);
//	}
//	else {
//		level1 = atoi(threadstr.c_str());
//		level2 = atoi(threadstr.substr(threadstr.find(',') + 1).c_str());
//	}
//	// cout << "LEVELS:" << level1 << ":" << level2 << endl;
//	auto scores = get_aligned<float>(g.n);
//	float score = 0.0;
//	t.reset();
//	// FIRST CANDIDATE IS CALCULATED HERE!
//	sample_ic_wide_forward(scores.get(), g, R);
//
//	auto cmp = [&](uint32_t left, uint32_t right) {
//		return (marginal_gain[left] < marginal_gain[right]); };
//	priority_queue<uint32_t, vector<unsigned>, decltype(cmp)> q(cmp);
//	for (int i = 0; i < g.n; i++) {
//		marginal_gain[i] = scores[i] / (double)R;
//		q.push(i);
//	}
//	float max_for_k = 0;
//	int max_u = 0;
//	int tries = 0;
//	int tries_for_k = 0;
//	auto threads = level1 * level2;
//	auto qs = get_aligned<unsigned>(threads * g.n);
//	auto in_qs = get_aligned<bool>(threads * g.n);
//	auto cache = get_aligned<uint64_t>(level1 * elems);
//	auto visiteds = get_aligned<uint64_t>(elems);
//	vector<vector<FastSet> > sets(level1, vector<FastSet>(level2, FastSet(g.n)));
//	omp_set_nested(1);
//	bool found = false;
//	vector <int> vertices(level1);
//	for (int k = 0; k < K;) {
//		int u = q.top();
//		if (iteration[u] == k) {
//			k++; // commit candidate
//			marginal_gain[u] = run_ic_64_s1_c(g, u, R, visiteds.get()) - score;
//			q.pop();
//			score += marginal_gain[u];
//			printf("%d\t%.2f\t%.2f\t%d\n", u, score, t.elapsed(), tries);
//			fflush(stdout);
//			continue;
//		}
//
//		for (int i = 0; i < level1; i++) {
//			tries++;
//			int u = vertices[i] = q.top();
//			q.pop();
//		}
//#pragma omp parallel for num_threads(level1)
//		for (int i = 0; i < level1; i++) {
//			const int u = vertices[i];
//			int thread_id_start = i * level2;
//
//			marginal_gain[u] = run_ic_64_s1_c(g, u, R,
//				qs.get() + thread_id_start * g.n,
//				in_qs.get() + thread_id_start * g.n,
//				visiteds.get(), cache.get() + i * elems, sets[i]);
//			iteration[u] = k;
//#pragma omp critical
//			q.push(u);
//		}
//	}
//}