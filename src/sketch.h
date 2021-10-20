#pragma once
#include "common.h"

int maxsum(char* start, char* end, char* mask) {
	__m256i acc = _mm256_setzero_si256();
	for (char* i = start; i < end; i += 32, mask += 32) {
		const auto a = _mm256_max_epi8(*(__m256i*)i, *(__m256i*)mask);
		acc = _mm256_add_epi16(acc, _mm256_add_epi16(
			_mm256_cvtepi8_epi16(_mm256_extractf128_si256(a, 0)),
			_mm256_cvtepi8_epi16(_mm256_extractf128_si256(a, 1))));
	}
	const int sum = (_mm256_extract_epi16(acc, 0) + _mm256_extract_epi16(acc, 1) +
		_mm256_extract_epi16(acc, 2) + _mm256_extract_epi16(acc, 3) +
		_mm256_extract_epi16(acc, 4) + _mm256_extract_epi16(acc, 5) +
		_mm256_extract_epi16(acc, 6) + _mm256_extract_epi16(acc, 7) +
		_mm256_extract_epi16(acc, 8) + _mm256_extract_epi16(acc, 9) +
		_mm256_extract_epi16(acc, 10) + _mm256_extract_epi16(acc, 11) +
		_mm256_extract_epi16(acc, 12) + _mm256_extract_epi16(acc, 13) +
		_mm256_extract_epi16(acc, 14) + _mm256_extract_epi16(acc, 15));
	return sum;
}
int maxsum_novec(char* start, char* end, char* mask) {
	int acc = 0;
	for (char* i = start; i < end; i++) {
		acc+=max(*i, *(mask++));
	}
	return acc;
}
inline __m256i expand_bits_to_bytes(uint32_t x){
	__m256i xbcast = _mm256_set1_epi32(x);
	__m256i shufmask = _mm256_set_epi64x(
		0x0303030303030303, 0x0202020202020202,
		0x0101010101010101, 0x0000000000000000);
	__m256i shuf = _mm256_shuffle_epi8(xbcast, shufmask);
	__m256i andmask = _mm256_set1_epi64x(0x8040201008040201);
	__m256i isolated_inverted = _mm256_and_si256(shuf, andmask);
	return _mm256_cmpeq_epi8(isolated_inverted, _mm256_setzero_si256());
}
//
//template <bool check_blocked>
//void simulate_sketch(const graph_t& g, const size_t R, char* hypers, int* rands, char* visited, float threshold) {
//	const int ITER_LIMIT = 20;
//	float total = 0;
//	vector<char> active(g.n, 1);
//	vector<char> nactive(g.n, 0);
//
//	for (int iter = 0; iter < ITER_LIMIT; iter++) {
//#pragma omp parallel for schedule(dynamic, 8192)// reduction(+:active_count) // reduction(or:cont)
//		for (int u = 0; u < g.n; u++) {
//			const uint32_t begin_p = g.xadj[u], end_p = g.xadj[u + 1];
//			bool flag = false;
//			auto hash_u = _mm_crc32_u32(0, u);
//			for (uint32_t i = begin_p; i < end_p; i++) {
//				const edge_t e = g.adj[i];
//				if (!active[e.v])
//					continue;
//				auto hash_uv = _mm_crc32_u32(hash_u, e.v) >> 1;
//				for (int r = 0; r < R; r ++) {
//					int rnd = rands[r];
//					if (((rnd ^ hash_uv) <= e.w) && (hypers[u * R + r] < hypers[e.v * R + r])) {
//						hypers[u * R + r] = hypers[e.v * R + r];
//						flag = true;
//					}
//				}
//			}
//			if (flag)
//				nactive[u] = true;
//
//		}
//		int64_t active_count = 0;
//		//#pragma omp parallel for reduction(+:active_count)
//		for (int i = 0; i < g.n; i++) {
//			active_count += nactive[i];
//		}
//		float active_rate = float(active_count) / g.n;
//		if (active_rate <= threshold) {
//			break;
//		}
//		swap(active, nactive);
//		parfill(nactive, char(0));
//	}
//}
//
#ifdef fillrate
uint64_t fr_filled = 0, fr_taken=0; 
#endif

template <bool check_blocked>
void simulate_sketch(const graph_t& g, const size_t R, char* hypers, int* rands, char* visited, float threshold) {
	const int ITER_LIMIT = 20;
	float total = 0;
	vector<char> active(g.n, 1);
	vector<char> nactive(g.n, 0);
	for (int iter = 0; iter < ITER_LIMIT; iter++) {

#pragma omp parallel for schedule(dynamic, 8192)
		for (int u = 0; u < g.n; u++) {
			const uint32_t begin_p = g.xadj[u], end_p = g.xadj[u + 1];
			bool flag = false;
			auto hash_u = _mm_crc32_u32(0, u);
			for (uint32_t i = begin_p; i < end_p; i++) {
				const edge_t e = g.adj[i];
				if (!active[e.v])
					continue;
				for (int r = 0; r < R; r += 32) {
					auto hash = _mm_crc32_u32(hash_u, e.v)>>1;//__hash(u, e.v)>>1;
					const auto edge_hash8 = _mm256_set1_epi32(hash),
						threshold = _mm256_set1_epi32(g.adj[i].w);
					unsigned packed = 0;
					for (int b = 0; b < 32; b += 8) {
						const auto prob_hash = _mm256_xor_si256(*(__m256i*)(&rands[r + b]), edge_hash8);
						packed |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(
							_mm256_cmpgt_epi32(threshold, prob_hash)))) << (b);//check shift latency
					}
					if(packed==0)
						continue;
#ifdef fillrate
#pragma omp atomic
					fr_taken ++;
					uint64_t ones = double(_mm_popcnt_u32(packed));
#pragma omp atomic
					fr_filled += ones;
#endif

					if constexpr (check_blocked)
						packed &= ~visited[u * (R/8) + r/8];// shift 3 bits

						//packed &= ~visited[(r >> 5) * g.n + u];
					const auto cmp = expand_bits_to_bytes(~packed);
					const auto cmp2 = _mm256_cmpgt_epi8(*(__m256i*) & hypers[e.v * R + r], *(__m256i*) & hypers[u * R + r]);
					const auto sel = _mm256_and_si256(cmp, cmp2);
					*(__m256i*)& hypers[u * R + r] = _mm256_blendv_epi8(
						*(__m256i*) & hypers[u * R + r], *(__m256i*) & hypers[e.v * R + r], sel);
					const auto vcmp = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(sel), _CMP_NEQ_OQ);
					flag |= _mm256_movemask_ps(vcmp);
				}
			}
			if (flag) nactive[u] = true;
		}
		int64_t active_count = 0;
		int step = 100;
#pragma omp parallel for reduction(+:active_count)
		for (int i = 0; i < g.n; i+=step) {
			active_count += nactive[i];
		}
		double active_rate = double(active_count)*double(step) / g.n;
		if (active_rate <= threshold) {
			break;
		}
		swap(active, nactive);
		parfill(nactive, char(0));
	}
}

double run_ic_vertpar(const graph_t& g, const uint32_t S, const size_t R, int32_t* rand_seeds, char* visited) {

	const uint32_t n = g.n, BLOCKSIZE = 64;
	const auto Roffset = R / 8;
	std::fill(visited + S * Roffset, visited + (S + 1) * Roffset, UINT8_MAX);
	const int ITER_LIMIT = 50;
	vector<char> active(g.n, 0);
	vector<char> nactive(g.n, 0);
	active[S] = 1;
	for (int iter = 0; iter < ITER_LIMIT; iter++) {
		bool cont = false;
#pragma omp parallel for schedule(dynamic, 8192) reduction(+:cont)// reduction(+:active_count) // 
		for (int u = 0; u < g.n; u++) {
			if (!active[u])
				continue;
			bool gflag = false;
			auto hash_u = _mm_crc32_u32(0, u);
			const uint32_t begin_p = g.xadj[u], end_p = g.xadj[u + 1];
			for (uint32_t i = begin_p; i < end_p; i++) {
				const edge_t e = g.adj[i];
				bool flag = false;
				const uint32_t v = e.v;
				const auto edge_hash8 = _mm256_set1_epi32(_mm_crc32_u32(hash_u,g.adj[i].v)>>1),
					threshold = _mm256_set1_epi32(g.adj[i].w);

				for (int r = 0; r < R; r += BLOCKSIZE) {
					const auto roffset = r / 8;
					const auto threshold = _mm256_set1_epi32(g.adj[i].w);
					const uint64_t curr_sims = *((uint64_t*)&(visited[u * (Roffset)+(roffset)]));

					uint64_t packed = 0;
					for (int b = 0; b < BLOCKSIZE; b += 8) {
						const auto prob_hash = _mm256_xor_si256(*(__m256i*)(&rand_seeds[r + b]), edge_hash8);
						packed |= uint64_t(_mm256_movemask_ps(_mm256_castsi256_ps(
							_mm256_cmpgt_epi32(threshold, prob_hash)))) << (b);
					}
					const auto will_visit = packed & curr_sims & ~(*(uint64_t*)&visited[v * (Roffset)+(roffset)]);
					if (will_visit) {
						*(uint64_t*)&visited[v * (Roffset)+(roffset)] |= will_visit;
						flag = true;
					}

				}
				if (flag) {
					nactive[v] = true;
					cont = true;
				}

			}
		}
		if (!cont)
			break;
		std::swap(active, nactive);
		parfill(nactive, char(0));
	}
	float score = parpopcnt((char*)visited, g.n * R / 8);
	return  score/R;
}

#ifdef _MSC_VER
#define __builtin_clzll __lzcnt64
#endif
void hyperfuser(const graph_t& g, const int K, const size_t R, float eps, float tr, float trc, bool sorted) {
	auto hypers = get_aligned<char>(R * g.n);
	auto mask = get_aligned<char>(R);
	std::fill(mask.get(), mask.get() + R, 0);
	auto rand_seeds = get_rands(R);
	if (sorted)
		std::sort(rand_seeds.get(), rand_seeds.get() + R);

#pragma omp parallel for
	for (int i = 0; i < g.n; i++)
		for (int j = 0; j < R; j++)
			hypers[i * R + j] = __builtin_clzll(__hash64(~(i * R + j + R)));
	//return;

	auto visited = get_aligned<char>(R * g.n / 8);
	t.reset();
	simulate_sketch<false>(g, R, hypers.get(), rand_seeds.get(), visited.get(), trc);
	vector<uint64_t> S;
	std::vector <int> estimates(g.n);

	float score = 0, base = 0;
	while (S.size() < K) {
#pragma omp parallel for 
		for (int i = 0; i < g.n; i++)
			estimates[i] = maxsum_novec(hypers.get() + (i * R), hypers.get() + ((i + 1ull) * R), mask.get());
		uint64_t s = std::distance(estimates.begin(), max_element(estimates.begin(), estimates.end()));
		float max_s = powf(2.0f, float(estimates[s]) / R);

		score = run_ic_vertpar(g, s, R, rand_seeds.get(), visited.get());

		float mg = score - base;
		S.push_back(s);
		float err = (max_s - mg) / mg;
		float ratio = abs(max_s - mg) / score;
		cout << s << "\t" << score << "\t" << t.elapsed() << "\t" << err * 100 << '%'
		// #ifdef fillrate
		// << " " <<fr_filled << "/" << fr_taken<<"="<<(fr_filled/fr_taken)<<" "
		// #endif 
		<< endl;
		if (err > eps && ratio > tr) {
#pragma omp parallel for
			for (int i = 0; i < g.n; i++)
				for (int j = 0; j < R; j++)
					hypers[i * R + j] = __builtin_clzll(__hash64(~(i * R + j + R)));

			for (auto i : S)
				memset(&hypers[i * R], 0, R);
			simulate_sketch<true>(g, R, hypers.get(), rand_seeds.get(), visited.get(), trc);
			std::fill(mask.get(), mask.get() + R, 0);
			base = score;
		}
		else {
			for (int j = 0; j < R; j += 32)
				*(__m256i*)& (mask[j]) = _mm256_max_epi8(*(__m256i*) & (mask[j]), *(__m256i*) & (hypers[s * R + j]));
		}
	}
#ifdef fillrate
	cout <<  "fillrate:" << (double(fr_filled) / (double(fr_taken)*sizeof(int)*8))  <<endl;
#endif

}
void fill_hypers_cpu(char* hypers, int n, int R, int batch_size, int offset) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < batch_size; j++)
			hypers[i * batch_size + j] = __builtin_clzll(__hash64(~(i * R + (offset +j) + R)));
}
void superfuser(const graph_t& g, const int K, const size_t R, float eps, float tr, float trc) {
	const auto num_streams = 4;
	const auto batch_size = R / num_streams;

	vector<unique_ptr<char[], decltype(&_aligned_free)>> hypers;
	for (int s = 0; s < num_streams; s++)
		hypers.push_back(move(get_aligned<char>(batch_size*g.n)));

	/*auto rands = get_aligned<int>(R);
	for (int i = 0; i < R; i++)
		rands[i] = __hash(-i - 1);*/
	auto rands = get_rands(R);
	std::sort(rands.get(), rands.get() + R);
	//for (int i = 0; i < R; i++)
	//	cout << int(rands[i]) << " ";
	//cout << endl;
	vector<unique_ptr<int[], decltype(&_aligned_free)>> rand_seeds;
	for (int s = 0; s < num_streams; s++) {
		rand_seeds.push_back(move(get_aligned<int>(batch_size)));
		memcpy(
			rand_seeds[s].get(),
			rands.get() + s * batch_size, 
			sizeof(int) * batch_size);
	}

	//auto rand_seeds = get_aligned<int>(R);
	/*auto hashes = get_aligned<char>(R * g.n);
	*/
	//auto mask = get_aligned<char>(R);
	vector<unique_ptr<char[], decltype(&_aligned_free)>> mask;
	for (int s = 0; s < num_streams; s++) {
		mask.push_back(move(get_aligned<char>(batch_size)));
		memset(mask[s].get(), 0, batch_size);
	}
	
	auto gs = split(g, R, rands.get(), num_streams);
	//free(g.adj); free(g.xadj);

	for (int s = 0; s < num_streams; s++) {
		fill_hypers_cpu(hypers[s].get(),g.n,R, batch_size, s*batch_size);
	}

	//auto visited = get_aligned<char>(R * g.n / 8);

	vector<unique_ptr<char[], decltype(&_aligned_free)>> visiteds;
	for (int s = 0; s < num_streams; s++) {
		visiteds.push_back(move(get_aligned<char>(batch_size / 8 *g.n)));
		memset(visiteds[s].get(), 0, batch_size/8*g.n);
	}
	//int node = 131;
	//for (int s = 0; s < num_streams; s++) {
	//	for (int i = 0; i < batch_size; i++)
	//		cout << int(hypers[s][node * batch_size + i]) << " ";
	//}
	//cout << endl;

	t.reset();
	for (int s = 0; s < num_streams; s++) {
		simulate_sketch<false>(gs[s], batch_size, hypers[s].get(), rand_seeds[s].get(), visiteds[s].get(), 0);
	}
	//int node = 0;
	//for (int s = 0; s < num_streams; s++) {
	//	for (int i = 0; i < batch_size; i++)
	//		cout << int(hypers[s][node * batch_size + i]) << " ";
	//}
	//cout << endl;
	//for (int s = 0; s < num_streams; s++) {
	//	for (int i = 0; i < batch_size; i++)
	//		cout << int(hypers[s][131 * batch_size + i]) << " ";
	//}cout << endl;
	//return;

	vector<uint64_t> S;
	//std::vector <int> estimates(g.n);
	vector<vector<int>> estimates(num_streams, vector<int>(g.n,0));
	float score = 0, base = 0;
	while (S.size() < K) {
		for (int sid = 0; sid < num_streams; sid++) {
#pragma omp parallel for
			for (int i = 0; i < g.n; i++)
				estimates[sid][i] = maxsum_novec(hypers[sid].get() + (i * batch_size), hypers[sid].get() + ((i + 1ull) * batch_size), mask[sid].get());
		}
#pragma omp parallel for
		for (int i = 0; i < g.n; i++) {
			int acc = 0;
			for (int sid = 0; sid < num_streams; sid++) {
				acc += estimates[sid][i];
			}
			estimates[0][i] = acc;
		}
		//for (int s = 0; s < num_streams; s++) {
		//	for (int i = 0; i < batch_size; i++)
		//		cout << int(hypers[s][131 * batch_size + i]) << " ";
		//}cout << endl;
		//return;
		
		uint64_t s = std::distance(estimates[0].begin(), max_element(estimates[0].begin(), estimates[0].end()));
		float max_s = powf(2.0f, float(estimates[0][s]) / R);
		//cout << s << "!" << max_s << endl;
		score = 0;
		for (int sid = 0; sid < num_streams; sid++) {
			score += run_ic_vertpar(gs[sid], s, batch_size, rand_seeds[sid].get(), visiteds[sid].get());
		}
		score /= num_streams;
		//cout << (s) << endl;
		float mg = score - base;
		S.push_back(s);
		float err = (max_s - mg) / mg;
		float ratio = abs(max_s - mg) / score;
		cout << s << "\t" << score << "\t" << t.elapsed() << "\t" << err * 100 << '%' << endl;
		if (err > eps && ratio > tr) {
//#pragma omp parallel for
			for (int s = 0; s < num_streams; s++) {
				fill_hypers_cpu(hypers[s].get(), g.n, R, batch_size, s * batch_size);
			}
			for (auto i : S)
				for (int s = 0; s < num_streams; s++) 
					memset(&hypers[s][i * batch_size], 0, batch_size);
			for (int s = 0; s < num_streams; s++)
				simulate_sketch<true>(gs[s], batch_size, hypers[s].get(), rand_seeds[s].get(), visiteds[s].get(), trc);
			for (int s = 0; s < num_streams; s++)
				std::fill(mask[s].get(), mask[s].get() + batch_size, 0);
			base = score;
		}
		else {
			for (int sid = 0; sid < num_streams; sid++)
				for (int b = 0; b < batch_size; b++)
					mask[sid][b] = max(mask[sid][b], hypers[sid][s * batch_size + b]);
			//for (int j = 0; j < R; j += 32)
			//	*(__m256i*)& (mask[j]) = _mm256_max_epi8(*(__m256i*) & (mask[j]), *(__m256i*) & (hypers[s * R + j]));
		}
	}
}
