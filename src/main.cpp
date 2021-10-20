#include "common.h"
#include "newgreedy.h"
#include "mixgreedy.h"
#include "sketch.h"
using namespace std;

void oracle(const graph_t& g, size_t K, const int R) {
	auto cache_ptr = make_unique<char[]>(R * g.n);
	auto rand_seeds = get_rands(R);
	float score = 0;
	for (int i; cin >> i;)
		score = run_ic_vertpar(g, i, R, rand_seeds.get(), cache_ptr.get());
	cout << score << endl;
}

int main(int argc, char* argv[]) {
	int K = 50, R = 256, c, blocksize = 32;
	bool directed = false, sorted = true;
	float p = 0.01, eps = 0.3, tr = 0.01, trc = 0.02;
	string method = "HyperFuser", filename;
	ofstream out;
	if (argc < 2){
		cerr << "Usage: " << argv[0] << " -M [Method=[~MixGreedy/HyperFuser]] -R [#MC="<<R<<"] -e [threshold="<<tr<<"] -o [output file(optional)]\n";
		exit(-1);
	}

	for (int i = 1; i < argc; i++) {
		string s(argv[i]);
		if (s == "-K") K = atoi(argv[++i]);
		else if (s == "-R") R = atoi(argv[++i]);
		else if (s == "-M") method = string(argv[++i]);
		else if (s == "-e") eps = atof(argv[++i]);
		else if (s == "-t") tr = atof(argv[++i]);
		else if (s == "-s") sorted = atoi(argv[++i]);
		else if (s == "-c") trc = atof(argv[++i]);
		else if (s == "-o") { out.open(argv[++i]); std::cout.rdbuf(out.rdbuf()); }
		else filename = s;
	}
	graph_t g = read_txt(filename);
	std::for_each(method.begin(), method.end(), [](char& c) {c = ::tolower(c);});
	std::cout << std::fixed << std::setprecision(2);
	if (method == "infuser")
		newgreedy(g, K, R, sorted);
	else if (method == "hyperfuser")
		hyperfuser(g, K, R, eps, tr, trc, sorted);
	else if (method == "oracle")
		oracle(g, K, R);
	return 0;
}
