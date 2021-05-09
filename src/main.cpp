#include "common.h"
#include "newgreedy.h"
#include "mixgreedy.h"
#include "sketch.h"
using namespace std;

void oracle(const graph_t& g, size_t K, const int R) {
	auto cache_ptr = make_unique<char[]>(R * g.n);
	auto rand_seeds = get_rands(R);//get_aligned<int>(R);
	//for (int i = 0; i < R; i++) rand_seeds[i] = __hash(-i - 1);
	float score = 0;
	for (int i; cin >> i;)
		score = run_ic_vertpar(g, i, R, rand_seeds.get(), cache_ptr.get());
	cout << score << endl;
}

int main(int argc, char* argv[]) {
	int K = 50, R = 256, c, blocksize = 32;
	bool directed = false;
	float p = 0.01, eps = 0.3, tr = 0.01, trc = 0.02;
	string method = "HyperFuser", filename;
	ofstream out;
	if (argc < 2)
		cerr << "Usage: " << argv[0] << " <-M [Method=[~MixGreedy/HyperFuser]]> <-R [#MC=512]> <-e [threshold=0.01]> <-o [output='STDOUT']>";

	for (int i = 1; i < argc; i++) {
		string s(argv[i]);
		if (s == "-K") K = atoi(argv[++i]);
		else if (s == "-R") R = atoi(argv[++i]);
		else if (s == "-M") method = string(argv[++i]);
		else if (s == "-e") eps = atof(argv[++i]);
		else if (s == "-t") tr = atof(argv[++i]);
		else if (s == "-c") trc = atof(argv[++i]);
		else if (s == "-B") blocksize = atoi(argv[++i]);
		else if (s == "-o") { out.open(argv[++i]); std::cout.rdbuf(out.rdbuf()); }
		else filename = s;
	}
	graph_t g = read_bin(filename);
	std::cout << std::fixed << std::setprecision(2);

	if ((method) == "NewGreedy")
		newgreedy(g, K, R);
	else if (method == "HyperFuser"){
		hyperfuser(g, K, R, eps, tr, trc);
	}
	else if (method == "oracle"){
		oracle(g, K, R);
	}
	return 0;
}
