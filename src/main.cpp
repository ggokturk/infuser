
#include "common.h"
#include "newgreedy.h"

using namespace std;


int main(int argc, char* argv[]) {
	int K=50, R=256, c, blocksize= 32;
	bool directed = false;
	float p=0.01, eps=0.3; 
	string method="MixGreedy", filename;
	ofstream out;
	if (argc < 2)
		cerr << "Usage: " << argv[0] << " [-M <Method=[~MixGreedy/MixGreedy2]>] [-R <#MC=512>] [-B BATCHSIZE] [-e <threshold=0.01>] [-o <output='STDOUT'>]";

	for (int i = 1; i < argc; i++) {
		string s(argv[i]);
		if (s == "-K") K = atoi(argv[++i]);
		else if (s == "-R") R = atoi(argv[++i]);
		else if (s == "-M") method = string(argv[++i]);
		else if (s == "-e") eps = atoi(argv[++i]);
		else if (s == "-B") blocksize = atoi(argv[++i]);
		else if (s == "-o") { out.open(argv[++i]); std::cout.rdbuf(out.rdbuf()); }
		else filename = s;
	}
	graph_t g = read_bin(filename);
	std::cout << std::fixed << std::setprecision(2);

	if ((method) == "MixGreedy")
		newgreedy(g, K, R);
	else if ((method) == "MixGreedy2")
		newgreedy2(g, K, R, blocksize);
	return 0;
}
