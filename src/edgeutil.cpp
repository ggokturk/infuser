#include "common.h"
using namespace std;

void convert_to_binary(graph_t g, string outfile) {
	ofstream wf(outfile, ios::out | ios::binary);
	if (!wf) {
		cerr << "Cannot open output file!" << endl;
		return;
	}
	// cout << g.n << "\t" << g.m << endl;
	wf.write(reinterpret_cast<char*>(&g.n), sizeof(g.n));
	wf.write(reinterpret_cast<char*>(&g.m), sizeof(g.m));
	wf.write(reinterpret_cast<char*>(g.xadj), sizeof(uint32_t) * (g.n + 1));
	wf.write(reinterpret_cast<char*>(g.adj), sizeof(edge_t) * (g.m));
	//for (int i = 0; i <= g.n; i++) {
	//	wf << g.xadj[i];
	//}
	//for (int j = 0; j < g.m; j++) {
	//	wf << g.adj[j].s << g.adj[j].v << g.adj[j].w << g.adj[j].h;
	//}
	wf.close();
}

void convert_to_snap(graph_t g, string outfile) {
	ofstream wf(outfile, ios::out);
	if (!wf) {
		cerr << "Cannot open output file!" << endl;
		return;
	}
	// wf << g.n <<'\t' <<g.m;
	
	for (int i = 0; i < g.n; i++) {
		for (int j = g.xadj[i]; j < g.xadj[i+1]; j++) {
			wf << g.adj[j].s << ' ' << g.adj[j].v << ' ' << float(g.adj[j].w)/(INT_MAX)  << '\n';
		}
	}
	wf.close();
}

graph_t read_file(string filename, bool directed, string randarg) {
	graph_t g;
	float p = 0.01;
	ifstream in(filename);
	uint32_t s, v, i = 0, j = 0;
	string line;
	getline(in, line);
	vector<int> degrees;
	vector<vector<pair<uint32_t, float> >> adjlist;

	bool wc = randarg == "w", normal = randarg[0] == 'N', uniform = randarg[0] == 'U';
	float p0 = 0, p1 = 0;
	std::default_random_engine generator;
	std::normal_distribution<float> normal_dist;
	std::uniform_real_distribution<float> uniform_dist;

	if (normal || uniform)
		p0 = stof(randarg.substr(2, randarg.find(','))),
		p1 = stof(randarg.substr(randarg.find(',') + 1, randarg.length() - randarg.find(',') - 1));
	else if (!wc)
		p = stof(randarg);
	if (normal) normal_dist.param(std::normal_distribution<float>::param_type(p0, p1));
	if (uniform) uniform_dist.param(std::uniform_real_distribution<float>::param_type(p0, p1));

	while (getline(in, line)) {
		if (line.length() > 0 && (line.at(0) == '#' || line.at(0) == '%'))
			continue;
		stringstream ss(line);
		float my_p = p;
		if (normal)
			my_p = normal_dist(generator);
		else if (uniform)
			my_p = uniform_dist(generator);
		else if (wc)
			my_p = 1.0f;
		ss >> s >> v;
		if(adjlist.size() < (max(s, v) + 1)){
			adjlist.resize(max(s, v)+1, vector<pair<uint32_t, float>>());
		}
		bool exists = false;
		for (auto& e : adjlist[s])
			if (e.first == v) {
				if (!wc)
					e.second = e.second + (1.0 - e.second) * (my_p);
				else
					e.second += my_p;
				exists = true;
			}
		if (!exists)
			adjlist[s].push_back(make_pair(uint32_t(v), my_p));
		if (!directed)
			for (auto& e : adjlist[v])
				if (e.first == s) {
					if (!wc) e.second = e.second + (1.0 - e.second) * (my_p);
					else e.second += my_p;
					exists = true;
				}
		if (!exists)
			adjlist[v].push_back(make_pair(uint32_t(s), my_p));

		g.n = std::max(size_t(std::max(s,v)+1), g.n);
	}
	if (wc) {
		degrees.resize(g.n, 0);
		for (auto es : adjlist)
			for (auto v : es)
				degrees[v.first] += v.second;
	}

	g.xadj = new unsigned[g.n + 1];
	g.xadj[0] = 0;
	for (int i = 0; i < g.n; i++)
		g.xadj[i + 1] = g.xadj[i] + adjlist[i].size();
	g.m = g.xadj[g.n];
	g.adj = new edge_t[g.m];

	for (int i = 0; i < g.n; i++) {
		for (int j = 0; j < adjlist[i].size(); j++) {
			uint32_t a = i, b = adjlist[i][j].first;
			float w = adjlist[i][j].second;
			if (wc) {
				if (degrees[b] != 0)
					w = (1.0f *w) / degrees[b];
				else w = 1.0;
			}
			float prob = std::max(0.0f, std::min(w, 1.0f)); //CLAMP
			int hash = directed ? (int)__hash((uint64_t(a) << 32) | b) : (int)__hash(a, b);
			g.adj[(g.xadj[i] + j)] = { a, b,  int(prob * double(HASHMASK)), hash };

		}
	}
	return g;
}

int main(int argc, char* argv[]) {
	bool directed = true, weighted = false;
	string filename, outfile, format="bin", p = "0.01";
	
	ofstream out;
	if (argc < 3)
		cerr << "Usage: " << argv[0] << " <-p 0.01/N(0,1)/U(0,1) > <-D directed=[!0!,1]> <-f !bin >  input output ";

	for (int i = 1; i < argc; i++) {
		string s(argv[i]);
		if (s == "-D") directed = !!(atoi(argv[++i]));
		else if (s == "-p") p = string(argv[++i]);
		else if (s == "-w") weighted = true;
		else if (s == "-f") format = string(argv[++i]);
		else if (filename=="") filename = argv[i];
		else outfile = argv[i];
	}
	graph_t g = read_file(filename, directed, p);
	if (format=="bin")
		convert_to_binary(g, outfile);
	else if (format=="edgelist")
		convert_to_snap(g, outfile);
	return 0;
}
