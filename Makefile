build:
	g++ -std=c++17 -funroll-loops -fopenmp -march=native -Ofast ./src/main.cpp -o ./bin/infuser
	g++ -std=c++17 -funroll-loops -fopenmp -march=native -Ofast ./src/edgeutil.cpp -o ./bin/edgeutil

