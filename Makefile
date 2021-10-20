build:
	g++ -std=c++17 -funroll-loops -fopenmp -march=native -Ofast -mavx2 ./src/main.cpp -o ./bin/infuser
	g++ -std=c++17 -funroll-loops -fopenmp -march=native -Ofast -mavx2 -Dfillrate ./src/main.cpp -o ./bin/infuser_fillrate

buildcuda:
	nvcc -w --std=c++17 -use_fast_math -Xcompiler -fopenmp -Xcompiler -march=skylake -Xcompiler -mavx2 -gencode arch=compute_80,code=sm_80 ./src/gpu.cu -o ./bin/superfuser
