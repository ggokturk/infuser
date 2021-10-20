# InFuseR

A Fast, Parallel Influence Maximization Software, using Fused Sampling and Memory Access Regularization. 

Currently support three algorithms; InFuseR(NewGreedy), HyperFuser(Sketch-based), SuperFuseR(GPU-specialized).

## Build Instructions
Building Infuser only requires AVX2/AVX512, and GCC 8.2 or better. Source code should be compatible with MSVC 2019 as well.
SuperFuseR requires CUDA 11.1+.

We provide a very simple Makefile for building from the source code. Executables will be placed under ./bin folder
```
cd infuser
make
```

## Usage

### Input Graph Format
For fast processing, InFuseR only accepts a text file structured as follows;

\# First line tells infuser to how many vertices and edges exists\
\# Then every other line has pairs of source_vertex_id target_vertex_id edge_probability\
\# We require source_vertex_ids to be sorted for performance and simplicity, using add_p automaticly sorts output files\
5 10 \
0 1 0.01 \
0 2 0.02 \
0 4 0.1 \
3 4 0.05 \
4 1 0.09 

### Preprocessing 

We provide a simple tool to add probabilities to SNAP edgelist format. The tool excepts multiple output at the same time.
```
python3 ./scripts/add_p --help
python3 ./scripts/add_p -p 0.01 -U 0.0 0.1 -pf constantprobfile.txt -Uf uniformdistfile.txt inputfile.txt
```
Example generates a graph with constant probility 0.01 named constantprobfile.txt and another file with uniform weights between 0 and 0.1 named uniformdistfile.txt 

### Running

```
./bin/infuser [-M method] [-K #seeds] [-R #MC] [-o output] edge_file
./bin/infuser [-g ndevices] [-K #seeds] [-R #MC] [-o output] edge_file
```
| Parameters |Description|  Default | 
|------------|-----| ---------|
| method | Using Infuser NewGreedy(Infuser) and HyperFuser(Sketch) methods are available, using SuperFuseR executable SingleGPU and MultiGPU methods are available. | HyperFuser, SingleGPU|
| #seed | Seed set size | 50|
| #MC   | Number of Monte-Carlo simulation performed, it must be a multiple of 32. | 256 |
| ndevices| Number of CUDA devices (only for Superfuser)| 1 | 
| output | Output file, leave empty for STDOUT | <empty> (STDOUT)|
### Example commands
```
./bin/infuser -M InFuseR -K 100 -R 256  ./amazon0302_0.01.txt
./bin/superfuser -g 1 -K 100 -R 256  ./amazon0302_0.01.txt
```

### Output

Output consist of tab seperated 4 elements per line; 
* Seed Vertex, 
* Time Spend, 
* Estimated Influence, 
* Number of Candidates Processed for NewGreedy, Sketch error rate for HyperFuser
```
14949   5.72	3.45	0
4429    11.42	3.45	1
33      16.74	3.45	2
10519	21.49	3.45	3
12771	26.14	3.45	4
8       30.62	3.45	5
481     34.93	3.45	7
5737    39.12	3.45	8
297     43.07	3.45	9
9106    46.64	3.45	12
...
```
### How to cite

You can use following BibTeX snippet to cite InFuseR;

```
@ARTICLE{9261128,
  author={G. {Göktürk} and K. {Kaya}},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Boosting Parallel Influence-Maximization Kernels for Undirected Networks With Fusing and Vectorization}, 
  year={2021},
  volume={32},
  number={5},
  pages={1001-1013},
  doi={10.1109/TPDS.2020.3038376}}
```
