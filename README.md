# InFuseR

A Fast, Parallel Influence Maximization Software, using Fused Sampling and Memory Access Regularization. 

Currently support two algorithms; InFuseR(NewGreedy), HyperFuser(Sketch-based).

## Build Instructions
Building Infuser only requires AVX2/AVX512, and GCC 8.2 or better. Source code should be compatible with MSVC 2019 as well.

We provide a very simple Makefile for building from the source code.
```
mkdir bin
make
```

## Usage

For fast processing, InFuseR only accepts a custom binary format that is directly maps to internal data-structures.

### Preprocessing 

We provide a simple tool to convert SNAP format to our binary format.
```
./bin/edgeutil [-D ?directed] [-p prob] [-w ?weighted]  snap_file binary_file
```
| Parameters |Description|  Default | 
|------------|-----| ---------|
| -D        | Is the input network directed? | 0/1                           | 1
| -p        | How edge probabilies should be distributed, a real number gives you constant probabilies, N(p1,p2) gives you normal distribution with mean p1 variance p2, U(p1,p2) gives uniform distribution between p1 and p2, and w gives weighted cascade probabilities 1/dv  | 0.01  
| -w        | Is the input graph weighted, weighted snap file should have 3 elements on all lines; source target weight  | 0

### Running


```
./bin/infuser [-M method] [-K #seeds] [-R #MC]  [-o output] binary_file
```
| Parameters |Description|  Default | 
|------------|-----| ---------|
| method | Only NewGreedy(Infuser) and HyperFuser(Sketch) is currently available.| HyperFuser|
| #seed | Seed set size | 50|
| #MC   | Number of Monte-Carlo simulation performed, it must be a multiple of 32. | 256 | 
<!-- | blocksize | Only for MixGreedy2, how many blocks should be processed by single thread, Note that each block is sized as AVX vector lenght. AVX2:8, AVX512:16 | 16 -->
| output | Output file, leave empty for STDOUT | <empty> (STDOUT)|
### Example command
```
./bin/edgeutil -D 0 -p 0.01 ./com-orkut.ungraph.txt ./bindata/orkut_001.bin 
./bin/infuser -M HyperFuser -K 100 -R 256  ./bindata/orkut_001.bin

```

### Output

Output consist of tab seperated 4 elements per line; 
* Seed Vertex, 
* Time Spend, 
* Estimated Influence, 
* Number of Candidates Processed for NewGreedy, Sketch error rate for HyperFuser    
```
14949	5.72	3.45	0
4429	11.42	3.45	1
33	16.74	3.45	2
10519	21.49	3.45	3
12771	26.14	3.45	4
8       30.62	3.45	5
481	34.93	3.45	7
5737	39.12	3.45	8
297	43.07	3.45	9
9106	46.64	3.45	12
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
