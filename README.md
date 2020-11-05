# InFuseR

A Fast, Parallel Influence Maximization Software, using Fused Sampling and Memory Access Regularization.

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
./bin/edgeutil [?directed] [-p prob] [-w ?weighted]  snap_file binary_file
```
| Parameters |Description|  Default | 
|------------|-----| ---------|
| ?directed        | Is the input network directed? | 0/1                           | 1
| -p        | How edge probabilies should be distributed, a real number gives you constant probabilies, N(p1,p2) gives you normal distribution with mean p1 variance p2, U(p1,p2) gives uniform distribution between p1 and p2, and w gives weighted cascade probabilities 1/dv  | 0.01  
| -w        | Is the input graph weighted, weighted snap file should have 3 elements on all lines; source target weight  | 0

### Running


```
./bin/infuser [-M method] [-K #seeds] [-R #MC] [-B blocksize] [-o output] binary_file
```
| Parameters |Description|  Default | 
|------------|-----| ---------|
| method | Only MixGreedy and MixGreedy2 is currently available, MixGreedy is parallel on vertices, MixGreedy2 is parallelized on monte-carlo simulations| MixGreedy|
| #seed | Seed set size | 50|
| #MC   | Number of Monte-Carlo simulation performed | 2048 | 
| blocksize | Only for MixGreedy2, how many blocks should be processed by single thread, Note that each block is sized as AVX vector lenght. AVX2:8, AVX512:16 | 16
| output | Output file, leave empty for STDOUT | <empty> (STDOUT)|

### Output

Output consist of tab seperated 4 elements per line; 
* Seed Vertex, 
* Time Spend, 
* Estimated Influence, 
* Number of Candidates Processed    
```
131     1.12    8.94     0
253     2.02    17.18    14
52453   1.12    21.84    21
131     1.12    24.25    35
...
```