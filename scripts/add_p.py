#!/usr/bin/env python3
from glob import glob
import os
from random import uniform, normalvariate
import numpy as np
from multiprocessing import Pool
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description='Prepare datasets.')
skip = 0

parser.add_argument('file')
parser.add_argument('-u','--undirected', help='when input is undirected graph, adds reverse edges to output', required=False, action='store_true')
parser.add_argument('--skip',help='skips lines', type=int)
parser.add_argument('-p','--prob', nargs='+', help='constant probabilites', required=False, type=float)
parser.add_argument('-pf','--probfile', nargs='+', help='constant probablity output files', required=False)
parser.add_argument('-U','--uniform', nargs='+', help='uniform probabilites by pairs', required=False, type=float)
parser.add_argument('-Uf','--uniformfile', nargs='+', help='uniform distribution output files', required=False)
parser.add_argument('-N','--normal', nargs='+', help='normal probabilites by pairs', required=False, type=float)
parser.add_argument('-Nf','--normalfile', nargs='+', help='normal distribution output files', required=False)
parser.add_argument('-id','--fixids', help='fixes vertex ids when input file has non-consequtive ids', action='store_true')

args = parser.parse_args()
for key, value in args._get_kwargs():
    if value is not None:
        print(key,value)

if (args.prob is not None) + (args.probfile is not None) == 1 or (args.uniform is not None) + (args.uniformfile is not None) == 1 or (args.normal is not None) + (args.normalfile is not None) == 1:
    print("Error missing argument!")
    exit(-1)

if args.prob is not None and len(args.prob)!=len(args.probfile) or args.normal is not None and len(args.normal)/ len(args.normalfile) != 2 or args.uniform is not None and len(args.uniform)/ len(args.uniformfile) != 2 :
    print("RNG and output arguments mismatch!")
    exit(-1)


file = args.file

# print(file)
n=0
from tempfile import NamedTemporaryFile
f = NamedTemporaryFile(delete=True,mode='wt')
i = 0
lookup = {}
for line in open(file):
    if '#' not in line and '%' not in line:
        if skip>0:
            skip-=1
            continue
        split = line.split()
        if len(split) == 2:
            s = split[0]
            t = split[1]
            if args.fixids:
                if s not in lookup:
                    lookup[s] = i
                    i += 1
                s = lookup[s]
                if t not in lookup:
                    lookup[t] = i
                    i += 1
                t = lookup[t]

            n = max(n,int(s),int(t))
            if args.undirected:f.write(f'{s}\t{t}\n{t}\t{s}\n')
            else: f.write(f'{s}\t{t}\n')
lookup = None
f.flush()
os.system(f'sort {f.name} -k1,1n -k2,2n -S 2G -o {f.name}')
cf = NamedTemporaryFile(delete=True)
os.system(f"uniq -c {f.name} > {cf.name}")
os.system(f"awk '{{print $2, $3, $1}}' {cf.name} > {f.name}")
m = 0
with open(f.name,'r') as infile:
    for _ in infile:
        m += 1
n += 1
from numpy.random import random
rand_size = 10000
if (args.prob is not None):
    for i in range(len(args.prob)):
        prob = args.prob[i]
        outfile = args.probfile[i]
        with open(outfile,'w') as out, open(f.name,'r') as infile :
            out.write(f'{n} {m}\n')
            for line in infile:
                s, t, c = line.split()
                c = float(c)
                w = 1.0 - ((1.0 - prob)**c)
                w = round(w, 4)
                out.write(f'{s}\t{t}\t{w}\n')


if args.uniform is not None:
    for i in range(0, len(args.uniform), 2):
        a = args.uniform[i]
        b = args.uniform[i+1] if len(args.uniform)>(i+1) else (args.uniform[i]/2)
        outfile = args.uniformfile[i]
        with open(outfile,'w') as out, open(f.name,'r') as infile:
            out.write(f'{n} {m}\n')
            for line in infile:
                s, t, c = line.split()
                c = float(c)
                prob = np.clip(uniform(a, b),a,b)
                w = 1.0 - ((1.0 - prob)**c)
                w = round(w, 4)
                out.write(f'{s}\t{t}\t{w}\n')



if args.normal is not None:
    for i in range(0, len(args.normal), 2):
        a = args.normal[i]
        b = args.normal[i+1] if len(args.normal)>(i+1) else (args.normal[i]/2)
        outfile = args.normalfile[i]
        with open(outfile,'w') as out, open(f.name,'r') as infile:
            out.write(f'{n} {m}\n')
            for line in infile:
                s, t, c = line.split()
                c = float(c)
                prob = np.clip(normalvariate(a, b),0.0,1.0)                
                w = 1.0 - ((1.0 - prob)**c)
                w = round(w, 4)
                out.write(f'{s}\t{t}\t{w}\n')
