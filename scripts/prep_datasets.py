#!/usr/bin/python3
from os import system
from os.path import exists,basename,splitext

from itertools import product
import glob
import subprocess
import time
from random import randint

slurm = "sbatch -N 1 -n 1 -c 8 -p short --time=4:00:00 --wrap=\'pwd;which python3;{cmd}\'"
slurm = "{cmd}"
base = '.'
path = base+'/data'
outpath = base+'/snapdata'
undirected_datasets = ['amazon0302','ca-HepPh','ca-HepTh']
reid = ['ca-HepPh','ca-HepTh','ego-twitter']


directed = [0, 1]
ps = ['0.005']
# ps = ['0.01', '0.1']
ns = [['0.05','0.025']]
us = [ ['0', '0.1'] ]
datasets = list(sorted(glob.glob(path+"/*.txt")))
print(datasets)
# exe="./bin/edgeutil"
for i in datasets:
    dataset = splitext(basename(i))[0]
    undirected = 'ungraph' in list(basename(i).split('.')) or dataset in undirected_datasets
    exe = "python3 infuser/scripts/add_p.py" 
    outf = ' '.join([f'{outpath}/{dataset}_{p}.txt' for p in ps])
    nf = ' '.join([f'{outpath}/{dataset}_N{p[0]}.txt' for p in ns])
    uf = ' '.join([f'{outpath}/{dataset}_U{p[1]}.txt' for p in us])
    u = " ".join(sum(us,[]))
    n = " ".join(sum(ns,[]))
    fixid = '-id' if basename(i).split('.')[0] in reid else '' 
    run_str = f'{exe} {i} -p {" ".join(ps)} {fixid} {"-u" if undirected else ""} -pf {outf}'

    # run_str = f'{exe} {i} -p {" ".join(ps)} -U {u} -N {n} {fixid} {"-u" if undirected else ""} -pf {outf} -Uf {uf} -Nf {nf}'
    # print(run_str)
    run_str = run_str.replace('(','\\(')
    run_str = run_str.replace(')','\\)')
    rcmd = slurm.format(cmd=run_str)
    print(rcmd)
    system(rcmd)


