#!/usr/bin/python3
from os import system
from os.path import exists,basename,splitext
import os
from itertools import product
import glob
import subprocess
import time
from random import randint, choice

command  = "{cmd}"
def experiment(expname, experiments, fmt, outfmt, testrun=False, **kwargs):
    print(":",experiments)
    for i,e in enumerate(experiments):
        print(e)
        args = dict(e,**kwargs)
        print(i+1,'/',len(experiments),end='\t')
        out = outfmt.format(**args)
        memout = out[:-4]+'.mem'
        
        if exists(out) and len(open(out).readlines())>=50:
            print("Existing experiment: {}".format(out))
        else:
            print(args)
            rcmd = fmt.format(out=out,memout=memout,**args)
            rcmd = rcmd.replace('(','\\(')
            rcmd = rcmd.replace(')','\\)')
            rcmd = command.format(cmd=rcmd)
            print(rcmd)
            if not testrun:
                system(rcmd)

K=['50']

base = '.'
path = base+'/snapdata'
outpath = base+'/fillresults'
files = list(sorted(glob.glob(path+"/*.txt")))
run_str = '{exe} -s {s} -M {method} -R {R} {file} > {out}'
exe=base+"/infuser/bin/infuser_fillrate"
methods = ['HyperFuser']
Rs = ['256']
experiments = [
    {
        'file':f,
        'dataset':splitext(basename(f))[0],
        'method':m,
        'e':0.3,
        'R': r,
        'K': k,
        's':s,
    } for f,k,m,r,s in product(files,K,methods,Rs,['0','1'])]
outfmt = '{outpath}/{dataset}_{method}_{K}_{e}_{R}_{s}.txt'
experiment('edges', experiments, run_str, outfmt, outpath=outpath, exe=exe)