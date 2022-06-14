#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import re
import itertools
import collections
import datetime

coldate, coltime, colnval = 0, 1, 2 # hardwired columns
valfmt = '{:.1f}' # valfmt.format(x)
checkalarm = False
debug = 1
header, dataobj = None, None

labels = ('date', 'time', 'seconds', 'cputemp', 'temperature', 'pressure', 'humidity', 'light', 'oxidised', 'reduced', 'nh3')

def strval(v): return valfmt.format(v/vnum) if type(v) is float else v

class DataPoint:
    def __init__(self):
        self.sums, self.num = 0, 0
        self.allvals = []
        self.wavg = None
    def add(self, val):
        if debug > 3: print(f'add {val}', file=sys.stderr)
        self.allvals.append(val)
        self.num += 1
        self.sums += val
        if self.wavg is None: self.wavg = val
        self.wavg +=  (val - self.wavg) * 2 / self.num
    def lazy_add(self, val):
        if debug > 3: print(f'lazy_add {val}', file=sys.stderr)
        self.allvals.append(val)
    def setup(self):
        self.lastval = None
        self.maxdiff = 0
        xv = None
        for v in self.allvals: # float or int
            try:
                self.sums += v
                if xv is not None and (dv := abs(v - xv)) > self.maxdiff: self.maxdiff = dv
                xv = v
            except (ValueError) as e:
                self.sums = None # not valid any more
        self.num += len(self.allvals)
    def avg(self):
        return self.sums/self.num
    def predict(self): 
        return 2 * self.allvals[-1] - self.allvals[-2]
    def err(self, v):
        if len(self.allvals) < 1: return None
        return v < self.allvals[-1] - self.maxdiff or v > self.allvals[-1] + self.maxdiff
    def err2(self, v):
        if len(self.allvals) < 2: return None
        return (2 * self.allvals[-2] - self.allvals[-1] < v) != (v < 3 * self.allvals[-1] - 2 * self.allvals[-2])

class DataDict(dict):
    def __missing__(self, key):
        self[key] = v = DataPoint()
        return v
    def __init__(self, name):
        self.omit, self.name = None, name
    def add(self, key, val):
        if debug > 3: print(f'add {key} {val}', file=sys.stderr)
        self[key].add(val)
    def lazy_add(self, key, val):
        if debug > 3: print(f'lazy_add {key} {val}', file=sys.stderr)
        self[key].lazy_add(val)

class DataFull(dict):
    def __init__(self, name):
        self.omit, self.name = None, name
    
## add daily weekly mode

def printrow(row): print('\t'.join(row))

def rowkey(row): return re.match('[0-2]?[0-9]:[0-5][0-9]', row[coltime]).group(0)

def timeofrow(row):
    return row[coldate]
    return rowkey(row)

def splitrow(row):
    return row[:3], row[3:]

def readcsv(f):
    for line in f:
        yield line.strip().split('\t')

def train(filenames):
    global header, dataobj
    for fn in filenames:
        if debug > 1: print(f'File {fn}', file=sys.stderr)
        with open(fn) as f:
            for row in readcsv(f):
                kk, vv = splitrow(row)
                if dataobj is None:
                    header = vv
                    dataobj = [DataDict(h) for h in vv]
                    if debug > 2: print(f'{len(header)}, {len(dataobj)}', file=sys.stderr)
                else:
                    tt = timeofrow(kk)
                    for v, obj in zip(vv, dataobj):
                        if obj.omit: continue
                        try:
                            obj[tt].lazy_add(float(v))
                        except (ValueError) as e:
                            if debug > 1: print(f'omit {obj.name}', file=sys.stderr)
                            self.omit = True
    for obj in dataobj:
        if not obj.omit:
            for pp in obj.values(): pp.setup()

def alarmval(name, tt, v):
    print(f'ALARM {name} {tt} {v}')

def checkrow(row):
    kk, vv = splitrow(row)
    tt = timeofrow(kk)
    for w, obj in zip(vv, dataobj):
        if obj.omit: continue
        try:
            v = float(w)
            if checked := obj[tt].err(v): alarmval(obj.name, tt, v)
            elif checked is None:
                print(f'NODATA {obj.name} {tt} {len(obj[tt].allvals)}', file=sys.stderr)
            obj.lastval = v
        except (ValueError) as e:
            if debug > 0: print(f'ignore {obj.name}', file=sys.stderr)

def report():
    for obj in dataobj:
        if obj.omit: continue
        for key, dat in obj.items():
            print(obj.name, key, dat.avg(), dat.num)

if __name__ == '__main__':
    prgmargs = sys.argv[1:]
    while prgmargs:
        if prgmargs[0] == '-c': checkalarm = not checkalarm
        elif len(prgmargs) == 1: break
        else: break
        prgmargs = prgmargs[1:]

    if prgmargs:
        if prgmargs[0][0] == '-':
            print(f'{sys.argv[0]} [-m] [csv-files]', file=sys.stderr)
            sys.exit(2)
        train(prgmargs)
    if checkalarm:
        print(sys.stdin.readline().strip())
        for row in readcsv(sys.stdin): checkrow(row)
    else:
        report()
