#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import re
import itertools
import collections
import datetime

coldate, coltime, colnval = 0, 1, 2 # hardwired columns
valfmt = '{:.1f}' # valfmt.format(x)
coluse = () # the value columns to really use, empty == all
inpfile = None
window = 1
debug = 0
header, dataobj = None, None

labels = ('date', 'time', 'seconds', 'cputemp', 'temperature', 'pressure', 'humidity', 'light', 'oxidised', 'reduced', 'nh3')

def strval(v): return valfmt.format(v/vnum) if type(v) is float else v

class DataPoint:
    'Data values for the smallest unit (one variable/measurement in a specific time of the day)'
    nvalues = 10
    def __init__(self):
        self.sums, self.num = 0, 0
        self.allvals = []
        self.wavg = None
        self.errvals, self.erridx = [], 0
    def add(self, val): #### unused
        if debug > 3: print(f'add {val}', file=sys.stderr)
        self.allvals.append(val)
        self.num += 1
        self.sums += val
        if self.wavg is None: self.wavg = val
        self.wavg +=  (val - self.wavg) * 2 / self.num
    def lazy_add(self, val):
        'Add a new value, that will be processed later'
        if debug > 3: print(f'lazy_add {val}', file=sys.stderr)
        self.allvals.append(val)
    def setup(self):
        'Process all added values, and calculate statistics'
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
        'Return the average value of all values.'
        return self.sums/self.num
    def err(self, v): # if len(self.allvals) < 1: return None #### unused
        'Return if the current value is considered unusual/error'
        return abs(v - self.allvals[-1]) > self.maxdiff * window
    def err2(self, v): # if len(self.allvals) < 1: return None # not ready to compare #### unused
        'Return if the current value is considered unusual/error, considering many previous values'
        if len(self.errvals) < DataPoint.nvalues: self.errvals.append(v)
        else: self.errvals[self.erridx % DataPoint.nvalues] = v
        self.erridx += 1
        if len(self.errvals) < DataPoint.nvalues: return None
        return all(abs(v - self.allvals[-1]) > self.maxdiff * window for v in self.errvals)
    def err2a(self, v): # if len(self.allvals) < 2: return None #### unused
        return (2 * self.allvals[-2] - self.allvals[-1] < v) != (v < 3 * self.allvals[-1] - 2 * self.allvals[-2])

class DataDict(dict):
    'Data values for all values of a variable/measurement, that is using keys to specify the points of the variable/measurement.'
    nhistory = 20
    def __missing__(self, key):
        self[key] = v = DataPoint()
        self.prevlst, self.avgslst, self.previdx, self.prevsum, self.avgssum = [0] * DataPoint.nvalues,  [0] * DataPoint.nvalues, 0, 0, 0
        return v
    def __init__(self, name, omit = None):
        self.omit, self.name = omit, name
    def predict(self, va, vc): # from last keys
        'Predict the next value, based on many previous values (independent of keys)'
        u = vc if self.previdx < DataDict.nhistory else va + (self.prevsum - self.avgssum) / DataPoint.nvalues
        self.prevsum += vc - self.prevlst[self.previdx % DataPoint.nvalues]
        self.avgssum += va - self.avgslst[self.previdx % DataPoint.nvalues]
        self.prevlst[self.previdx % DataPoint.nvalues] = vc
        self.avgslst[self.previdx % DataPoint.nvalues] = va
        self.previdx += 1
        return u
    def err(self, key, v):
        'Return if the current value has a large deviation from the predicted value.'
        xv = self.predict(self[key].avg(), v)
        if debug == -1: print(f'{xv:.2f}, {v:.2f}, {abs(xv - v):.3f}')
        return abs(xv - v) > self[key].maxdiff * window

class DataFull(dict): ####
    def __init__(self, name):
        self.omit, self.name = None, name
    
## add daily weekly mode

def printrow(row): print('\t'.join(row))

def rowhhmm(row): return re.match('[0-2]?[0-9]:[0-5][0-9]', row[coltime]).group(0)

def timeofrow(row):
    return rowhhmm(row)
    return row[coldate]

def splitrow(row):
    return row[:3], row[3:]

def readcsv(f):
    for line in f:
        yield line.strip().split('\t')

def train(filenames):
    'Record statistcal characteristics of all variables/measurements, to model their behaviour.'
    global header, dataobj
    for fn in filenames:
        if debug > 1: print(f'File {fn}', file=sys.stderr)
        with open(fn) as f:
            for row in readcsv(f):
                kk, vv = splitrow(row)
                if dataobj is None:
                    header = vv
                    dataobj = [DataDict(h, bool(coluse)) for h in vv]
                    for x in coluse:
                        i = vv.index(x) if x in vv else int(x)-1
                        dataobj[i].omit = False
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
    if debug == -2: report(sys.stderr)

def alarmval(obj, tt, v):
    print(f'ALARM {obj.name} {tt} {v:.2f} ({obj[tt].avg():.2f},{obj[tt].maxdiff:.2f})')

def checkrow(row):
    'Check all variables/measurements in a row for unexpected values'
    kk, vv = splitrow(row)
    tt = timeofrow(kk)
    for w, obj in zip(vv, dataobj):
        if obj.omit: continue
        try:
            v = float(w)
            if checked := obj.err(tt, v): alarmval(obj, tt, v)
            elif checked is None:
                if debug > 0: print(f'NODATA {obj.name}[{tt}] all={len(obj[tt].allvals)} n={obj[tt].erridx}', file=sys.stderr)
            obj.lastval = v
        except (ValueError) as e:
            if debug > 0: print(f'ignore {obj.name}', file=sys.stderr)

def checkcsv(f):
    'Check all rows in a csv file'
    print(f.readline().strip(), file=sys.stderr)
    for row in readcsv(f): checkrow(row)

def report(f = sys.stdout):
    'Report statistical values for the variables/measurements.'
    for obj in dataobj:
        if obj.omit: continue
        for key, dat in obj.items():
            print(f'{obj.name}[{key}] num={dat.num} avg={dat.avg():.2f} diff={dat.maxdiff:.2f}', file=f)

if __name__ == '__main__':
    prgmargs = sys.argv[1:]
    while prgmargs:
        if len(prgmargs) > 1 and prgmargs[0] == '-i': inpfile = prgmargs.pop(1)
        elif len(prgmargs) > 1 and prgmargs[0] == '-c': coluse = prgmargs.pop(1).split(',')
        elif len(prgmargs) > 1 and prgmargs[0] == '-g': debug = int(prgmargs.pop(1))
        elif len(prgmargs) > 1 and prgmargs[0] == '-w': window = float(prgmargs.pop(1))
        elif len(prgmargs) == 1: break
        else: break
        prgmargs = prgmargs[1:]

    if prgmargs:
        if prgmargs[0][0] == '-':
            print(f'{sys.argv[0]} [-m] [csv-files]', file=sys.stderr)
            sys.exit(2)
        train(prgmargs)

    if not inpfile:
        report()
    elif inpfile == '-':
        checkcsv(sys.stdin)
    else:
        with open(inpfile) as f: checkcsv(f)
