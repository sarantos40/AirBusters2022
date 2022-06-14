#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import re
import itertools
import enum

# average time measurements, so that they correspond to a time unit: day/hour/minute/second (-t).
# set the unit value on the time column.
# apply only on arithmetic values (to automatically decide)
# set the number of samples on column <colnval> (optional, is not None)
# merge (-m) read them in memory and sort them

## every N (groups of) rows
## could automatically detect DATE/TIME columns
## could declare exclude columns
## could implement 10 minute or 10 second averaging intervals
## add weekly mode
## add blur?

class Xtime(enum.Enum):
    SECOND = enum.auto()
    MINUTE = enum.auto()
    HOUR = enum.auto()
    DAY = enum.auto()

tmode, merge, valfmt = Xtime.MINUTE, False, '{:.1f}'
coldate, coltime, colnval = 0, 1, 2 # hardwired columns

def printrow(row): print('\t'.join(row))

def readcsv(f):
    for line in f:
        yield line.strip().split('\t')

def rowkey(row):
    if tmode == Xtime.DAY: return row[coldate]
    if tmode == Xtime.SECOND: return row[coltime]
    return re.match('[0-2]?[0-9]:[0-5][0-9]' if tmode == Xtime.MINUTE else '[0-2]?[0-9]', row[coltime]).group(0)

def timeline(g):
    def avg(vals):
        vnum, vsum = 0, 0.0
        for v in vals:
            try:
                vnum += 1
                vsum += float(v)
            except (ValueError) as e:
                return v # any value
        return valfmt.format(vsum/vnum)

    vnum = None
    for kk, gg in itertools.groupby(g, rowkey):
        row = [avg(vals) for vals in zip(*gg)]
        row[coltime] = kk
        if colnval is not None: row[colnval] = str(vnum)
        yield row

def daytime(g):
    def inival(v):
        try:
            return float(v)
        except (ValueError) as e:
            return v # any value
    def strval(v): return valfmt.format(v/vnum) if type(v) is float else v

    sums, nums = {}, {}
    for row in g:
        kk = rowkey(row)
        if kk in sums:
            nums[kk] += 1
            vv = sums[kk]
            for i, v in enumerate(row):
                if type(vv[i]) is float:
                    try:
                        vv[i] += float(v)
                    except (ValueError) as e:
                        vv[i] = v # no float any more
        else:
            sums[kk], nums[kk] = [inival(v) for v in row], 1

    for kk in sorted(nums): # sort times
        vnum = nums[kk]
        row = [strval(v) for v in sums[kk]]
        row[coltime] = kk
        if colnval is not None: row[colnval] = str(vnum)
        yield row

def day(g): return daytime(g) if merge else timeline(g)

if __name__ == '__main__':
    prgmargs = sys.argv[1:]
    while prgmargs:
        if prgmargs[0] == '-m': merge = not merge
        elif len(prgmargs) == 1: break
        elif prgmargs[0] == '-t': tmode = Xtime(int(prgmargs.pop(1)))
        else: break
        prgmargs = prgmargs[1:]

    if prgmargs:
        if prgmargs[0][0] == '-':
            print(f'{sys.argv[0]} [-t 1-4] [-m] [csv-files]', file=sys.stderr)
            sys.exit(2)
        for fn in prgmargs:
            with open(fn) as f:
                print(f.readline().strip()) # print a header each time!
                for row in day(readcsv(f)): printrow(row)
    else:
        print(sys.stdin.readline().strip())
        for row in day(readcsv(sys.stdin)): printrow(row)
