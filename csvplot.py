#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import os.path
import re
import argparse
import itertools
import collections
import gzip, bz2, lzma

intprefix = '#'

#done
# define pattern in columns
# number preprocessing / empty numbers
# if anycols finds a pattern with more than one columns, return the different part as a name
# define range in columns
# spit figure into subplots

parser = argparse.ArgumentParser(description='''Plot the specified columns of each csv file in a figure.''')
parser.add_argument('-N', '--ncolumns', type=int, default=-1, ###
                    help='The number of columns to expect in input (for delimited auto detection). If negative, it denotes a minimum. 0 denotes ragged table.')
parser.add_argument('-n', '--names', default='',
                    help='The legend displayed names of the plotted columns/lines. Auto-delimited, or containing {} for using the column name. Default is the csv column labels, possibly (only for those?) without a matching pattern. "." specifies the line index')
parser.add_argument('-c', '--columns', default=None,
                    help='The names or (1 based) indexes of the plotted columns. Auto-delimited, or containing {} for using the label list.')
parser.add_argument('-o', '--omitcolumns', default=None,
                    help='The names or (1 based) indexes of the columns (and their names) to omit, if specified.')
parser.add_argument('-l', '--labelxcolumn', default='.',
                    help='The name or (1 based) index of the column to use for xtick labels. Default: "." for any illegal formatted column with nonempty values and "-" for none.')
parser.add_argument('--xticknum', type=int, default=8,
                    help='The number of ticks (default=8) on the x-axis.')

parser.add_argument('-m', '--mapnames', action='append', metavar='VAL:REPL', type=str, default=[],
                    help='Specify a VALUE:REPLACEMENT for the resulting names. Can be repeated.')
parser.add_argument('-p', '--maxplots', type=int, default=None, # auto value
                    help='Make different (sub)figures for every "maxplots" plots. 0=all in one (default 0/1 for 1/many files).')
# check the following, for side effects:
parser.add_argument('--xaxis', action='store_true',
                    help='Provide distinct x-axes for each subplot.')
parser.add_argument('--yaxis', action='store_true',
                    help='Provide distinct y-axes for each subplot.')
parser.add_argument('--xtickfontsize', type=int, default=None,
                    help='The font size for the x units (0=no show).')
parser.add_argument('--ytickfontsize', type=int, default=None,
                    help='The font size for the y units (0=no show).')
parser.add_argument('--xlabel', default=None,
                    help='The label in the x axis.')
parser.add_argument('--ylabel', default=None,
                    help='The label in the y axis.')
parser.add_argument('--title', default=None,
                    help='The title of the figures.')
parser.add_argument('--style', default=None,
                    help='The plotting style, e.g. "o".')
parser.add_argument('--width', type=float, default=1.0,
                    help='The plotting width.')
parser.add_argument('--occurrences', action='store_true',
                    help='Process the values: show only the frequencies of all the values, in frequency order.')
parser.add_argument('--noccurrences', action='store_true',
                    help='Process the values: show only the frequencies of the (numeric) values, in value order.')
parser.add_argument('--frequencies', action='store_true',
                    help='Process the values: for each integer starting at 1 show only the number of different values.') ###
parser.add_argument('--sumprevious', action='store_true',
                    help='Process the values: sum all previous values on each new value.')

parser.add_argument('--log', action='store_true',
                    help='Process the values: use their log base 10.')
parser.add_argument('--squeeze', type=int, default=1,
                    help='Show the average value every SQUEEZE samples.')
parser.add_argument('--blur', type=int, default=1,
                    help='Show the average value of every <blur> window.')

parser.add_argument('-f', '--maxfiles', type=int, default=0,
                    help='Make different (sub)figures for every "maxfiles" files. 0=all in one (and do not use the input filename in saving file names). The files should have the same (selected) columns (e.g. when using column ranges) and the same axes.')
parser.add_argument('--subplots', default='1',
                    help='Make many subplots for each file. Default="1,1". Example 4 or 2,3.')
parser.add_argument('--figplots', default='1',
                    help='Make many subplots for the argument files. Default="1". Example 4 or 1,2.')
parser.add_argument('--savefig', default='',
                    help='Save in file (in a filename derived from "savefig"), instead of showing, the plot.')
parser.add_argument('--dpi', type=int, default=300,
                    help='The dpi parameter to savefig, for controlling the image analysis.')
parser.add_argument('--size', action='append', metavar='', type=int, default=[],
                        help='Specify the size of the image. Can be repeated to specify the second dimension.')
parser.add_argument('--identify', default='.',
                    help='Use "identify" to pick identification information to from the input file name in labeling and file names. If "-", use the file (1 based) index instead of the file name. Default, "-", is auto. Number is component index(positive or negative).')
parser.add_argument('--useplotindex', action='store_true',
                    help='Use the line (1 based) index instead of the column name in file names.')
parser.add_argument('--delimiter', default='\t',
                    help='The default CSV delimiter . Default=TAB.')
parser.add_argument('-i', '--illegalomit', action='store_true',
                    help='Omit any columns with any cells containing illegal values.')
parser.add_argument('-z', '--noheader', action='store_true',
                    help='Assume headerless csv files.')
parser.add_argument('--alias', default='',
                    help='Use "alias" header for the (headerless) csv files.')
parser.add_argument('--legendfontsize', type=int, default=10,
                    help='The font size for the figure legend. When 0, do not output a legend.')
parser.add_argument('--prependfile', type=int, default=-1,
                    help='Whether to prepend the file name to each label. 0/1/-1=auto.')
parser.add_argument('-Z', '--zeroblanks', action='store_true',
                    help='Consider as zero any blank values.')
parser.add_argument('-B', '--keepblanks', action='store_true',
                    help='Consider any blank values - and produce an error.')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='Be more verbose on output, showing parsing details.')
parser.add_argument('--explain', action='count', default=0,
                    help='Show the details/file names involved in r/w. Can be repeated.')
parser.add_argument('filenames', nargs='*', default = ['-'],
                    help='csv filenames.')

def smartopen(fn, mode = 'r'):
    'Open a file for r or w, even if it is compressed.'
    if  args.explain and not (args.explain == 1 and 'w' in mode): print(('>\t' if 'w' in mode else '<\t') + fn)
    if fn == '-': return os.fdopen(os.dup((sys.stdout if 'w' in mode else sys.stdin).fileno()), mode=mode)
    if mode in {'r', 'w'}:
        if fn.endswith('.bz2'): return bz2.open(fn, mode + 't') # bz2.BZ2File(fn, mode)
        if fn.endswith('.gz'): return gzip.open(fn, mode + 't') # gzip.GzipFile(fn, mode)
        if fn.endswith('.xz'): return lzma.open(fn, mode + 't') # lzma.LZMAFile(fn, mode)
    return open(fn, mode)

def mydelim(line, x = ''):
    for c in x + '\t;|, ': # reserve ':' for ranges
        if c in line: return c
    return '\n'

def numcols(header, offset = 0): return (args.ncolumns if header is None else len(header)) + offset

def gencols(txt, header = None, mayberange = True):
    'In its simplest form: yield int(txt) - 1 if isint(txt) else header.index(txt). Yield column indexes and names for the column(s) specified in "txt": can contain an exact name, a number, a range or a string to match column name(s).'
    if header is not None and txt in header: yield header.index(txt), txt
    elif re.fullmatch('[0-9]+', txt): # permit leading zeros
        assert header is None or int(txt) <= numcols(header), 'Out of range column ' + txt
        v = int(txt)
        assert v or header, 'Last column is only valid on tables with headers'
        i = v - 1 if v else numcols(header, -1) # use 0 for last
        if args.verbose > 5: sys.stderr.write('Number: {} ({})\n'.format(i + 1, repr(txt)))
        yield i, header and header[i] or intprefix + txt
    elif mayberange and re.fullmatch('[0-9]*:[0-9]+(:[0-9]+)?' if header is None else '[0-9]*:[0-9]*(:[0-9]+)?$', txt): # index or range
        q = re.fullmatch('(?P<b>[0-9]*):(?P<e>[0-9]*)(:(?P<s>[0-9])+)?', txt)
        b, e, s  = q.group('b') or 1, q.group('e') or numcols(header), q.group('s') or 1 ######### 'NoneType' has no len()
        if args.verbose > 5: sys.stderr.write('Range: {}:{}:{}\n'.format(b, e, s))
        for i in range(int(b)-1, int(e), int(s)): yield i, header and header[i] or intprefix + str(i+1)
    else: # re search
        ii = header and [i for i, nam in enumerate(header) if re.search(txt, nam)]
        assert ii, 'Unknown column ' + repr(txt)
        if len(ii) == 1:
            yield ii[0], header[ii[0]] or txt or intprefix + str(ii[0]+1)
        else:
            for i in ii:
                yield i, re.sub(txt, '', header[i]) or txt or intprefix + str(i+1)

def nofloatlist(x):
    'Return the elements of "x" that cannot be converted to float.'
    def one(txt):
        try:
            float(txt) # check for a valid conversion only
        except (ValueError) as e:
            yield txt
    return [v for txt in x for v in one(txt)]

def nofloats(x):
    'Return a printable form of the no-numbers in "x".'
    y = nofloatlist(x)
    if len(y) < 10: return repr(y)
    z = set(y)
    if len(z) < 5: return repr(z)
    return '{}/{}/{} values'.format(len(z), len(y), len(x))

def uunames(names):
    'Omit unnecessary leading and trailing path components in NAMES so that the remaining components are unique.'
    ww, m = [list(os.path.split(fn)) for fn in names], len(set(names))
    while len(set(tuple(p[1:]) for p in ww)) != m: ww = [list(os.path.split(p[0]))+p[1:] for p in ww]
    j = 2
    while len(set(tuple(p[1:j]) for p in ww)) != m: j += 1
    return [os.path.join(*p[1:j]) for p in ww]
    #return [os.path.join(*p[1:]) for p in ww] # [os.path.join(*p[1:1+len(p)]) for p in ww]

def rnames(names, n = -1):
    'Omit up to N common suffixes in NAMES.'
    if names: # unless names is empty all along
        while n:
            ww, ee = zip(*[os.path.splitext(fn) for fn in names])
            if not ee[0] or any(e != ee[0] for e in ee): return names
            names = ww
            n -= 1
    return names

def doublefilenames(ww):
    def component(fn, n): ##### complete this one
        while (n < 0): fn = os.path.split(fn)[1]
        return os.path.basename(fn)
    if args.identify == '-': return [(fn, str(i)) for i, fn in enumerate(ww, 1)]
    if args.identify == '.': return list(zip(ww, rnames(uunames(ww))))
    ### USE IT AS A NUMBER: -1
    return [(fn, os.path.splitext(component(fn, int(args.identify))[0])) for fn in ww]

####
# if not ncols, the delimiter cannot be autodetected, and the first one will be assumed.
def genanyrows(fn, delim = None, ncols = -1, delimslist = ('\t', ',;', ':', '|#')):
    'Return a csv reader reading rows from file "fn" and the delimiter, by guessing delimiter from "delimslist", where delimiters on the same level have the same priority. If "ncols", that many columns must result - exactly (if positive) to at least (if negative). If 1, each input line is a single column value / text, and do not try to guess delimiter.'
    f = smartopen(fn, 'r') if fn != '-' else sys.stdin
    if ncols == 1: return ([line] for line in genlines(f)), None # no delimiter is needed
    if delim: return genrows(f, delim), delim # delimiter is known
    # read-ahead lines
    lines = f.readlines(8192*16)
    if not lines: error('File {} is empty'.format(repr(fn)), 2)
    if f == sys.stdin:
        g = itertools.chain(lines, f)
    else:
        try:
            f.seek(0)
        except:
            f.close()
            f = smartopen(fn, 'r')
        g = f
    # find delimiter (with constant repetitions) in lines
    ndelims = abs(ncols) - 1
    qdelim, qzero = None, None # best and zero frequency delimiter (so far)
    for mydelims in delimslist:
        qfreq, qdelim = None, None
        for delim in mydelims:
            freqs = [line.count(delim) for line in lines]
            if all(v == freqs[0] for v in freqs) and ((freqs[0] >= ndelims) if ncols < 0 else (freqs[0] == ndelims)) and (not qdelim or qfreq < freqs[0]): qfreq, qdelim = freqs[0], delim
        if qfreq: break # return the best/first delimiter on the round with >0 frequency
        if not qzero and qdelim: qzero = qdelim # remember the first delimiter with 0 occurrences
    else:
        if qzero: qdelim = qzero
    if args.explain: sys.stderr.write('delimiter={}, quoting={}, columns={}\n'.format(repr(qdelim), repr(args.quoting), qfreq + 1))
    return genrows(g, qdelim) if qdelim else ([line] for line in genlines(g)), qdelim

def inittable(g, nam, mycolumns, mynames): # mycolumns = text strings
    'Return the requested columns, their proposed names, their short names (for file name) and the row delimiter. Use any header, on named columns.'
    def isint(txt): return re.fullmatch('[1-9][0-9]*', txt)
    def maprow(d, row): return [d.get(c,c) for c in row]
    if args.noheader: # I could readahead
        header, delimiter = args.alias.split(','), args.delimiter
        ###header, delimiter = args.alias.split(',') if args.alias else None, args.delimiter
    else:
        headerline = next(g).strip(' \r\n')
        delimiter = mydelim(headerline, args.delimiter + ':')
        header = headerline.split(delimiter)
    omitcols = [w[0] for c in args.omitcolumns.split(mydelim(args.omitcolumns)) for w in gencols(c, header)] if args.omitcolumns else []
    cols, nams = zip(*[w for c in mycolumns for w in gencols(c, header) if w[0] not in omitcols])
    names = mynames if isinstance(mynames, list) else nams if mynames is None else [mynames.format(v) for v in nams] if mynames else [intprefix + str(i+1) for i in range(len(nams))]
    if args.verbose > 3 and header: sys.stderr.write('len(header): {}\n'.format(len(header)) if args.verbose == 4 else 'header: {}\n'.format(repr([str(i+1)+'='+header[i] for i in cols])) if args.verbose == 5 else 'header: {}\n'.format(repr(header)))
    if args.prependfile and args.legendfontsize:
        if len(names) == 1 and names[0] == '#1': # For a headerless single column
            names = [nam] # keep the file name only
        else:
            names = [nam + ('#' if v[0] != '#' else '') + v if v else nam for v in names]
    if args.mapnames: names = maprow(dict(x.rpartition(':')[::2] for x in args.mapnames), names)
    fnams = names if isinstance(mynames, list) else [header[i] for i in cols] if header else [str(i+1) for i in cols]
    if args.verbose > 2: sys.stderr.write('columns: {} names: {}\n'.format(repr([i+1 for i in cols]), repr(names)))
    if '-' != args.labelxcolumn != '.': # a real (addtional) column - add it
        labelxpairs = [w for w in gencols(args.labelxcolumn, header, False)]
        assert len(labelxpairs) == 1, 'Only one labelxcolumn should be defined: ' + repr(labelxpairs)
        cols = cols + (labelxpairs[0][0],) # one more than the names
    return cols, names, fnams, delimiter

def fnamecompose(path, parts):
    'Return file name "fn" with "n" before its extension.'
    root, ext = os.path.splitext(path)
    return root + ''.join('_' + s for s in parts) + ext

def mysave(fig, parts, done = True):
    if done and not args.savefig: plt.show()
    fn = fnamecompose(args.savefig, parts) if parts else args.savefig
    if args.explain > 1 and fn: sys.stderr.write('>\t' + fn + '\n')
    if args.savefig: fig.savefig(fn, dip=args.dpi, bbox_inches='tight')
    plt.close(fig)

def flushfigs():
    for i, (fig, axes, parts) in enumerate(globalfigs): mysave(fig, parts, i == 0)
    globalfigs[:] = []

def occurrences(y):
    pp = ((v, k) for k, v in collections.Counter(y).items())
    return zip(*sorted(pp, reverse=True))

def noccurrences(y):
    return zip(*sorted(((int(k), v) for k, v in collections.Counter(y).items())))

def frequencies(y):
    d = collections.Counter(y)
    return [d[str(k)] for k in range(1, 1 + max(d.values()))]

def sumprevious(y, w = 0):
    for v in y:
        w += v
        yield w

def squeeze1(y, n): ## fix
    'For every <n> values from <y> yield their squeeze'
    try:
        while True:
            yield sum(next(y) for _ in range(n)) / n
    except (StopIteration) as e:
        return

def squeeze2(y, n): ## fix
    'For every <n> values from <y> yield their squeeze'
    try:
        while True:
            xxx = 0
            for _ in range(n):
                xxx += next(y)
            yield xxx / n
    except (StopIteration) as e:
        return

def squeeze(y, n): # n = 1
    'For every <n> values from <y> yield their squeeze'
    xxx = 0
    for i, v in enumerate(y, 1):
        xxx += v
        if i % n == 0:
            yield xxx / n
            xxx = 0

def blur(y, n): # n = 1
    'Yield the average value of every <blur> sample window.'
    memsz = n-1
    mem = [next(y) for _ in range(memsz)]
    xxx = sum(mem)
    for i, v in enumerate(y):
        xxx += v
        yield xxx / n
        xxx -= mem[i % memsz]
        mem[i % memsz] = v

def byrows(g, nam, mycolumns, mynames):
    # header -> llbl
    # select rows
    # fix inittable
    cols, names, fnams, delimiter = inittable(g, nam, mycolumns, mynames)
    yy = [[s.strip(' \r\n').split(delimiter)[i] for i in cols] for s in g]
    return names, fnams, yy, None

def bycols(g, nam, mycolumns, mynames):
    cols, names, fnams, delimiter = inittable(g, nam, mycolumns, mynames)
    yy = list(zip(*[[s.strip(' \r\n').split(delimiter)[i] for i in cols] for s in g]))
    x = yy.pop() if len(yy) > len(names) else None # labelxcolumn
    return names, fnams, yy, x

def idxseq(gi, gv):
    'if gv has (many) repeated values, use only one.'
    cidx, clbl, cval = [], [], None
    for i, v in zip(gi, gv):
        if v != cval: # find distinct values and their index
            cidx.append(i)
            clbl.append(v) ## w or v
            cval = v
    return cidx, clbl

def onticks(n, cidx, clbl):
    'If first tick is too close, omit it.'
    tidx, tlbl = cidx[::((len(cidx)-1) // n) or 1], clbl[::((len(cidx)-1) // n) or 1]
    if len(tidx) > 4 < n and (tidx[2]-tidx[1]) / 2 > (tidx[1]-tidx[0]) < (tidx[-1]-tidx[1]) / 8: tidx, tlbl = tidx[1:], tlbl[1:]
    return tidx, tlbl

def doplot(fn, nam, mycolumns, mynames):
    'Plot columns from file "fn", prefixing them with name "nam".'
    global curfile
    names, fnams, yy, llbl = bycols(smartopen(fn), nam, mycolumns, mynames)
    xlbl, xidx = None, None # global x axis display values
    if llbl is None and not args.xaxis and args.occurrences: # form a global x axis
        for y in yy: pass #############

    if llbl is not None and len(llbl) > 1 and args.xticknum: # labelxcolumn
        # the labels change if the values do:
        if args.squeeze > 1: llbl = llbl[args.squeeze//2::args.squeeze]
        if args.blur > 1: llbl = llbl[args.blur//2:-(args.squeeze+1)//2]
        ## try matchin dates too.
        if all(re.fullmatch('[0-2]?[0-9]:[0-5][0-9]:[0-5][0-9]', v) for v in llbl): # it is hh:mm:ss
            # find better labels: select exact hours ## OR find without seconds ## OR select day changes
            for pat in ('[0-2]?[0-9]', '[0-2]?[0-9]:[0-5][0-9]'):
                cidx, clbl = idxseq(range(len(llbl)), (re.match(pat, v).group(0) for v in llbl))
                if len(cidx) >= args.xticknum: # enough distinct values
                    xidx, xlbl = onticks(args.xticknum, cidx, clbl)
                    break
        elif all(re.fullmatch('[0-2]?[0-9]:[0-5][0-9]', v) for v in llbl): # it is hh:mm
            cidx, clbl = idxseq(range(len(llbl)), (re.match('[0-2]?[0-9]', v).group(0) for v in llbl))
            if len(cidx) >= args.xticknum: xidx, xlbl = onticks(args.xticknum, cidx, clbl)
        if xidx is None: # no better derivative sequence found
            ###xidx, xlbl = list(range(0, len(llbl), ((len(llbl)-1) // args.xticknum))), llbl[::((len(llbl)-1) // args.xticknum)]
            xidx, xlbl = onticks(args.xticknum, *idxseq(range(len(llbl)), llbl))
    nfigs = 1 if args.maxplots == 0 else ((len(names) - 1) // (spltx * splty * args.maxplots) + 1) # figures that a file is spread
    nc = 0 # first line to plot
    for figno in range(nfigs):
        nf = nc + (len(names) - nc - 1) // (nfigs - figno) + 1 # remaining lines after this figure
        if curfile % remember == 0:
            fig, axes = plt.subplots(nrows=splty*fplty, ncols=spltx*fpltx, sharex=not args.xaxis, sharey=not args.yaxis, squeeze=False)
            fig.subplots_adjust(left=0.08, bottom=0.06, right=0.97, top=0.96, wspace=0.01, hspace=0.02)
            if xidx: plt.xticks(xidx, xlbl)
            parts = ([nam] if args.maxfiles else []) + ([str(figno+1) if args.useplotindex else fnams[nc]] if nfigs > 1 else [])
            if remember > 1: globalfigs.append((fig, axes, parts)) # store for addition of plots by next files
        else: fig, axes, parts = globalfigs[figno]
        klbl = None
        for sj in range(splty):
            nj = nc + (nf - nc - 1) // (splty - sj) + 1 # remaining lines after this subplot row
            #sjj = ((curfile // (figfiles * fpltx)) % fplty) * splty + sj # columns first
            sjj = ((curfile // (figfiles * fpltx)) % fplty) + sj * fplty # files first
            for si in range(spltx):
                ni = nc + (nj - nc - 1) // (spltx - si) + 1 # remaining lines after this subplot
                #sii = ((curfile // (figfiles)) % fpltx) * spltx + si # columns first
                sii = ((curfile // (figfiles)) % fpltx) + si * fpltx # files first
                for y, lbl in zip(yy[nc : ni], names[nc : ni]):
                    if args.explain > 2: sys.stderr.write('>\t\t' + repr([figno, sjj, sii]) + ' ' + lbl + '\n')
                    x = [v or '0' for v in y] if args.zeroblanks else y if args.keepblanks else [v for v in y if v]
                    try:
                        if args.occurrences: x, klbl = occurrences(x)
                        if args.noccurrences: klbl, x = noccurrences(x)
                        if args.frequencies:
                            x = frequencies(x)
                            klbl = range(1, len(x)+1)
                            print(len(x), x)
                        xx = map(float, x)
                        if args.squeeze > 1: xx = squeeze(xx, args.squeeze)
                        if args.blur > 1: xx = blur(xx, args.blur)
                        x = list(xx)
                        if args.sumprevious: x = list(sumprevious(x))
                        if args.log: x = [math.log(v) for v in x]
                        if args.verbose: sys.stderr.write('{} {}: {}/{}\n'.format(nam or fn, lbl, len(x), len(y)))
                        if args.xtickfontsize: plt.xticks(fontsize = args.xtickfontsize)
                        if args.ytickfontsize: plt.yticks(fontsize = args.ytickfontsize)
                        if args.xlabel: plt.xlabel(args.xlabel)
                        if args.ylabel: plt.ylabel(args.ylabel)
                        if args.title: plt.title(args.title)
                        style = (',' if len(x) > 1000 else '.' if len(x) > 100 else '-') if args.style is None else args.style
                        axes[sjj, sii].plot(x, style, label=lbl, linewidth=args.width, markersize=args.width)
                        if klbl: plt.xticks(range(len(klbl)), klbl)
                    except (ValueError) as e:
                        if args.verbose: sys.stderr.write('{} {}: contains no float {}\n'.format(nam or fn, lbl, nofloats(x)))
                        if args.labelxcolumn == '.' and args.xticknum and all(y): # autodetected xlabel column, only if all values non empty
                            plt.xticks(list(range(0, len(y), ((len(y)-1) // args.xticknum))), y[0::((len(y)-1) // args.xticknum)])
                        else:
                            assert args.illegalomit, str(e) + nofloats(x)
                if args.legendfontsize and nc < ni:
                    axes[sjj, sii].legend(fontsize=args.legendfontsize) # or prop={'size': 6}
                nc = ni
        if remember == 1: mysave(fig, parts)
    curfile += 1
    if curfile % remember == 0: flushfigs()

args = parser.parse_args()
if args.maxplots is None: args.maxplots = int(bool(args.maxfiles))
if args.columns is None: args.columns, args.illegalomit = ':', True # defaults if nothing specified
mynames = False if args.names == '.' else args.names if '{}' in args.names else args.names.split(mydelim(args.names)) if args.names else None
assert isinstance(mynames, list) or not ('{}' in args.columns)
mycolumns = [args.columns.format(c) for c in mynames] if '{}' in args.columns else args.columns.split(mydelim(args.columns))
#assert not isinstance(mynames, list) or len(mynames) == len(mycolumns), str(len(mynames)) + ' != ' + str(len(mycolumns)) ### dc elements from 15 files
subdims = args.subplots.split(',')
spltx, splty = (int(subdims[0]), int(subdims[-1])) if args.subplots else (1, 1)
subdims = args.figplots.split(',')
fpltx, fplty = (int(subdims[0]), int(subdims[-1])) if args.figplots else (1, 1)
figfiles = (len(args.filenames) or 1) if args.maxfiles == 0 or args.maxfiles > len(args.filenames) else args.maxfiles # used in each plot
remember = figfiles * fpltx * fplty
if args.prependfile < 0: args.prependfile = int(remember != 1)
globalfigs, curfile = [], 0 # when remember != 1
#if args.legendfontsize: plt.rcParams.update({'legend.fontsize': args.legendfontsize})
if args.verbose > 1: sys.stderr.write('Columns: {} Names: {}\n'.format(repr(mycolumns), repr(mynames)))
if args.size:
    if len(args.size) == 1: args.size.append(args.size[0]*3/4)
    matplotlib.rc('figure', figsize=args.size[:2])
for fn, nam in doublefilenames(args.filenames) or (None, None): doplot(fn, nam, mycolumns, mynames)
if curfile % remember: flushfigs()

# side legend
# fig, ax=plt.subplots(num=10, clear=True)
# figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# define (other) names for input columns --inpcolumns or --rename or --alias
## if labelxcolumn = '.' we count the row in output.
# read/plot rows instead of columns

# csvplot.py --noheader --columns :5 /tmp/tt? --sub 2,2 --fig 2,2 --maxplot 2 --explain --explain --explain
# csvplot.py --noheader --columns :5 ~/sik/tests/csv/nn --sub 2,2 --fig 2,2 --maxplot 2 --explain --explain --explain

## fix for noheader
### assert dc elements from 15 files
### names from many files
## 'NoneType' has no len()
## len(header)
## use ncolumns
## with no header and no alias, avoid column name
## if column name is equal for all, omit it.

## on date/time axis: remove seconds hh:mm:ss & tick whole hours
## squeeze n values together to one value
## plot an expression ({temp}-{cputemp})/4 ή ($5-$4)/4

## auto x & y namef from column labels
## dimensions / size (pixels) of image in file and screen

## --bar?
## nnoccurrences: noccurrences with all numbers and xticks
