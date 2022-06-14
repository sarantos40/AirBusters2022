#!/usr/bin/env python3

import os.path
import sys
import time
import collections
import threading
import subprocess

SILENT = False
SHOWPROGRESS = 0
DELAY = 10
AVGVALS = 10
OMIT = 1
LEDPATH, LEDPATTERN = '/sys/class/leds/led1/brightness', [['0', 1.0], ['255', 3.0]]

CPUTEMP, BME280, LTR559, ENVIROPLUS, PMS5003 = True, True, True, True, False

if BME280:
    try:
        import bme280
        # BME280 temperature/pressure/humidity sensor
        bme280 = bme280.BME280()
    except (ImportError):
        BME280 = False
        if not SILENT: sys.stderr.write('BME280 temperature/pressure/humidity module is not available\n')

if LTR559:
    try:
        try:
            # Transitional fix for breaking change in LTR559
            from ltr559 import LTR559
            ltr559 = LTR559()
        except ImportError:
            import ltr559
    except (ImportError):
        LTR559 = False
        if not SILENT: sys.stderr.write('LTR559 proximity/light module is not available\n')

if ENVIROPLUS:
    try:
        import enviroplus
        from enviroplus import gas ## !! to make gas appear !!
    except (ImportError):
        ENVIROPLUS = False
        if not SILENT: sys.stderr.write('ENVIROPLUS gas module is not available\n')

if PMS5003:
    try:
        import pms5003
        # PMS5003 particulate sensor
        pms5003 = pms5003.PMS5003()
    except (ImportError):
        PMS5003 = False
        if not SILENT: sys.stderr.write('PMS5003 particulate module is not available\n')

def get_mtu():
    'Get some measurement.'
    obj = subprocess.run(['/usr/bin/cat', '/sys/class/net/enp1s0/mtu'], stdout=subprocess.PIPE)
    return float(obj.stdout)

def cpu_get_temperature():
    'Get the temperature of the CPU.'
    obj = subprocess.run(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE, universal_newlines=True)
    ## if SHOWPROGRESS: print(obj.returncode, file=sys.stderr)
    ## maybe also check obj.returncode - and output parsing
    return float(obj.stdout[obj.stdout.index('=') + 1:obj.stdout.rindex("'")])

class SmoothValue(): ### float?
    ### use setmem for different local mem
    def reset(self): self.sum, self.num = 0.0, 0
    def __str__(self): return f'{self.avg():.1f}'
    def __init__(self, mem = None): # global mem
        self.reset()
    def add(self, v):
        self.sum += v
        self.num += 1
    def avg(self): return self.sum / self.num
    def avgreset():
        v = self.sum / self.num
        self.reset()
        return v
    def memorize(self): self.reset()

class HistoryValue(list): ###
    # Maintain length of list
    ### use setmem for different local mem
    ## def __init__(self, mem = None): # global mem
    def memorize(self, val):
        del self[0]
        self.append(val)

class TimeTable():
    def __init__(self, csvfile, nomit):
        self.csvfile, self.n = csvfile, -nomit
    def memorize(self, vals):
        if self.n == 0: # print only the used header labels
            print('date\ttime\tseconds\t' + '\t'.join(vals.keys()), file=self.csvfile)
            if SHOWPROGRESS: print(','.join(vals.keys()), file=sys.stderr)
        if self.n >= 0: # print the used values, in the order they are set
            tg = time.localtime()
            tsec = int(time.mktime(tg))
            if SHOWPROGRESS: print(time.strftime('%H:%M:%S', tg), file=sys.stderr)
            ## if SHOWPROGRESS: print(vals, file=sys.stderr)
            print(time.strftime('%Y-%m-%d\t%H:%M\t', tg) + f'{tsec}\t' + '\t'.join([f'{w.avg():.1f}' for w in vals.values()]), file=self.csvfile)
        self.n += 1

def maketable(csvfile):
    global CPUTEMP, BME280, LTR559, ENVIROPLUS, PMS5003
    delay = DELAY/AVGVALS - 0.1
    tbl = TimeTable(csvfile, OMIT)
    vals = collections.defaultdict(SmoothValue)
    time.sleep(1.0)
    while True:
        for _ in range(AVGVALS): # Smooth out with some averaging to decrease jitter

            if CPUTEMP:
                try:
                    vals['cputemp'].add(cpu_get_temperature()) #units = 'C'
                except (FileNotFoundError):
                    CPUTEMP = False
                    if not SILENT: sys.stderr.write('CPU temperature is not available\n')
                    del vals['cputemp']

            if BME280:
                try:
                    vals['temperature'].add(bme280.get_temperature()) #units = 'C'
                    vals['pressure'].add(bme280.get_pressure()) #units = 'hPa'
                    vals['humidity'].add(bme280.get_humidity()) #units = '%'
                except ():
                    BME280 = False ## warn too
                    if not SILENT: sys.stderr.write('BME280 temperature/pressure/humidity sensor is not available\n')

            if LTR559:
                try:
                    proximity = ltr559.get_proximity() ## use for input
                    vals['light'].add(ltr559.get_lux() if proximity < 10 else 1) #units = 'Lux'
                    ## if proximity ... we could do SOMETHING
                except ():
                    LTR559 = False ## warn too
                    if not SILENT: sys.stderr.write('LTR559 proximity/light sensor is not available\n')

            if ENVIROPLUS:
                try:
                    gas_data = enviroplus.gas.read_all()
                    vals['oxidised'].add(gas_data.oxidising / 1000) #units = 'kO'
                    vals['reduced'].add(gas_data.reducing / 1000) #units = 'kO'
                    vals['nh3'].add(gas_data.nh3 / 1000) #units = 'kO'
                except ():
                    ENVIROPLUS = False
                    if not SILENT: sys.stderr.write('ENVIROPLUS gas sensor is not available\n')

            if PMS5003:
                try:
                    pms_data = pms5003.read()
                    vals['pm1'].add(float(pms_data.pm_ug_per_m3(1.0))) #units = 'ug/m3'
                    vals['pm25'].add(float(pms_data.pm_ug_per_m3(2.5))) #units = 'ug/m3'
                    vals['pm10'].add(float(pms_data.pm_ug_per_m3(10))) #units = 'ug/m3'
                except (pms5003.SerialTimeoutError, pms5003.ReadTimeoutError):
                    PMS5003 = False
                    if not SILENT: sys.stderr.write('PMS5003 particulate sensor is not available\n')

            time.sleep(delay)
        tbl.memorize(vals)
        for k in vals: vals[k].memorize() # keep the order of the keys

def nextfn(fn, n = 0):
    'Find a file name that does not exist, based on <fn> and the next number.'
    root, ext = os.path.splitext(fn)
    while os.path.exists(fn) and os.path.getsize(fn) > 0:
        n += 1
        fn = root + str(n) + ext
    return fn

def blinkloop():
    'Blink for ever the system LED in the specified pattern: list of [brightness, time].'
    ## could look at a global variable to finish - also should quit with led ON
    try:
        while True:
            for v, t in LEDPATTERN:
                with open(LEDPATH, 'w') as f: f.write(v)
                time.sleep(t)
    except (FileNotFoundError):
        if not SILENT: sys.stderr.write('Blinking LED not used\n')

if __name__ == '__main__':
    # blink the led to show that the machine is working gathering data
    threading.Thread(target=blinkloop).start()
    try:
        maketable(open(nextfn(sys.argv[1]), 'w') if len(sys.argv) > 1 else sys.stdout)
    # Exit cleanly
    except (KeyboardInterrupt):
        sys.exit(0)


## depricate omit 
## use per minute values
