#!/usr/bin/env python

import numpy as np
from os import listdir
from os.path import isfile


bers = filter(lambda x: x.startswith('ber') and isfile(x), listdir('./'))

for ber in bers:
    print("{}:".format(ber))
    data = np.loadtxt(ber, delimiter=',', skiprows=1)
    if data.size == 0:
        print("\tEmpty")
        continue
    print("\t{} samples".format(data[:,1].size))
    print("\tmean\t{}".format(np.mean(data[:,1])))
    print("\tmedian\t{}".format(np.median(data[:,1])))
    print("\tmax\t{}".format(max(data[:,1])))
    print("\tmin\t{}".format(min(data[:,1])))
    print("")

