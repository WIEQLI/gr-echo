#!/usr/bin/env python

import numpy as np
import sys

n = input("How many bits? ")
try:
    n = int(n)
except:
    print("You must enter an integer")
    sys.exit(1)

bits = np.random.randint(0, 2, n)

np.savetxt("preamble", bits, fmt='%d', newline=',')

print("Saved to file 'preamble'")

