#!/usr/bin/env python

import sys
import csv
from matplotlib import pyplot as plt
plt.rc("figure", figsize=[12, 8])
plt.rc("font", size=14)

def main():
    fnames = sys.argv[1:]
    data = {}
    for fn in fnames:
        with open(fn, "r") as f:
            rdr = csv.reader(f)
            data[fn] = map(lambda x: int(x), [row for row in rdr][0])
    for key in data:
        plt.plot(data[key], list(range(len(data[key]))),
                marker='o', linestyle='none', label=key)
    plt.title("Errors vs Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Cumulative Error Count")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide at least one csv file name for plotting")
        sys.exit(1)
    main()
    sys.exit(0)
