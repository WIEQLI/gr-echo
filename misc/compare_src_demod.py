#!/usr/bin/env python
import sys
import struct
from functools import partial
from matplotlib import pyplot as plt
import csv

def main(f1, f2):
    cnt = 0
    errcnt = 0
    err_idxs = []
    with open(f1, "rb") as src:
        with open(f1 + ".ascii", "w") as srcdump:
            with open(f2, "rb") as demod:
                with open(f2 + ".ascii", "w") as demoddump:
                    for c1, c2 in zip(iter(partial(src.read, 1), ''),
                                      iter(partial(demod.read, 1), '')):
                        isrc = struct.unpack("B", c1)[0]
                        idemod = struct.unpack("B", c2)[0]
                        if isrc != idemod:
                            print("unequal at {} bits: {} != {}".format(cnt, isrc, idemod))
                            errcnt += 1
                            err_idxs.append(cnt)
                        srcdump.write(str(isrc))
                        demoddump.write(str(idemod))
                        cnt += 1
                        if cnt % 16 == 0:
                            srcdump.write('\n')
                            demoddump.write('\n')
    print("{} errors out of {} bits: BER = {}".format(errcnt, cnt, errcnt*1.0/cnt))
    # Plot result
    plt.plot(err_idxs, list(range(errcnt)), marker='o', linestyle=None)
    plt.title("Error Occurences")
    plt.xlabel("Sample Index")
    plt.ylabel("Cumulative Errors")
    plt.show()
    # Write result for later comparison
    with open("errs.csv","w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(err_idxs)
    


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need two filenames: src_file, demod_file")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
    sys.exit(0)
