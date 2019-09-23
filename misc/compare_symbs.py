#!/usr/bin/env python
import sys
import struct
from functools import partial
import numpy as np


def main(f1, f2):
    cnt = 0
    errcnt = 0
    with open(f1, "rb") as src:
        with open(f1 + ".ascii", "w") as srcdump:
            with open(f2, "rb") as chan:
                with open(f2 + ".ascii", "w") as chandump:
                    for c1, c2 in zip(iter(partial(src.read, 8), ''),
                                      iter(partial(chan.read, 8), '')):
                        csrc = struct.unpack("ff", c1)
                        csrc = np.round(csrc[0] + 1j * csrc[1], 5)
                        cchan = struct.unpack("ff", c2)
                        cchan = np.round(cchan[0] + 1j * cchan[1], 5)
                        if np.abs(csrc - cchan) > 1e-3:
                            print("unequal at {} symbols {} != {}".format(cnt, csrc, cchan))
                            errcnt += 1
                        srcdump.write(str(csrc))
                        chandump.write(str(cchan))
                        cnt += 1
    print("{} errors out of {} symbols, rate = {}".format(errcnt, cnt, errcnt*1.0/cnt))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need two filenames: src_file, chan_file")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
    sys.exit(0)
