#!/usr/bin/env python

from __future__ import division
import argparse


def main():
    p = argparse.ArgumentParser(description="Calculate packet sizes in symbols and samples")
    p.add_argument("--bps", help="Bits per symbol", type=int, default=2)
    p.add_argument("--sps", help="Samples per symbol", type=int, default=2)
    p.add_argument("--corr-reps", help="Golay correlator header repetitions", type=int, default=2)
    p.add_argument("-g", "--guard-symbs", help="Guard interval symbols", type=int, default=64)
    p.add_argument("-s", "--spy-bits", help="Classic spy bits", type=int, default=128)
    p.add_argument("--no-info-header", help="No packet info header present", action="store_true")
    p.add_argument("-n", "--preamble-bits", help="Number of bits in echo preamble", type=int, required=True)
    args = p.parse_args()

    bits = 0
    bits += 2 * args.preamble_bits
    bits += args.spy_bits
    if not args.no_info_header:
        bits += 2 * 6 * 8  # bits + RS parity
    body_symbs = (bits + args.bps - 1) // args.bps

    print("{:-6d} bits".format(bits))
    print("{:-6d} body symbols".format(body_symbs))

    tot = 0
    tot += 2 * args.corr_reps * 256
    tot += 2 * args.guard_symbs
    tot += body_symbs

    print("{:-6d} total symbols".format(tot))
    print("{:-6d} total samples".format(tot * args.sps))


if __name__ == "__main__":
    main()
