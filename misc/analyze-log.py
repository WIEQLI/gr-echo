#!/usr/bin/env python

import sys
import re
import pandas as pd
import datetime as dt
import numpy as np


def to_time(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S,%f')


def main(filename):
    detector_sizes = {}
    bits_per_symbol = {}
    # Format: (time string, block, uuid, samples, time elapsed)
    vals = []
    spy_errors = []
    with open(filename, "r") as f:
        i = 0
        redetect = re.compile('([0-9\-]+ [0-9:,]+) :DEBUG: (packet detect) ([a-f0-9]{6}) [\w\s\(\)]+ ([0-9]+) samples in ([0-9\.]+) seconds')
        reother = re.compile('([0-9\-]+ [0-9:,]+) :DEBUG: ([a-z2_\ ]+) ([a-f0-9]{6}) [\w\s]+ ([0-9]+) [\w\s]+ ([0-9\.\-e]+) seconds')
        respy = re.compile('([0-9\-]+ [0-9:,]+) :DEBUG: ([a-z2_\ ]+) ([a-f0-9]{6}) spy ber ([0-9\.]+) above threshold ([0-9\.]+)')
        for line in f:
            # Read in initialization info
            if i < 25:
                s = re.search('DETECTOR ([a-f0-9]{6}): Packet Size = ([0-9]+) samples', line)
                if s is not None:
                    detector_sizes[s.group(1)] = int(s.group(2))
                s = re.search('([\w]+ [\w]+) ([a-f0-9]{6}): ([0-9]+) bits per symbol', line)
                if s is not None:
                    bits_per_symbol[s.group(2)] = (int(s.group(3)), s.group(1))
            i += 1
            # Read runtime logging
            det = redetect.search(line)
            oth = reother.search(line)
            spy = respy.search(line)
            if det is not None:
                # time, type, uuid, samples, telapsed
                vals.append([to_time(det.group(1)), det.group(2), det.group(3),
                             int(det.group(4)), float(det.group(5))])
            elif oth is not None:
                # time, type, uuid, bits, telapsed
                vals.append([to_time(oth.group(1)), oth.group(2), oth.group(3),
                             int(oth.group(4)), float(oth.group(5))])
            if spy is not None:
                # time, type, uuid, BER, cutoff
                spy_errors.append([to_time(spy.group(1)), spy.group(2), spy.group(3),
                             float(spy.group(4)), float(spy.group(5))])

    # Print initialization info
    print("Detector Packet Sizes")
    for k in detector_sizes:
        print("\tdetector {} {}".format(k, detector_sizes[k]))
    print("Bits Per Symbol")
    for k in bits_per_symbol:
        print("\t{} \t{} \t{}".format(bits_per_symbol[k][1], k, bits_per_symbol[k][0]))

    # Analyze log body
    # Processing time info
    df = pd.DataFrame(vals, columns=['log_time', 'block', 'uuid', 'samples', 'time'])
    vals = []
    for b in pd.unique(df.block):
        x = df.loc[df['block'] == b]
        calls = x.shape[0]
        cumtime = sum(x.time)
        rate = sum(x.samples) * 1.0 / cumtime
        vals.append([b, calls, cumtime, rate])
    dfrate = pd.DataFrame(vals, columns=['block', 'calls', 'cumulative time', 'symbols / sec'])
    dfrate['relative time'] = dfrate['cumulative time'] / dfrate['cumulative time'].sum()
    dfrate['secs / call'] = dfrate['cumulative time'] / dfrate['calls']
    dfrate['max calls / sec'] = dfrate['calls'] / dfrate['cumulative time']
    dfrate = dfrate.sort_values(by=['relative time'], ascending=False).reset_index(drop=True)
    cols = list(dfrate.columns)
    tmp = cols[3]
    cols[3] = cols[4]
    cols[4] = tmp
    dfrate = dfrate.reindex(columns=cols)
    # Bad packet info
    dfspy = pd.DataFrame(spy_errors, columns=['log_time', 'block', 'uuid', 'ber', 'cutoff'])
    vals = []
    for b in pd.unique(dfspy.block):
        x = dfspy.loc[dfspy['block'] == b]
        calls = x.shape[0]
        mean = x.ber.mean()
        min_ = x.ber.min()
        max_ = x.ber.max()
        vals.append([b, calls, mean, min_, max_])
    dfspy_summary = pd.DataFrame(vals, columns=['block', 'num_bad_packets', 'mean ber', 'min ber', 'max ber'])

    # Print packet size info
    try:
        samps = df.loc[df['block'] == 'neural mod', 'samples']
        s0 = samps.index[0]
    except:
        samps = df.loc[df['block'] == 'classic mod', 'samples']
        s0 = samps.index[0]
    bit_per_packet = samps.loc[s0]
    print("")
    print("Echo Data Size (bits):\t{}".format(bit_per_packet))

    # Print time and packet rate info
    telapsed = max(df.log_time) - min(df.log_time)
    round_trips = dfrate.loc[dfrate['block'] == 'neural mod', 'calls']
    if round_trips.empty:
        round_trips = dfrate.loc[dfrate['block'] == 'classic mod', 'calls']
    s0 = round_trips.index[0]
    round_trips = round_trips.loc[s0]
    print("")
    print("Total Elapsed Time:\t{}".format(telapsed))
    print("Round Trips:\t\t{}".format(round_trips))
    print("Packets / s:\t\t{}".format(round_trips / (telapsed.seconds + telapsed.microseconds * 1e-6)))
    print("Bits / s:\t\t{}".format(bit_per_packet * round_trips / (telapsed.seconds + telapsed.microseconds * 1e-6)))

    # Print bad packet summary 
    print("")
    print(dfspy_summary)
    np.set_printoptions(precision=1)
    print("Bad Packet Rate: {}%".format(100. * dfspy_summary['num_bad_packets'].values / round_trips))

    # Print dataframe summary
    print("")
    print(dfrate.to_string(justify='center'))

    # Print bad packet statistics
    print("")
    print(dfspy_summary.to_string(justify='center'))
    np.set_printoptions(precision=1)
    print("Bad Packet Rate: {}%".format(100. * dfspy_summary['num_bad_packets'].values / round_trips))

    # Print statistics for block timing
    print("")
    for b in pd.unique(dfrate.block):
        mx = df.loc[df['block'] == b, 'time'].max()
        med = df.loc[df['block'] == b, 'time'].median()
        print("{}\tmax time {:.5f}\tmedian time {:.5f}".format(b, mx, med))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} LOGFILE".format(sys.argv[0]))
        sys.exit(1)
    fname = sys.argv[1]
    main(fname)
