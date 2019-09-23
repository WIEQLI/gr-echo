#!/usr/bin/env python

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def tee(s, f):
    print(s)
    f.write(s + '\n')


def run_options(dirs):
    options = set(filter(lambda d: 'srn1' in d, dirs))
    if not options:
        return "NONE"
    ids = [d.split('-')[-2] for d in options]
    return ids


def plot_mod_const(mdir, mname):
    bname = os.path.basename(mname)
    centers = np.load(mname)
    # Get axis limits
    xlim = [-1.5, 1.5]
    ylim = [-1.5, 1.5]
    # Plot individual constellations
    nlabels = len(np.unique(centers))
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = colors[:nlabels]
    plt.scatter(np.real(centers), np.imag(centers), c=colors)
    plt.title(bname[:-4])
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join(mdir, bname[:-4] + ".png"))
    plt.close()


def plot_demod_const(ddir, dname):
    data = np.load(dname, allow_pickle=True)
    bname = os.path.basename(dname)
    # Get axis limits
    xlim = [-1.5, 1.5]
    ylim = [-1.5, 1.5]
    nlabels = 1
    iq = data['iq']
    labels = data['labels']
    nlabels = max(nlabels, max(labels) + 1)
    # Plot individual constellations
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = np.array(colors)
    plt.scatter(iq.real, iq.imag, c=colors[labels])
    for c in np.unique(labels):
        c_iq = iq[labels == c]
        center = (np.mean(c_iq.real), np.mean(c_iq.imag))
        plt.annotate(c, center, fontsize=12, fontweight='bold')
    plt.title(bname[:-4])
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join(ddir, bname[:-4] + ".png"))
    plt.close()


def analyze_run(run_id, skip_plot=False):
    alldirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    dirs = sorted(list(filter(lambda d: run_id in d, alldirs)))
    dir1 = dirs[0]
    dir2 = dirs[1]

    # Load BERs
    fber1 = list(filter(lambda f: 'ber_echo' in f and 'csv' in f, os.listdir(dir1)))[0]
    fber1 = os.path.join(dir1, fber1)
    fber2 = list(filter(lambda f: 'ber_echo' in f and 'csv' in f, os.listdir(dir2)))[0]
    fber2 = os.path.join(dir2, fber2)
    
    ber1 = np.loadtxt(fber1, delimiter=',', skiprows=1)
    ber2 = np.loadtxt(fber2, delimiter=',', skiprows=1)
    
    # Select measurement data
    diff1 = np.diff(ber1[:,0])
    idx1 = np.nonzero(diff1 < 10)[0][0]
    measure1 = ber1[idx1:,1]
    diff2 = np.diff(ber2[:,0])
    idx2 = np.nonzero(diff2 < 10)[0][0]
    measure2 = ber2[idx2:,1]

    # Analyze
    with open(os.path.join(dir1, 'results'), 'w') as f:
        s = "Agent 1 {} samples".format(measure1.size)
        tee(s, f)
        s = "Agent 1 mean BER {}".format(np.mean(measure1))
        tee(s, f)
        s = "Agent 1 min BER {}".format(np.min(measure1))
        tee(s, f)
        s = "Agent 1 max BER {}".format(np.max(measure1))
        tee(s, f)
        s = "Agent 1 approx training symbols {}".format(int(ber1[idx1, 0] * 256 * 2))
        tee(s, f)
    with open(os.path.join(dir2, 'results'), 'w') as f:
        s = "Agent 2 {} samples".format(measure2.size)
        tee(s, f)
        s = "Agent 2 mean BER {}".format(np.mean(measure2))
        tee(s, f)
        s = "Agent 2 min BER {}".format(np.min(measure2))
        tee(s, f)
        s = "Agent 2 max BER {}".format(np.max(measure2))
        tee(s, f)
        s = "Agent 2 approx training symbols {}".format(int(ber2[idx2, 0] * 256 * 2))
        tee(s, f)

    # Final mod/demod configurations
    if not skip_plot:
        # Modulators
        try:
            fmod1 = sorted(list(filter(lambda f: 'neural_mod' in f and 'npy' in f, os.listdir(dir1))))[-1]
            fmod1 = os.path.join(dir1, fmod1)
            plot_mod_const(dir1, fmod1)
        except Exception as e:
            print("Could not get agent1 modulator constellation: {}".format(e))
        try:
            fmod2 = sorted(list(filter(lambda f: 'neural_mod' in f and 'npy' in f, os.listdir(dir2))))[-1]
            fmod2 = os.path.join(dir2, fmod2)
            plot_mod_const(dir2, fmod2)
        except Exception as e:
            print("Could not get agent2 modulator constellation: {}".format(e))
        # Demodulators
        try:
            fdemod1 = sorted(list(filter(lambda f: 'neural_demod' in f and 'npz' in f, os.listdir(dir1))))[-1]
            fdemod1 = os.path.join(dir1, fdemod1)
            plot_demod_const(dir1, fdemod1)
        except Exception as e:
            print("Could not get agent1 demodulator constellation: {}".format(e))
        try:
            fdemod2 = sorted(list(filter(lambda f: 'neural_demod' in f and 'npz' in f, os.listdir(dir2))))[-1]
            fdemod2 = os.path.join(dir2, fdemod2)
            plot_demod_const(dir2, fdemod2)
        except Exception as e:
            print("Could not get agent2 demodulator constellation: {}".format(e))


def parse_log():
    try:
        with open("log", "r") as f:
            log = f.readlines()
    except IOError as e:
        print("Could not open ./log: {}".format(e))
        print("Have you copied it down from srn3?")
        return
    idx = 0
    while idx < len(log) - 2:  # Log ends with \n====...
        # Look for start of a log entry
        if log[idx][:5] == "=====":
            # Parse the entry
            idx += 2
            modeline = log[idx].strip().split()
            mode = modeline[2]
            duration = float(modeline[-1])
            idx += 1
            idline = log[idx].strip().split()
            sid = int(idline[-1].split('-')[-1])
            idx += 1
            cmdline = log[idx].strip().split()
            # Set defaults for seeds in case of classics
            modseed = -1
            demodseed = -1
            shared = False
            pretrained = False
            source_dir = ""
            for i, w in enumerate(cmdline):
                if w == "--mod-seed":
                    modseed = int(cmdline[i+1])
                if w == "--demod-seed":
                    demodseed = int(cmdline[i+1])
                if "tx-gain" in w:
                    txgain = float(cmdline[i+1])
                if "rx-gain" in w:
                    rxgain = float(cmdline[i+1])
                if "bits-per-symb" in w:
                    bps = int(cmdline[i+1])
                if w == "--shared-preamble":
                    shared = cmdline[i+1] != '""'
                if w == "--mod-init-weights":
                    pretrained |= cmdline[i+1] != '""'
                    if pretrained:
                        source_dir = os.path.dirname(cmdline[i+1].strip('"'))
                if w == "--demod-init-weights":
                    pretrained |= cmdline[i+1] != '""'
                    if pretrained:
                        source_dir = os.path.dirname(cmdline[i+1].strip('"'))
            share_prefix = "shared-" if shared else ""
            pretrain_prefix = "pretrained-" if pretrained else ""
            basedir = pretrain_prefix + share_prefix + '-'.join([mode, str(sid)])
            for srn in ('-srn1', '-srn2'):
                curdir = basedir + srn
                try:
                    with open(os.path.join(curdir, 'sid'), 'w') as f:
                        f.write(str(sid))
                    with open(os.path.join(curdir, 'bps'), 'w') as f:
                        f.write(str(bps))
                    with open(os.path.join(curdir, 'duration'), 'w') as f:
                        f.write(str(duration))
                    with open(os.path.join(curdir, 'mod-seed'), 'w') as f:
                        f.write(str(modseed))
                    with open(os.path.join(curdir, 'demod-seed'), 'w') as f:
                        f.write(str(demodseed))
                    with open(os.path.join(curdir, 'tx-gain'), 'w') as f:
                        f.write(str(txgain))
                    with open(os.path.join(curdir, 'rx-gain'), 'w') as f:
                        f.write(str(rxgain))
                    with open(os.path.join(curdir, 'weights-source'), 'w') as f:
                        f.write(str(source_dir))
                except IOError as e:
                    print("Failed to save run configuration: {}".format(e))
        # Go to the next line
        idx += 1


def parse_args():
    p = argparse.ArgumentParser(description="Master run analysis script")
    p.add_argument("-n", "--no-plot", action="store_true", help="Do not plot constellations")
    p.add_argument("-l", "--log-only", action="store_true", help="Only parse the log file. Does not calculate run results")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-s", "--seed", type=int, help="Seed/ID of single run to analyze")
    g.add_argument("-a", "--all", action="store_true", help="Analyze all runs in the current directory")
    args = p.parse_args()
    if not args.all and not args.no_plot and not args.log_only and args.seed is None:
        print("You must specify --all or a run id", file=stderr)
        sys.exit(1)
    return args

def main():
    args = parse_args()
    parse_log()
    if args.log_only:
        return 0
    # Parse args and get dirs
    alldirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    if args.all:
        options = run_options(alldirs)
        for run in sorted(options):
            print("Analyzing run {}...".format(run))
            try:
                analyze_run(run, args.no_plot)
            except Exception as e:
                print("Failed: {}".format(e))
            print("")
    else:
        dirs = sorted(list(filter(lambda d: run_id in d, alldirs)))
        if not dirs:
            print("ID {} not found, options are {}".format(run_id, run_options(alldirs)))
            sys.exit(1)
        analyze_run(run_id, args.no_plot)
    

if __name__ == "__main__":
    main()

