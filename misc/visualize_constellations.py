#!/usr/bin/env python
# Visual Modulator Constellation
# Josh Sanz
# 2019-01-24

import sys
import six
from os import environ, listdir
from os.path import isfile, join
import argparse
from matplotlib import pyplot as plt
import numpy as np


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def plot_mod_timelapse(config, data):
    nlabels = len(data[config['mod_files'][0]])
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = colors[:nlabels]
    # Collate data
    x = np.zeros((len(data), nlabels))
    y = np.zeros((len(data), nlabels))
    for i, f in enumerate(config['mod_files']):
        x[i, :] = np.real(data[f])
        y[i, :] = np.imag(data[f])
    xlim = [min(np.amin(x), -1), max(np.amax(x), 1)]
    ylim = [min(np.amin(y), -1), max(np.amax(y), 1)]
    alpha = np.linspace(0.25, 1, x.shape[0])
    for i in range(nlabels):
        # Set up color vector
        rgb = hex_to_rgb(colors[i])
        rgba = np.zeros((x.shape[0], 4))
        rgba[:, 0] = rgb[0] / 255.
        rgba[:, 1] = rgb[1] / 255.
        rgba[:, 2] = rgb[2] / 255.
        rgba[:, 3] = alpha
        # Plot data
        plt.scatter(x[:, i], y[:, i], c=rgba)
    plt.title("Timelapse of Constellation")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.xlim(xlim)
    plt.ylim(ylim)
    if config['save']:
        plt.savefig(join(config['dir'], config['prefix_mod'] + "_timelapse.png"))
        plt.close()
    else:
        plt.show()


def plot_mod_individuals(config, data):
    # Get axis limits
    xlim = [-1, 1]
    ylim = [-1, 1]
    for d in six.itervalues(data):
        x = np.real(d)
        y = np.imag(d)
        if min(x) < xlim[0]:
            xlim[0] = min(x)
        if max(x) > xlim[1]:
            xlim[1] = max(x)
        if min(y) < ylim[0]:
            ylim[0] = min(y)
        if max(y) > ylim[1]:
            ylim[1] = max(y)
    # Plot individual constellations
    nlabels = len(data[config['mod_files'][0]])
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = colors[:nlabels]
    for fname in config['mod_files']:
        plt.scatter(np.real(data[fname]), np.imag(data[fname]), c=colors)
        plt.title(fname[:-4])
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.xlim(xlim)
        plt.ylim(ylim)
        if config['save']:
            plt.savefig(join(config['dir'], fname[:-4] + ".png"))
            plt.close()
        else:
            plt.show()


def plot_demod_individuals(config, data):
    # Get axis limits
    xlim = [-1, 1]
    ylim = [-1, 1]
    nlabels = 1
    for d in six.itervalues(data):
        iq = d['iq']
        x = np.real(iq)
        y = np.imag(iq)
        if min(x) < xlim[0]:
            xlim[0] = min(x)
        if max(x) > xlim[1]:
            xlim[1] = max(x)
        if min(y) < ylim[0]:
            ylim[0] = min(y)
        if max(y) > ylim[1]:
            ylim[1] = max(y)
        l = d['labels']
        nlabels = max(nlabels, np.unique(l).size)
    # Plot individual constellations
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = np.array(colors)
    for fname in config['demod_files']:
        iq = data[fname]['iq']
        labels = data[fname]['labels']
        plt.scatter(iq.real, iq.imag, c=colors[labels])
        for c in np.unique(labels):
            c_iq = iq[labels == c]
            center = (np.mean(c_iq.real), np.mean(c_iq.imag))
            plt.annotate(c, center, fontsize=12, fontweight='bold')
        plt.title(fname[:-4])
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.xlim(xlim)
        plt.ylim(ylim)
        if config['save']:
            plt.savefig(join(config['dir'], fname[:-4] + ".png"))
            plt.close()
        else:
            plt.show()


def plot_ber_curves(config, data):
    ylim = [0, 1]
    for fname in config['ber_files']:
        it = data[fname][:, 0]
        ber = data[fname][:, 1]
        plt.stem(it, ber)
        plt.ylabel("BER")
        plt.xlabel("Training Iterations")
        plt.title(fname[:-4])
        # plt.ylim(ylim)
        if config['save']:
            plt.savefig(join(config['dir'], fname[:-4] + ".png"))
            plt.close()
        else:
            plt.show()


def parse_args():
    """Parse input arguments."""
    save = False
    parser = argparse.ArgumentParser(description="Display the constellations over time")
    parser.add_argument("directory", help="The directory to read and plot constellations from",
                        nargs='?', default=environ["HOME"])
    parser.add_argument("-d", "--prefix-demod", help="Filename prefix for filtering demod files",
                        default="neural_demod_constellation")
    parser.add_argument("-m", "--prefix-mod", help="Filename prefix for filtering mod files",
                        default="neural_mod_constellation")
    parser.add_argument("-s", "--save", help="Save the plots to disk", action="store_true")
    ber_group = parser.add_mutually_exclusive_group()
    ber_group.add_argument("-b", "--ber", help="Plot BER curves", action="store_true")
    ber_group.add_argument("-B", "--BER", help="Plot BER curves only", action="store_true")
    args = parser.parse_args()
    if args.save:
        save = True
    dirname = args.directory
    demod_files = sorted(filter(lambda x: (x.startswith(args.prefix_demod) and
                                           x.endswith('npz') and
                                           isfile(join(dirname, x))),
                         listdir(dirname)), key=lambda f: f[-19:-4])
    mod_files = sorted(filter(lambda x: (x.startswith(args.prefix_mod) and
                                         x.endswith('npy') and
                                         isfile(join(dirname, x))),
                       listdir(dirname)), key=lambda f: f[-19:-4])
    ber_files = sorted(filter(lambda x: (x.startswith('ber') and
                                         x.endswith('.csv') and
                                         isfile(join(dirname, x))),
                              listdir(dirname)))
    return {'dir': dirname, 'demod_files': demod_files, 'mod_files': mod_files, 'save': save,
            'prefix_demod': args.prefix_demod, 'prefix_mod': args.prefix_mod,
            'ber_files': ber_files, 'ber': args.ber, 'ber_only': args.BER}


def main():
    """Parse inputs, parse file, plot results."""
    config = parse_args()

    if not config['ber_only']:
        # Plot demod consetllations
        data = {}
        # Load in data
        for fname in config['demod_files']:
            data[fname] = np.load(join(config['dir'], fname))
        # Plot individual constellations
        plot_demod_individuals(data=data, config=config)

        # Plot mod constellations
        data = {}
        # Load in data
        for fname in config['mod_files']:
            data[fname] = np.load(join(config['dir'], fname))
        # Plot individual constellations
        plot_mod_individuals(data=data, config=config)
        # Plot a timelapse of constellation points
        plot_mod_timelapse(data=data, config=config)

    if config['ber'] or config['ber_only']:
        # Plot ber curves
        data = {}
        for fname in config['ber_files']:
            data[fname] = np.loadtxt(fname, delimiter=',', skiprows=1)
        plot_ber_curves(config=config, data=data)

    # Exit
    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)

