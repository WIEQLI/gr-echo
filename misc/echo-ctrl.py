#!/usr/bin/env python

import sys
import time
import argparse
import zmq
import pmt


def parse_args():
    p = argparse.ArgumentParser(description="Send control messages to a running echo flowgraph")
    p.add_argument("-a", "--address", help="ZMQ connect address", default="tcp://localhost")
    p.add_argument("-p", "--tx-port", help="ZMQ TX connect port; RX defaults to TX + 1", type=int, default=5555)
    p.add_argument("--rx-port", help="ZMQ RX connect port; defaults to TX + 1", type=int)
    p.add_argument("--tx-gain", help="Set TX gain", type=float)
    p.add_argument("--rx-gain", help="Set RX gain", type=float)
    g = p.add_mutually_exclusive_group()
    g.add_argument("-f", "--freeze", help="Freeze the learned models and save the weights",
                   action="store_true")
    g.add_argument("-t", "--train", help="Unfreeze the learned models and resume training",
                   action="store_true")
    return p.parse_args()
   

def main():
    args = parse_args()
    # Make sure an action was specified
    if (args.tx_gain is None and
        args.rx_gain is None and
        not args.freeze and
        not args.train):
        print("At least one of {rx-gain, tx-gain, freeze, train} is required")
        sys.exit(0)
    # Initialize ZMQ
    ctx = zmq.Context()
    tx_sock = ctx.socket(zmq.PUSH)
    rx_sock = ctx.socket(zmq.PUSH)
    # Connect to flowgraph
    tx_sock.connect("{}:{}".format(args.address, args.tx_port))
    if args.rx_port is not None:
        rx_port = args.rx_port
    else:
        rx_port = args.tx_port + 1
    rx_sock.connect("{}:{}".format(args.address, rx_port))
    # Build message dicts
    tx_dict = pmt.make_dict()
    rx_dict = pmt.make_dict()
    if args.tx_gain is not None:
        tx_dict = pmt.dict_add(tx_dict, pmt.intern("gain"), pmt.to_pmt(args.tx_gain))
    if args.rx_gain is not None:
        rx_dict = pmt.dict_add(rx_dict, pmt.intern("gain"), pmt.to_pmt(args.rx_gain))
    if args.freeze:
        tx_dict = pmt.dict_add(tx_dict, pmt.intern("freeze"), pmt.PMT_NIL)
        rx_dict = pmt.dict_add(rx_dict, pmt.intern("freeze"), pmt.PMT_NIL)
    if args.train:
        tx_dict = pmt.dict_add(tx_dict, pmt.intern("train"), pmt.PMT_NIL)
        rx_dict = pmt.dict_add(rx_dict, pmt.intern("train"), pmt.PMT_NIL)
    # Send message dicts
    tx_sock.send(pmt.serialize_str(tx_dict))
    rx_sock.send(pmt.serialize_str(rx_dict))
    # Let ZMQ finish sending
    time.sleep(1.)


if __name__ == "__main__":
    main()
