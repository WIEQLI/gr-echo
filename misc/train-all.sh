#!/bin/bash

ssh -J bwrc-srn-001 root@192.168.42.233 'gr-echo/misc/echo-ctrl.py -t'
ssh -J bwrc-srn-002 root@192.168.42.95 'gr-echo/misc/echo-ctrl.py -t'
