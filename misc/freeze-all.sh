#!/bin/bash

ssh -J <your-usrp-host>-001 root@192.168.42.233 'gr-echo/misc/echo-ctrl.py -f'
ssh -J <your-usrp-host>-002 root@192.168.42.95 'gr-echo/misc/echo-ctrl.py -f'
