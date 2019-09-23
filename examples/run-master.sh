#!/bin/bash

echo "BEGIN" > log
echo "===============" >> log

for num in {1..20}; do
    ./run-test.sh --nn --tx-gain 16.5 --rx-gain 17 --bps 2 --time 600 -s | tee -a log
    echo "" >> log
    echo "===============" >> log
done

