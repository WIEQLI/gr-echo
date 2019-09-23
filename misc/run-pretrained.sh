#!/bin/bash

echo "Running pretrained agent tests"

cd ../examples

echo "BEGIN" > log
echo "=====================" >> log

for dir in $(cat /root/trained-models/dirlist.txt); do
    #for gain in 5.5 10 12 14 15; do  # Eval with equivalent SNRs
    for gain in 6 10.5 13 14.5 16; do  # Eval with equivalent BERs
        shared=$(cat /root/trained-models/$dir/shared) 
        mode=$(cat /root/trained-models/$dir/mode) 
        time=$(cat /root/trained-models/$dir/duration)
        echo "Running test $dir $shared $mode $time..."
        ./run-test.sh --$mode --tx-gain $gain --rx-gain 31 -a 0.5 --mod-init-weights /root/trained-models/$dir/mod-weights.mdl --demod-init-weights /root/trained-models/$dir/demod-weights.mdl -t 1 | tee -a log
        echo "" >> log
        echo "=====================" >> log
    done
done

