#!/bin/bash

cp reedsolomon.patch ../reedsolomon/
cd ../reedsolomon
git apply --index reedsolomon.patch

test $? && echo "Unable to apply patch, exiting" && exit 1

rm reedsolomon.patch
/usr/bin/env python setup.py install

