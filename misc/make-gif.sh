#!/bin/bash

convert -delay 15 -loop 0 neural_mod_*.png mod_constellation.gif
convert -delay 15 -loop 0 neural_demod_*.png demod_constellation.gif
