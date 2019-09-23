#!/bin/bash

for f in $(ls); do
    if [[ ${f/".png"} = $f ]]; then
        continue
    fi

    # index=$(echo $f | cut -d '_' -f 4)
    index=$(echo $f | cut -d '_' -f 5)
    index=${index:1}
    if [[ ${#index} < 2 ]]; then
        mv "$f" "000${index}${f}"
    elif [[ ${#index} < 3 ]]; then 
        mv "$f" "00${index}${f}"
    elif [[ ${#index} < 4 ]]; then 
        mv "$f" "0${index}${f}"
    fi
done

