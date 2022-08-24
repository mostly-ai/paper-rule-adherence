#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
    rm -rfv engine-ws
    python adult_syn_00.py
    python adult_syn_01.py
    python adult_syn_02.py
    rm -rfv engine-ws/model-data
    python adult_syn_03.py
    python adult_syn_04.py
    mkdir runs/$i
    cp *gz runs/$i
done
