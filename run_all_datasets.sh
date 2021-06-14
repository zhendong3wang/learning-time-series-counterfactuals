#!/bin/bash
python notebooks/latentcf-search.py --dataset ItalyPowerDemand --pos 1 --neg 2 --output italypower-outfile3.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset MoteStrain --pos 1 --neg 2 --output mote-outfile3.csv --shallow-cnn;

python notebooks/latentcf-search.py --dataset Wafer --pos 1 --neg -1 --output wafer-outfile3.csv;

python notebooks/latentcf-search.py --dataset PhalangesOutlinesCorrect --pos 1 --neg 0 --output phalanges-outfile3.csv;

python notebooks/latentcf-search.py --dataset FreezerRegularTrain --pos 1 --neg 2 --output freezerregular-outfile3.csv;

python notebooks/latentcf-search.py --dataset FreezerSmallTrain --pos 1 --neg 2 --output freezesmall-outfile3.csv;

python notebooks/latentcf-search.py --dataset FordA --pos 1 --neg -1 --output forda-outfile3.csv; 

python notebooks/latentcf-search.py --dataset FordB --pos 1 --neg -1 --output fordb-outfile3.csv; 

python notebooks/latentcf-search.py --dataset HandOutlines --pos 1 --neg 0 --output hand-outfile3.csv 
