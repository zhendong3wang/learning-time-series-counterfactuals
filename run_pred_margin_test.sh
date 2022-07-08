#!/bin/bash
# datasets with size larger than 1000
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile.csv --n-lstmcells 8 --lr-list 0.0001 0.001 0.001 --w-type local;

python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --lr-list 0.0001 0.001 0.001 --w-type global;

python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --lr-list 0.0001 0.001 0.001 --w-type uniform;
