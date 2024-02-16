#!/bin/bash
# Ablation study over the prediction margin weight w - ranging from 1 to 0, with a step of 0.1
# local;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 1.0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.9 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.8 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.7 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.6 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.4 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.3 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.2 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0.1 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile1.csv --n-lstmcells 8 --w-type local --w-value 0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;

# global;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 1.0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.9 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.8 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.7 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.6 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.4 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.3 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.2 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0.1 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile2.csv --n-lstmcells 8 --w-type global --w-value 0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;

# uniform;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 1.0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.9 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.8 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.7 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.6 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.4 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.3 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.2 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0.1 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile3.csv --n-lstmcells 8 --w-type uniform --w-value 0 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
