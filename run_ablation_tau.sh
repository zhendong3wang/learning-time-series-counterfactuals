#!/bin/bash
# Ablation study over the decision boundary threshold tau - ranging from 0.4 to 0.99, with an average step of 0.1
# local;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.4 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.6 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.7 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.8 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.9 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile11.csv --n-lstmcells 8 --w-type local --w-value 0.5 --tau-value 0.99 --lr-list 0.0001 0.0001 0.0001;

# global;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.4 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.6 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.7 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.8 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.9 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile22.csv --n-lstmcells 8 --w-type global --w-value 0.5 --tau-value 0.99 --lr-list 0.0001 0.0001 0.0001;

# uniform;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.4 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.6 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.7 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.8 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.9 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile33.csv --n-lstmcells 8 --w-type uniform --w-value 0.5 --tau-value 0.99 --lr-list 0.0001 0.0001 0.0001;

# unconstrained;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.4 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.5 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.6 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.7 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.8 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.9 --lr-list 0.0001 0.0001 0.0001;
python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile00.csv --n-lstmcells 8 --w-type global --w-value 1 --tau-value 0.99 --lr-list 0.0001 0.0001 0.0001;

