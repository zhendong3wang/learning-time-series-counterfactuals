#!/bin/bash
# datasets with size larger than 1000
python src/gc_latentcf_search.py --dataset Yoga --pos 1 --neg 2 --output yoga-outfile.csv --n-lstmcells 128;

python src/gc_latentcf_search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile.csv --n-lstmcells 8;

python src/gc_latentcf_search.py --dataset ItalyPowerDemand --pos 1 --neg 2 --output italypower-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset MoteStrain --pos 1 --neg 2 --output mote-outfile.csv  --n-lstmcells 128;

python src/gc_latentcf_search.py --dataset Wafer --pos 1 --neg -1 --output wafer-outfile.csv  --n-lstmcells 64;

python src/gc_latentcf_search.py --dataset PhalangesOutlinesCorrect --pos 1 --neg 0 --output phalanges-outfile.csv --n-lstmcells 128;

python src/gc_latentcf_search.py --dataset FreezerRegularTrain --pos 1 --neg 2 --output freezerregular-outfile.csv --n-lstmcells 128;

python src/gc_latentcf_search.py --dataset FreezerSmallTrain --pos 1 --neg 2 --output freezesmall-outfile.csv --n-lstmcells 128;

python src/gc_latentcf_search.py --dataset FordA --pos 1 --neg -1 --output forda-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset FordB --pos 1 --neg -1 --output fordb-outfile.csv --n-lstmcells 64; 

python src/gc_latentcf_search.py --dataset HandOutlines --pos 1 --neg 0 --output hand-outfile.csv --n-lstmcells 128; 

# # datasets with size between 500 and 1000
python src/gc_latentcf_search.py --dataset Strawberry --pos 1 --neg 2 --output strawberry-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset SonyAIBORobotSurface2 --pos 1 --neg 2 --output sony2-outfile.csv --n-lstmcells 64; 

python src/gc_latentcf_search.py --dataset SemgHandGenderCh2 --pos 1 --neg 2 --output semg2-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset MiddlePhalanxOutlineCorrect --pos 1 --neg 0 --output middlephalanx-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset ProximalPhalanxOutlineCorrect --pos 1 --neg 0 --output proximalphalanx-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset ECGFiveDays --pos 1 --neg 2 --output ecgfive-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset DistalPhalanxOutlineCorrect --pos 1 --neg 0 --output distalphalanx-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset SonyAIBORobotSurface1 --pos 1 --neg 2 --output sony1-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset Computers --pos 1 --neg 2 --output computers-outfile.csv --n-lstmcells 8;

# # datasets with size lower than 500
python src/gc_latentcf_search.py --dataset Earthquakes --pos 1 --neg 0 --output earthquakes-outfile.csv --n-lstmcells 64; 

python src/gc_latentcf_search.py --dataset GunPointAgeSpan --pos 1 --neg 2 --output GunPointAgeSpan-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset GunPointMaleVersusFemale --pos 1 --neg 2 --output GunPointMaleVersusFemale-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset GunPointOldVersusYoung --pos 1 --neg 2 --output GunPointOldVersusYoung-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset Chinatown --pos 1 --neg 2 --output Chinatown-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset PowerCons --pos 1 --neg 2 --output PowerCons-outfile.csv --n-lstmcells 8;

python src/gc_latentcf_search.py --dataset ToeSegmentation1 --pos 1 --neg 0 --output ToeSegmentation1-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset WormsTwoClass --pos 1 --neg 2 --output WormsTwoClass-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset Ham --pos 1 --neg 2 --output Ham-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset ECG200 --pos 1 --neg -1 --output ECG200-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset GunPoint --pos 1 --neg 2 --output GunPoint-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset ShapeletSim --pos 1 --neg 0 --output ShapeletSim-outfile.csv --n-lstmcells 64; 

python src/gc_latentcf_search.py --dataset ToeSegmentation2 --pos 1 --neg 0 --output ToeSegmentation2-outfile.csv --n-lstmcells 128; 

python src/gc_latentcf_search.py --dataset HouseTwenty --pos 1 --neg 2 --output HouseTwenty-outfile.csv  --n-lstmcells 8; 

# # python src/gc_latentcf_search.py --dataset DodgerLoopGame --pos 1 --neg 2 --output DodgerLoopGame-outfile.csv; # filtered due to missing values

# # python src/gc_latentcf_search.py --dataset DodgerLoopWeekend --pos 1 --neg 2 --output DodgerLoopWeekend-outfile.csv; # filtered due to missing values

python src/gc_latentcf_search.py --dataset Herring --pos 2 --neg 1 --output Herring-outfile.csv --n-lstmcells 64;

python src/gc_latentcf_search.py --dataset Lightning2 --pos 1 --neg -1 --output Lightning2-outfile.csv --n-lstmcells 8;  

python src/gc_latentcf_search.py --dataset Wine --pos 1 --neg 2 --output Wine-outfile.csv --n-lstmcells 64;  

python src/gc_latentcf_search.py --dataset Coffee --pos 1 --neg 0 --output Coffee-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset BeetleFly --pos 1 --neg 2 --output BeetleFly-outfile.csv --n-lstmcells 8; 

python src/gc_latentcf_search.py --dataset BirdChicken --pos 1 --neg 2 --output BirdChicken-outfile.csv --n-lstmcells 8; 
