#!/bin/bash
# datasets with size larger than 1000
python notebooks/latentcf-search.py --dataset Yoga --pos 1 --neg 2 --output yoga-outfile.csv;

python notebooks/latentcf-search.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile.csv --shallow-cnn;

python notebooks/latentcf-search.py --dataset ItalyPowerDemand --pos 1 --neg 2 --output italypower-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset MoteStrain --pos 1 --neg 2 --output mote-outfile.csv --shallow-cnn;

python notebooks/latentcf-search.py --dataset Wafer --pos 1 --neg -1 --output wafer-outfile.csv;

python notebooks/latentcf-search.py --dataset PhalangesOutlinesCorrect --pos 1 --neg 0 --output phalanges-outfile.csv;

python notebooks/latentcf-search.py --dataset FreezerRegularTrain --pos 1 --neg 2 --output freezerregular-outfile.csv;

python notebooks/latentcf-search.py --dataset FreezerSmallTrain --pos 1 --neg 2 --output freezesmall-outfile.csv;

python notebooks/latentcf-search.py --dataset FordA --pos 1 --neg -1 --output forda-outfile.csv; 

python notebooks/latentcf-search.py --dataset FordB --pos 1 --neg -1 --output fordb-outfile.csv; 

python notebooks/latentcf-search.py --dataset HandOutlines --pos 1 --neg 0 --output hand-outfile.csv 

# datasets with size between 500 and 1000
python notebooks/latentcf-search.py --dataset Strawberry --pos 1 --neg 2 --output strawberry-outfile.csv; 

python notebooks/latentcf-search.py --dataset SonyAIBORobotSurface2 --pos 1 --neg 2 --output sony2-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset SemgHandGenderCh2 --pos 1 --neg 2 --output semg2-outfile.csv; 

python notebooks/latentcf-search.py --dataset MiddlePhalanxOutlineCorrect --pos 1 --neg 0 --output middlephalanx-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset ProximalPhalanxOutlineCorrect --pos 1 --neg 0 --output proximalphalanx-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset ECGFiveDays --pos 1 --neg 2 --output ecgfive-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset DistalPhalanxOutlineCorrect --pos 1 --neg 0 --output distalphalanx-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset SonyAIBORobotSurface1 --pos 1 --neg 2 --output sony1-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset Computers --pos 1 --neg 2 --output computers-outfile.csv --shallow-cnn

# datasets with size lower than 500
python notebooks/latentcf-search.py --dataset Earthquakes --pos 1 --neg 0 --output earthquakes-outfile.csv; 

python notebooks/latentcf-search.py --dataset GunPointAgeSpan --pos 1 --neg 2 --output GunPointAgeSpan-outfile.csv --shallow-cnn --shallow-lstm; 

python notebooks/latentcf-search.py --dataset GunPointMaleVersusFemale --pos 1 --neg 2 --output GunPointMaleVersusFemale-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset GunPointOldVersusYoung --pos 1 --neg 2 --output GunPointOldVersusYoung-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset Chinatown --pos 1 --neg 2 --output Chinatown-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset PowerCons --pos 1 --neg 2 --output PowerCons-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset ToeSegmentation1 --pos 1 --neg 0 --output ToeSegmentation1-outfile.csv; 

python notebooks/latentcf-search.py --dataset WormsTwoClass --pos 1 --neg 2 --output WormsTwoClass-outfile.csv; 

python notebooks/latentcf-search.py --dataset Ham --pos 1 --neg 2 --output Ham-outfile.csv; 

python notebooks/latentcf-search.py --dataset ECG200 --pos 1 --neg -1 --output ECG200-outfile.csv --shallow-cnn; 

python notebooks/latentcf-search.py --dataset GunPoint --pos 1 --neg 2 --output GunPoint-outfile.csv --shallow-cnn --shallow-lstm ;

python notebooks/latentcf-search.py --dataset ShapeletSim --pos 1 --neg 0 --output ShapeletSim-outfile.csv --shallow-lstm ; 

python notebooks/latentcf-search.py --dataset ToeSegmentation2 --pos 1 --neg 0 --output ToeSegmentation2-outfile.csv; 

python notebooks/latentcf-search.py --dataset HouseTwenty --pos 1 --neg 2 --output HouseTwenty-outfile.csv; 

# python notebooks/latentcf-search.py --dataset DodgerLoopGame --pos 1 --neg 2 --output DodgerLoopGame-outfile.csv; # filtered due to missing values

# python notebooks/latentcf-search.py --dataset DodgerLoopWeekend --pos 1 --neg 2 --output DodgerLoopWeekend-outfile.csv; # filtered due to missing values

python notebooks/latentcf-search.py --dataset Herring --pos 2 --neg 1 --output Herring-outfile.csv --shallow-cnn --shallow-lstm; 

python notebooks/latentcf-search.py --dataset Lightning2 --pos 1 --neg -1 --output Lightning2-outfile.csv --shallow-cnn --shallow-lstm; 

python notebooks/latentcf-search.py --dataset Wine --pos 1 --neg 2 --output Wine-outfile.csv --shallow-cnn --shallow-lstm; 

python notebooks/latentcf-search.py --dataset Coffee --pos 1 --neg 0 --output Coffee-outfile.csv --shallow-lstm; 

python notebooks/latentcf-search.py --dataset BeetleFly --pos 1 --neg 2 --output BeetleFly-outfile.csv; 

python notebooks/latentcf-search.py --dataset BirdChicken --pos 1 --neg 2 --output BirdChicken-outfile.csv; 
