#!/bin/bash
# datasets with size larger than 1000

python src/generate_cfs_baseline.py --dataset TwoLeadECG --pos 1 --neg 2 --output twoleadecg-outfile-baseline.csv ;

python src/generate_cfs_baseline.py --dataset Yoga --pos 1 --neg 2 --output yoga-outfile-baseline.csv;

python src/generate_cfs_baseline.py --dataset ItalyPowerDemand --pos 1 --neg 2 --output italypower-outfile-baseline.csv;

python src/generate_cfs_baseline.py --dataset MoteStrain --pos 1 --neg 2 --output mote-outfile-baseline.csv; 

python src/generate_cfs_baseline.py --dataset Wafer --pos 1 --neg -1 --output wafer-outfile-baseline.csv  ;

python src/generate_cfs_baseline.py --dataset PhalangesOutlinesCorrect --pos 1 --neg 0 --output phalanges-outfile-baseline.csv;

python src/generate_cfs_baseline.py --dataset FreezerRegularTrain --pos 1 --neg 2 --output freezerregular-outfile-baseline.csv ;

python src/generate_cfs_baseline.py --dataset FreezerSmallTrain --pos 1 --neg 2 --output freezesmall-outfile-baseline.csv ;

python src/generate_cfs_baseline.py --dataset FordA --pos 1 --neg -1 --output forda-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset FordB --pos 1 --neg -1 --output fordb-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset HandOutlines --pos 1 --neg 0 --output hand-outfile-baseline.csv ; 
# zhwa9764@s045:~/Projects/ds_extension$ ps -o cmd fp 61082
# CMD
# python src/generate_cfs_baseline.py --dataset HandOutlines --pos 1 --neg 0 --output hand-outfile-b
# https://blog.csdn.net/u011412226/article/details/120588286

# # datasets with size between 500 and 1000
python src/generate_cfs_baseline.py --dataset Strawberry --pos 1 --neg 2 --output strawberry-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset SonyAIBORobotSurface2 --pos 1 --neg 2 --output sony2-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset SemgHandGenderCh2 --pos 1 --neg 2 --output semg2-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset MiddlePhalanxOutlineCorrect --pos 1 --neg 0 --output middlephalanx-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset ProximalPhalanxOutlineCorrect --pos 1 --neg 0 --output proximalphalanx-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset ECGFiveDays --pos 1 --neg 2 --output ecgfive-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset DistalPhalanxOutlineCorrect --pos 1 --neg 0 --output distalphalanx-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset SonyAIBORobotSurface1 --pos 1 --neg 2 --output sony1-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset Computers --pos 1 --neg 2 --output computers-outfile-baseline.csv ;

# # # datasets with size lower than 500
python src/generate_cfs_baseline.py --dataset Earthquakes --pos 1 --neg 0 --output earthquakes-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset GunPointAgeSpan --pos 1 --neg 2 --output GunPointAgeSpan-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset GunPointMaleVersusFemale --pos 1 --neg 2 --output GunPointMaleVersusFemale-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset GunPointOldVersusYoung --pos 1 --neg 2 --output GunPointOldVersusYoung-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset Chinatown --pos 1 --neg 2 --output Chinatown-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset PowerCons --pos 1 --neg 2 --output PowerCons-outfile-baseline.csv ;

python src/generate_cfs_baseline.py --dataset ToeSegmentation1 --pos 1 --neg 0 --output ToeSegmentation1-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset WormsTwoClass --pos 1 --neg 2 --output WormsTwoClass-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset Ham --pos 1 --neg 2 --output Ham-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset ECG200 --pos 1 --neg -1 --output ECG200-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset GunPoint --pos 1 --neg 2 --output GunPoint-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset ShapeletSim --pos 1 --neg 0 --output ShapeletSim-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset ToeSegmentation2 --pos 1 --neg 0 --output ToeSegmentation2-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset HouseTwenty --pos 1 --neg 2 --output HouseTwenty-outfile-baseline.csv  ; 

# python src/generate_cfs_baseline.py --dataset DodgerLoopGame --pos 1 --neg 2 --output DodgerLoopGame-outfile-baseline.csv; # filtered due to missing values

# python src/generate_cfs_baseline.py --dataset DodgerLoopWeekend --pos 1 --neg 2 --output DodgerLoopWeekend-outfile-baseline.csv; # filtered due to missing values

python src/generate_cfs_baseline.py --dataset Herring --pos 2 --neg 1 --output Herring-outfile-baseline.csv ;

python src/generate_cfs_baseline.py --dataset Lightning2 --pos 1 --neg -1 --output Lightning-outfile-baseline.csv ;  

python src/generate_cfs_baseline.py --dataset Wine --pos 1 --neg 2 --output Wine-outfile-baseline.csv ;  

python src/generate_cfs_baseline.py --dataset Coffee --pos 1 --neg 0 --output Coffee-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset BeetleFly --pos 1 --neg 2 --output BeetleFly-outfile-baseline.csv ; 

python src/generate_cfs_baseline.py --dataset BirdChicken --pos 1 --neg 2 --output BirdChicken-outfile-baseline.csv ; 
