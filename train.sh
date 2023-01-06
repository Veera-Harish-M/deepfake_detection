#!/usr/bin/env bash
DEVICE=0

cho ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on DFDC                   |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
DFDC_FACES_DIR=1m_faces_00
# DFDC_FACES_DF=/your/dfdc/faces/dataframe/path
python train_binclass.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path 1m_faces_00\
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device 0

python train_binclass.py --net EfficientNetB4 --traindb dfdc-35-5-10 --valdb dfdc-35-5-10 --dfdc_faces_df_path 1m_faces_00\ --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 30000 --seed 41 --attention --device 0 

