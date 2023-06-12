# ROAD-R Challenge 
This repository contains baseline code for the first and second tasks of the [ROAD-R Challenge](https://sites.google.com/view/road-r/).
The code is built on top of [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset).

The first task requires developing models for scenarios where only little annotated data is available at training time. 
More precisely, only 3 out of 15 videos (from the training partition train_1 of the ROAD-R dataset) are used for training the models in this task.
The videos' ids are: 2014-07-14-14-49-50_stereo_centre_01, 2015-02-03-19-43-11_stereo_centre_04, and 2015-02-24-12-32-19_stereo_centre_04.

The second task requires that the models' predictions are compliant with the 243 requirements provided in `constraints/requirements.txt`.

## Table of Contents
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#prostprocessing'>Post-processing</a>



## Dependencies and data preparation
For the dataset preparation and packages required to train the models, please see the [Requirements](https://github.com/gurkirt/3D-RetinaNet#requirements) section from 3D-RetinaNet for ROAD.  

To download the pretrained weights, please see the end of the [Performance](https://github.com/gurkirt/3D-RetinaNet#performance) section from 3D-RetinaNet for ROAD.  

## Training

To train the model, provide the following positional arguments:
 - `DATA_ROOT`: path to a directory in which `road` can be found, containing `road_test_v1.0.json`, `road_trainval_v1.0.json`, and directories `rgb-images` and `videos`.
 - `SAVE_ROOT`: path to a directory in which the experiments (e.g. checkpoints, training logs) will be saved.
 - `MODEL_PATH`: path to the directory containing the weights for the chosen backbone (e.g. `resnet50RCGRU.pth`).

Example train command (to be run from the root of this repository):

```
DATASET="${HOME}/datasets/"
EXPDIR="${HOME}/experiments/ROAD-R_Challenge_SSL/"
KINETICS="${HOME}/experiments/kinetics-pt/"

mode=train
max_epochs=150
milestones="130,145"
lr=0.0041
batch_size=4

tnorm=Godel
req_weight=10

unlabelled_proportion=0.10
agentness_threshold=0.25

python main.py ${DATASET} ${EXPDIR}/${EXP_ID}/ ${KINETICS} \
        --MODE=$mode --MAX_EPOCHS=${max_epochs} --MILESTONES=$milestones \
        --LR=$lr --BATCH_SIZE=${batch_size} \
        --LOGIC=$tnorm --req_loss_weight=${req_weight} \
        --unlabelled_proportion=${unlabelled_proportion} --agentness_th=${agentness_th} 

```

## Testing 
Below is an example command to test a model.

```
CUDA_VISIBLE_DEVICES=1 python main.py /home/user/datasets /home/user/experiments/  /home/user/kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --NUM_WORKERS=8 --req_loss_weight=10.0 --LOGIC=Product 
```

This command will generate a file containing the detected boxes at the following location:
`/home/user/road/road/log-lo_cache_logic_<LOGIC>_<req_loss_weight>/<experiment-name>/detections-30-08-50_test/log-lo_ROAD_R_predictions_I3D_logic-Product-10.0.txt`.

