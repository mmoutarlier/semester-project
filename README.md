# semester-project
# Unsupervised Object Discovery and Dynamics Prediction

## Overview
This project evaluates the effects of different approaches for unsupervised object discovery and dynamics prediction. Leveraging datasets representing videos of objects, the study focuses on discovering object representations, understanding their interactions, and predicting their future states.

The methodology centers around SlotFormer, a Transformer-based model designed to predict object dynamics from video clips by breaking scenes into objects and modeling their temporal interactions.

## Features
- **Object Discovery: Savi Model**: Exploration of unsupervised techniques for detecting objects in dynamic scenes.
- **Dynamics Prediction: SlotFormer Model**: Prediction of future object states based on learned temporal interactions.


## Contents
The notebook includes:
1. **Motivation**: Explaining the significance of unsupervised learning in object detection and dynamics prediction.
2. **Background on SlotFormer**: Detailed description of the model and its capabilities.
3. **Experiments**: Evaluations using datasets with video clips, with a focus on:
   - Object discovery performance.
   - Dynamics prediction accuracy.


## To load the checkpoints : 

Go on https://drive.google.com/drive/folders/1XdpecT6CsHtgXxovH20Blvl7UAwO5sUS?usp=share_link
and download /logs for original SA and ISA for the slot attention and invariant slot attention 
and /checkpoints for Slot Former


## To load the datasets : 

- Tetraminoes Download the relevent datasets and store them somewhere on your machine, using the following commands:

# Tetrominoes dataset: 
wget https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords


- CLEVR
# CLEVR dataset:
wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords

Then in the \slot code, modify the invariant_slot_attention/lib/input_pipeline.py files by replacing the datasets' PATH_ variables with the actual path of the data on your disk.

- OBJ3D

This dataset is adopted from G-SWM. You can download it manually from the Google drive : https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view. Once downloaded, put it in the \data\OBJ3D.

Since the videos in OBJ3D are already extracted to frames, we don't need to further process them.

- CLEVRER
Please download CLEVRER from the official website : http://clevrer.csail.mit.edu

We will need Training Videos, Annotations, Validation Videos, Annotations for all tasks
If you want to experiment on the video prediction task, please download Object Masks and Attributes as we will evaluate the quality of the predicted object masks and bboxes.
To accelerate training, we can extract videos to frames in advance. Please run python scripts/data_preproc/clevrer_video2frames.py. You can modify a few parameters in that file.


## Requirements
- **Python 3.x**
- Libraries: The exact dependencies will be listed in the notebook itself. Ensure you have the required packages installed by checking the notebook or using `pip install`.

## Getting Started

1. Clone the repository and open the Jupyter notebook.
2. Follow the instructions in the notebook to:
   - Load datasets
   - Train or evaluate the SlotFormer model.
   - Visualize results.


## To run the comparaison of invariant and slot attention: 

To run the comparaison of invariant and slot attention in the /slot folder: 

1. Create and/or activate a new Python virtual environment (tested with 3.11.8).

2. Modify the invariant_slot_attention/lib/input_pipeline.py files by replacing the datasets' PATH_ variables with the actual path on your disk.

3. Modify the mode and training configurations (especially batch_size) if required, by changing the invariant_slot_attention/configs/<dataset>/<equiv_...>.py files.

For Tetrominoes, the default configuration can be used on RTX 3090.
For CLEVR, the batch_size must be set to 32 on RTX 3090.

4. From this repository's root, run the /slot/phd-google-code/slot_test1.sh, modify the python according to what you want to run


## To run the Slot former

### Install :

We recommend using conda for environment setup:

conda create -n slotformer python=3.8.8
conda activate slotformer

Then install PyTorch which is compatible with your cuda setting. In our experiments, we use PyTorch 1.10.1 and CUDA 11.3:

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
The codebase heavily relies on nerv for project template and Trainer. You can easily install it by:

git clone git@github.com:Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
pip install -e .
This will automatically install packages necessary for the project. Additional packages are listed as follows:

pip install pycocotools scikit-image lpips
pip install einops==0.3.2  # tested on 0.3.2, other versions might also work
pip install phyre==0.2.2  # please use the v0.2.2, since the task split might slightly differs between versions
Finally, clone and install this project by:

cd ..  # move out from nerv/
git clone git@github.com:pairlab/SlotFormer.git
cd SlotFormer
pip install -e .
We use wandb for logging, please run wandb login to log in.


### To train slot former on OBJ3D :

We experiment on video prediction task in this dataset.

Pre-train SAVi on OBJ3D videos

Run the following command to train SAVi on OBJ3D videos. Please launch 3 runs and select the best model weight.

python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --fp16 --ddp --cudnn
Alternatively, we provide pre-trained SAVi weight as pretrained/savi_obj3d_params/model_40.pth.

Then, we'll need to extract slots and save them. Please use extract_slots.py and run:

python slotformer/base_slots/extract_slots.py \
    --params slotformer/base_slots/configs/savi_obj3d_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/OBJ3D/slots.pkl')
This will extract slots from OBJ3D videos, and save them into a .pkl file (~692M).

Alternatively, we also provide pre-computed slots as described in benchmark.md.

Video prediction

For the video prediction task, we train SlotFormer over slots, and then evaluate the generated frames' visual quality.

Train SlotFormer on OBJ3D slots

Train a SlotFormer model on extracted slots by running:

python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_obj3d_params.py \
    --fp16 --ddp --cudnn
Alternatively, we provide pre-trained SlotFormer weight as pretrained/slotformer_obj3d_params/model_200.pth.

Evaluate SlotFormer in video prediction

To evaluate the video prediction task, please use test_vp.py and run:

python slotformer/video_prediction/test_vp.py \
    --params slotformer/video_prediction/configs/slotformer_obj3d_params.py \
    --weight $WEIGHT
This will compute and print all the metrics. Besides, it will also save 10 videos for visualization under vis/obj3d/$PARAMS/. If you only want to do visualizations (i.e. not testing the metrics), simply use the --save_num args and set it to a positive value.



### To train slot former on CLEVRER

We experiment on video prediction and VQA task in this dataset.

Pre-train SAVi on CLEVRER videos

Run the following command to train SAVi on CLEVRER videos. Please launch 3 runs and select the best model weight.

python scripts/train.py --task base_slots \
    --params slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --fp16 --ddp --cudnn
Alternatively, we provide pre-trained SAVi weight as pretrained/stosavi_clevrer_params/model_12.pth.

Then, we'll need to extract slots and save them. Please use extract_slots.py and run:

python slotformer/base_slots/extract_slots.py \
    --params slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/CLEVRER/slots.pkl')
This will extract slots from CLEVRER videos, and save them into a .pkl file (~13G).

Alternatively, we also provide pre-computed slots as described in benchmark.md.

Video prediction

For the video prediction task, we train SlotFormer over slots, and then evaluate the generated frames' visual quality, and object trajectories (mask/bbox).

Train SlotFormer on CLEVRER slots

Train a SlotFormer model on extracted slots by running:

python scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --fp16 --ddp --cudnn
Alternatively, we provide pre-trained SlotFormer weight as pretrained/slotformer_clevrer_params/model_80.pth.

Evaluate video prediction results

To evaluate the video prediction task, please use test_vp.py and run:

python slotformer/video_prediction/test_vp.py \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --weight $WEIGHT
This will compute and print all the metrics. Besides, it will also save 10 videos for visualization under vis/clevrer/$PARAMS/. If you only want to do visualizations (i.e. not testing the metrics), simply use the --save_num args and set it to a positive value.

## Results
The project aims to showcase:
- Improved object discovery metrics in an unsupervised setting.
- Accurate predictions of object dynamics using SlotFormer.

## Reference
https://github.com/google-research/slot-attention-video
https://github.com/pairlab/SlotFormer/tree/master

Special thanks to researchers and contributors who developed SlotFormer and the datasets used in this analysis.
