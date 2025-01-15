# Unsupervised Object Discovery and Dynamics Prediction

## Overview
This project evaluates various approaches for unsupervised object discovery and dynamics prediction using video datasets. The primary focus is on discovering object representations, understanding their interactions, and predicting their future states. The methodology uses SlotFormer, a Transformer-based model that learns object dynamics from video clips by decomposing scenes into objects and modeling their temporal interactions.

## Features
- **Object Discovery (Savi Model)**: Exploration of unsupervised techniques for detecting objects in dynamic scenes.
- **Dynamics Prediction (SlotFormer Model)**: Prediction of future object states based on learned temporal interactions.

## Contents
The notebook includes:
1. **Motivation**: Discussing the importance of unsupervised learning in object detection and dynamics prediction.
2. **Background on SlotFormer**: Detailed description of the SlotFormer model and its capabilities.
3. **Experiments**: Evaluations based on video clip datasets, focusing on:
   - Object discovery performance.
   - Dynamics prediction accuracy.

## Dataset & Checkpoints
To use the checkpoints and datasets, follow these steps:

### To load the checkpoints:
Download the following from the Google Drive link:  
[Checkpoint and Log Files](https://drive.google.com/drive/folders/1XdpecT6CsHtgXxovH20Blvl7UAwO5sUS?usp=share_link)

- **/logs**: Contains the original SA and ISA files for Slot Attention and Invariant Slot Attention.
- **/checkpoints**: Contains the SlotFormer model checkpoints.

### To load the datasets:
Download the relevant datasets and store them on your machine. Use the following commands:

#### Tetrominoes dataset:
```bash
wget https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords
```

#### CLEVR dataset:
```bash
wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords
```

After downloading, modify the `PATH_` variables in the `invariant_slot_attention/lib/input_pipeline.py` file to point to the local paths of the datasets.

#### OBJ3D dataset:
This dataset is from G-SWM. Download it manually from [this link](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view) and place it in the `\data\OBJ3D` folder.

Since the OBJ3D videos are already extracted to frames, no additional processing is required.

#### CLEVRER dataset:
Download CLEVRER from the official website: [CLEVRER Dataset](http://clevrer.csail.mit.edu).  
Ensure you download the following files:
- Training and Validation Videos
- Annotations for all tasks
- Object Masks and Attributes for video prediction tasks

To accelerate training, you can convert videos to frames in advance by running:
```bash
python scripts/data_preproc/clevrer_video2frames.py
```

## Requirements
- The exact dependencies are listed in the notebook. Use `pip install` to install the necessary libraries.

## Getting Started

1. Clone the repository and open the Jupyter notebook.
2. Follow the instructions to:
   - Load datasets
   - Train or evaluate the SlotFormer model
   - Visualize results

## Comparing Invariant and Slot Attention

To compare Invariant and Slot Attention, follow these steps:

1. Create or activate a new Python virtual environment (tested with Python 3.11.8).
2. Update the `PATH_` variables in the `invariant_slot_attention/lib/input_pipeline.py` file to point to the correct paths.
3. Adjust configuration settings (e.g., batch size) in the `invariant_slot_attention/configs/<dataset>/<equiv_...>.py` files.
   - For Tetrominoes: Use the default configuration on an RTX 3090.
   - For CLEVR: Set batch size to 32 on RTX 3090.
4. Run the comparison script:
```bash
/slot/phd-google-code/slot_test1.sh
```
Modify the Python script to run the desired experiment.

## Running SlotFormer

### Installation:

Use **conda** to set up the environment:

```bash
conda create -n slotformer python=3.8.8
conda activate slotformer
```

Install PyTorch compatible with your CUDA setup (e.g., PyTorch 1.10.1, CUDA 11.3):

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Install the **nerv** project template:

```bash
git clone git@github.com:Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
pip install -e .
```

Then, install additional dependencies:

```bash
pip install pycocotools scikit-image lpips einops==0.3.2 phyre==0.2.2
```

Finally, clone and install this project:

```bash
cd ..  # move out of nerv/
git clone git@github.com:pairlab/SlotFormer.git
cd SlotFormer
pip install -e .
```

Log in to **wandb** for logging:

```bash
wandb login
```

### To train SlotFormer on OBJ3D

1. Pre-train SAVi on OBJ3D videos:
```bash
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/savi_obj3d_params.py --fp16 --ddp --cudnn
```
Alternatively, use the pre-trained model:  
`pretrained/savi_obj3d_params/model_40.pth`

2. Extract slots from OBJ3D videos:
```bash
python slotformer/base_slots/extract_slots.py --params slotformer/base_slots/configs/savi_obj3d_params.py --weight $WEIGHT --save_path $SAVE_PATH
```

### Video Prediction on OBJ3D

1. Train SlotFormer on extracted OBJ3D slots:
```bash
python scripts/train.py --task video_prediction --params slotformer/video_prediction/configs/slotformer_obj3d_params.py --fp16 --ddp --cudnn
```
Alternatively, use the pre-trained model:  
`pretrained/slotformer_obj3d_params/model_200.pth`

2. Evaluate video prediction:
```bash
python slotformer/video_prediction/test_vp.py --params slotformer/video_prediction/configs/slotformer_obj3d_params.py --weight $WEIGHT
```

### To train SlotFormer on CLEVRER

1. Pre-train SAVi on CLEVRER videos:
```bash
python scripts/train.py --task base_slots --params slotformer/base_slots/configs/stosavi_clevrer_params.py --fp16 --ddp --cudnn
```
Alternatively, use the pre-trained model:  
`pretrained/stosavi_clevrer_params/model_12.pth`

2. Extract slots from CLEVRER videos:
```bash
python slotformer/base_slots/extract_slots.py --params slotformer/base_slots/configs/stosavi_clevrer_params.py --weight $WEIGHT --save_path $SAVE_PATH
```

### Video Prediction on CLEVRER

1. Train SlotFormer on CLEVRER slots:
```bash
python scripts/train.py --task video_prediction --params slotformer/video_prediction/configs/slotformer_clevrer_params.py --fp16 --ddp --cudnn
```
Alternatively, use the pre-trained model:  
`pretrained/slotformer_clevrer_params/model_80.pth`

2. Evaluate video prediction:
```bash
python slotformer/video_prediction/test_vp.py --params slotformer/video_prediction/configs/slotformer_clevrer_params.py --weight $WEIGHT
```

## Results
This project demonstrates:
- Improved object discovery performance in unsupervised settings.
- Accurate object dynamics predictions using SlotFormer.

## References
- [Slot Attention Video](https://github.com/google-research/slot-attention-video)
- [SlotFormer](https://github.com/pairlab/SlotFormer/tree/master)

Special thanks to the researchers and contributors of SlotFormer and the datasets used in this analysis.

