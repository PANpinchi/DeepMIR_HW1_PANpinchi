# DeepMIR HW1: Instrument activity detection

## Overview
### 1.  Multi-label music tagging transfer learning:
* Self-supervise representation learning
* Implementation of transfer learning downstream task
* (Huggingface practice)


## Getting Started 
```bash
# Clone the repo:
git clone https://github.com/PANpinchi/DeepMIR_HW1_PANpinchi.git
# Move into the root directory:
cd DeepMIR_HW1_PANpinchi
```
## Environment Settings
```bash
# Create a virtual conda environment:
conda create -n deepmir_hw1 python=3.10

# Activate the environment:
conda activate deepmir_hw1

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```
## Download the Required Data
#### 1. Pre-trained Models
Run the commands below to download the pre-trained model with and without frozen encoder.
```bash
# The pre-trained model without frozen encoder.
gdown --folder https://drive.google.com/drive/folders/1gL3yVAo5Ffd4mAwxU-a812QyIO--jNzX?usp=drive_link
# The pre-trained model with frozen encoder. 
gdown --folder https://drive.google.com/drive/folders/1RrZl45Raiqc3F5FzBWmWDaTwG7pBusCn?usp=drive_link
```
Note: `*.pth` files should be placed in the `/results_classifier` and `/results_classifier_frozen` folders.

#### 2. Datasets
Run the commands below to download the Nsynth datasets.
If you just want to run the inference code, you can download the testing dataset only.
```bash
# Download hw1 TA support folder and unzip it:
gdown --id 1gRykfOOmKJsxppBo3DT7Nm7gMiPGVUAa

unzip hw1.zip
```


You need to unzip the contents and put them in `/hw1`.

#### The data directory structure should follow the below hierarchy.
```
${ROOT}
|-- hw1
|   |-- slakh
|   |   |-- test
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- ...
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |-- train
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- ...
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |-- validation
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |   |-- ...
|   |   |   |-- Trackxxxxx_xx.npy
|   |   |-- test_labels.json
|   |   |-- train_labels.json
|   |   |-- validation_labels.json
|   |-- test_track
|   |   |-- Trackxxxxx.flac
|   |   |-- Trackxxxxx.mid
|   |   |   |-- ...
|   |   |-- Trackxxxxx.flac
|   |   |-- Trackxxxxx.mid
|   |-- class_idx2MIDIClass.json
|   |-- idx2instrument_class.json
|   |-- MIDIClassName2class_idx.json
|   |-- plot_pianoroll.py
|   |-- requirements.txt
```

## 【Task: Multi-label music tagging transfer learning】
* Use a self-supervise audio encoder model and a classifier to implement multilabel instruments classification
* Please predict the instruments activity every 5 seconds

#### Training
```bash
# Training model with frozen encoder
python train.py --frozen
# Training model without frozen encoder
python train.py
```

#### Testing
```bash
# Testing model with frozen encoder (th: 0.2)
python test.py --threshold 0.2 --frozen
# Testing model without frozen encoder (th: 0.2)
python test.py --threshold 0.2
```

#### Plot Piano Roll
```bash
# Plot piano roll by multi-label music tagging model with frozen encoder (th: 0.2)
python plot_pianoroll.py --threshold 0.2 --save_dir results_classifier_frozen
# Plot piano roll by multi-label music tagging model without frozen encoder (th: 0.2)
python plot_pianoroll.py --threshold 0.2 --save_dir results_classifier
```
Note: Results will be stored in the `/results_classifier` and `/results_classifier_frozen` folders.

