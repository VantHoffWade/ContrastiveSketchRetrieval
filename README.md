# <center>ContrastiveSketchRetrieval</center>

An experimental repo, we plan to use sketchRNN as the sketch encoder and 
ULIP as the image encoder, for an overview you can refer to the [picture](images/overview.png)

## 📻 Notice

## ⚙️ Quickstart
### 1️⃣ Set up environment
First install Pytorch 1.10.0 or later and torchvision, and other 
dependencies. A machine with CUDA 11.3 or later is recommended.
```
conda create -n sketch python=3.8
conda activate sketch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
cd ContrastiveSketchRetrieval
pip install -r requirements.txt
```
### 2️⃣ Download preprocessed datasets
Three datasets are available for download:
#### Sketchy
Download [Sketchy](https://pan.quark.cn/s/a5c094c5d3f6) 
from Quark Drive.

## 🔥 Train&Eval
Make sure you have downloaded the datasets and put them in the desired location. 
The [options.py](/options.py) is the configuration file, change the 
`--data_path` and `--dataset` arguments if you place the dataset in a 
different directory.

train the model:
```
python train.py
```
eval the model:
```
python test.py
```