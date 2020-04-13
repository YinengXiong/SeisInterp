# Seismic Data Interpolation using Convolutional Neural Network

## Prerequisites
- Linux
- Python 3
- Nvidia GPU + CUDA cuDNN

## Requirements
- Pytorch >= 0.4
- opencv-python
- numpy

## Getting Started
### Installation

* Install [PyTorch](https://pytorch.org/) and other dependencies
```Shell
pip install -r requirements.txt
```

* Clone this repo:
```Shell
git clone https://github.com/YinengXiong/SeisInterp.git
cd SeisInterp
```

### Datasets
Download dataset and place it in ./SeisInterp/Data/ \
The dataset has 96 train, 12 validation and 12 testing shot record, respectively.
Each shot record has the size of `500 X 6001` (500 traces with 10m interval, 6001 time sampling points with 0.5 ms)

[Google Drive](https://drive.google.com/file/d/10kZO2y1LcoWkupEztTkpvd3aPymXV_mb/view?usp=sharing) | [Baidu Downloads](https://pan.baidu.com/s/1bc7g_Y3b09S31mQJXjWXFQ) Code: q97i

### SeisInterp Train/Test
For example, if you want to train an interpolation model which needs pre-interpolation and only interpolate on X- direction at the scale of 4 

```bash
python train.py --gpu 0 --dataroot ./Data/MarmousiP20HzAGC500/ --num_traces 500 --nComp 1 --prefix shotp --scale 4 --diretion 0 --arch vdsr
```

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{xiong2019efficient,
  title={Efficient Seismic Data Interpolation Using Deep Convolutional Networks and Transfer Learning},
  author={Xiong, Y and Cheng, J},
  booktitle={81st EAGE Conference and Exhibition 2019},
  volume={2019},
  number={1},
  pages={1--5},
  year={2019},
  organization={European Association of Geoscientists \& Engineers}
}
```