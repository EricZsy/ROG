# Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection

This repo is the official implementation of the ICCV 2023 paper **Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection**
\
Shaoyu Zhang, Chen Chen, and Silong Peng.


## Requirements 
- Python 3.6+
- PyTorch 1.8+
- torchvision 0.9+
- mmdet 2.25
- mmcv 1.4


## Usage
### 1. Install
~~~
# Clone the ROG repository.
git clone https://github.com/EricZsy/ROG.git
cd ROG 

# Create conda environment.
conda create --name rog python=3.8 -y 
conda activate rog

# Install PyTorch.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install mmcv and mmdetection
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmdet==2.25.2 
pip install -v -e .

# Install lvis-api. 
pip install lvis
~~~

### 2. Data
Please download [LVIS dataset](https://www.lvisdataset.org/dataset). The folder `data` should be like this:
~~~
    data
    ├── lvis
    │   ├── lvis_annotations
    │   │   │   ├── lvis_v1_train.json
    │   │   │   ├── lvis_v1_val.json
    │   ├── train2017
    │   │   ├── 000000100582.jpg
    │   │   ├── 000000102411.jpg
    │   │   ├── ......
    │   └── val2017
    │       ├── 000000062808.jpg
    │       ├── 000000119038.jpg
    │       ├── ......
~~~





## Citation
If you find this work useful in your research, please cite:

	@InProceedings{Zhang_2023_ICCV,
        author    = {Zhang, Shaoyu and Chen, Chen and Peng, Silong},
        title     = {Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2023},
        pages     = {18982-18992}
    }