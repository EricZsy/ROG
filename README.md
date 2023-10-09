# Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection

This repo is the official implementation of the ICCV 2023 paper **Reconciling Object-Level and Global-Level Objectives for Long-Tail Detection** 
by Shaoyu Zhang, Chen Chen, and Silong Peng.


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
conda install pytorch torchvision torchaudio cudatoolkit

# Install mmcv and mmdetection.
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmdet==2.25.2 
pip install -v -e .
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

### 3. Train
Use the following commands to train a model.


```train
# Single GPU
python tools/train.py ${CONFIG_FILE}

# Multi GPU distributed training
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

For example, to train a Mask R-CNN model for 12 epochs with ROG:
```train
# Single GPU
python tools/train.py configs/rog/rog_r50_sample1e-3_1x.py

# Multi GPU distributed training (for 4 gpus)
bash ./tools/dist_train.sh configs/rog/rog_r50_sample1e-3_1x.py 4
```  
Other configs can be found at ./configs/rog/. 
You may also use custom loss or sampling method with ROG.


### 4. Test
Use the following commands to test a trained model. 
```test
bash ./tools/dist_test.sh \
 configs/rog/rog_r50_sample1e-3_1x.py work_dirs/rog_r50_sample1e-3_1x.py/latest.pth 4 --eval bbox segm
```



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



## Acknowledgement

Thanks MMDetection team for the wonderful open source project!
