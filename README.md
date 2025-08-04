<div align="center">
  <h1>Lightweight Masked Reconstruction for Real-Time Sensor-Driven Anomaly Detection in Industrial IoT</h1>
</div>
<p align="center">
  <img src=assets/image/LiMRframework.png width="100%">
</p>
This is an official Pytorch implementation of the paper "Lightweight Masked Reconstruction for Real-Time Sensor-Driven Anomaly Detection in Industrial IoT".

## Quick start
### Dataset Preparation
We experimence on AeBAD blade dataset and MVTec AD.You can download the AeBAD from [here](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) and MVTec AD from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Then put the dataset in the `./datasets` folder, and the structure should be like this:
```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- object (bottle, etc.)
                |-- train
                |-- test
                |-- ground_truth
    |-- AeBAD
        |-- AeBAD_S
        |-- AeBAD_V
```
### Pre-trained models
Download the pre-trained model of MobileViTv2 for ours model at [here](https://1drv.ms/u/c/6b209148572e70d3/ESC3x54E77JPvRTwUPhlDg8BohB11CxU20xy4rf3noqLLg?e=ebKAOJ). You can also download the pretrained model from timm library, then use `./utils/weight_trans.py` to change the keys of the model to fit our model.

### Virtual Environment
We recommend using a virtual environment as follows:
```
python>=3.10
pytorch>=1.12
cuda>=11.6
```
More details can be found in the `requirements.txt` file.

### Train and Test for AeBAD,MVTec
Corresponding config for different datasets can be found in `./method_config/`. To change the datasets, you can change the default config path in `./utils/parser_.py` and start by `main.py` or just start by following code:
```bash
sh mvtec_run.sh
```
```bash
sh AeBAD_S_run.sh
```
```bash
sh AeBAD_V_run.sh
```
Once you start the training, the model will be saved in `./logs_and_models` which you can define in config file named `OUTPUT_ROOT_DIR`. After training, testing will be automatically performed and the results will be saved in same directory. You can also only start the testing by `main_test.py` after changing the default model path.

### Results
Visualized and numerical results can be seen as follows. You can download the best model from [here](). More details can be found in [paper]().
<p align="center">
  <img src=assets/image/aebads.png width="100%">
</p>
<p align="center">
  <img src=assets/image/aebads_table.png width="100%">
</p>

<p align="center">
  <img src=assets/image/aebadv.png width="100%">
</p>
<p align="center">
  <img src=assets/image/aebadv_table.png width="100%">
</p>
<p align="center">
  <img src=assets/image/aebads_fig.png width="100%">
  <img src=assets/image/aebadv_fig.png width="100%">
</p>

### ONNX and TensorRT
We provide official implementation of the transformation between pytorch and ONNX, which can be also converted to TensorRT engine file. More details could be seen in the [ONNX](https://github.com/ShowayLiao/LiMR/tree/onnx) branch. 

c++ implementation via TensorRT will be released in the future.

### Acknowledgements
We acknowledge the excellent implementation as following:
[ConvMAE](https://github.com/Alpha-VL/ConvMAE),
[MobileViTv2](https://github.com/apple/ml-cvnets),
[MobileViTv2-pytorch](https://github.com/HowardLi0816/MobileViTv2_pytorch),
[MMR](https://github.com/zhangzilongc/MMR/blob/master/models/MMR/MMR.py)






