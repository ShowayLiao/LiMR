<div align="center">
  <h1>Lightweight Masked Reconstruction for Real-Time Sensor-Driven Anomaly Detection in Industrial IoT</h1>
</div>


<p align="center">
  <img src=assets/image/LiMRframework.png width="100%">
</p>
This is an official Pytorch to ONNX/TensorRT implementation of the paper "Lightweight Masked Reconstruction for Real-Time Sensor-Driven Anomaly Detection in Industrial IoT".

## Quick start
### Preparation
You can see the more implementation details in [main](https://github.com/ShowayLiao/LiMR/tree/main) branch, while we only introduce how to convert pytorch to ONNX or TensorRT.

You should use this branch to convert the model due to some operation is not support in ONNX. More details could be seen at MobileViT block [file](./models/LiMR/mobileViTv2/mobilevit_v2_block.py).

Before start transformation, confirm that you install the `TensorRT` and `ONNX`.

### Convertation
First, put the weight file at the root directory and rename the file to `best_student_model_175.pth` or change the path in the [file](./ToONNX.py). Then, just start the `./ToONNX.py`, all convertation would be done soon.

### Python inference
Confirm that the parameter in config file named `cfg.TEST.TensorRT.enable` is True, and then change the engine path `cfg.TEST.TensorRT.stu_path` correctly. Then, just start the `main_test.py` and all results from meassurement would be recorded into log in `./logs_and_models`.

### c++ inference
Release soon.

### Acknowledgements
We acknowledge the excellent implementation as following:
[ConvMAE](https://github.com/Alpha-VL/ConvMAE),
[MobileViTv2](https://github.com/apple/ml-cvnets),
[MobileViTv2-pytorch](https://github.com/HowardLi0816/MobileViTv2_pytorch),
[MMR](https://github.com/zhangzilongc/MMR/blob/master/models/MMR/MMR.py)





