# Description 

- This is my project of tensorflow model for microcontroller now it is only sute for tflite microcontroller 

- There are only mobilenetv1 and mobilenetv3 implemented,the converting between mobilenetv3 and mobilennetv3.tflite has a degradation problem you may check this issue to catch problem [conversion issue](https://github.com/tensorflow/models/issues/9287)

- onnx2tflite tool is implicated just find it in [forx_onnx/onnx2tflite.py](./for_onnx/onnx2tflite.py)

# TODO

- [ ] Chinese readme document
- [ ] merge data gengerate code and train code

# Compitation

- The following test is only test on 2024_05_06_val data_set with qvga image format and brightness from 200 to 800,`the image is collected fom openart and processed which crope to simulate real situation process code can get form enhance_img branch`

| model name    | qvga_200 | qvga_400 | qvga_600 | qvga_800 |
| :--:          |    :--:  |   :--:   |     :--: |    :--:  |
|mobilenetv1_128| 23.483%  | 27.461%  | 21.429%  | 44.083%  |
|djl_tupian_128 | 7.812%   | 4.145%   | 2.551%   | 2.663%   |