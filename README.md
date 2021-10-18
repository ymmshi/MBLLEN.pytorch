# MBLLEN
This is an unofficial PyTorch implementation of paper "MBLLEN: Low-light Image/Video Enhancement Using CNNs", thank Lv for his work!
![](./src/network.jpg)

## Result 
|  | PSNR	 |  SSIM      |
| ---------  |------------| --------- |
| MBLLEN_Dark| 26.07     |  0.882    | 
| MBLLEN_Low | 25.64     |  0.870    | 

## Installtion
```
pip install -r requirements.txt
```

## Train
1. Download [MBLLEN](http://phi-ai.buaa.edu.cn/project/MBLLEN/index.htm) dataset and unzip all files (Notice: `test_dark` directory contains 200 images but `test` directory only contains 144 images, you have to remove the extra images in `test_dark`).

2. Modify `config.py`

3. `python main.py`


## Test
1. `python infer.py`

2. `python metric.py`

## Reference
1. [Keras version](https://github.com/Lvfeifan/MBLLEN) by [Lvfeifan](https://lvfeifan.github.io)

## Related work
- [x] [MBLLEN.pytorch](https://github.com/ymmshi/MBLLEN.pytorch)