# SVHN Yolov5

![example image](data/images/example.png)

## Preprocess

### 1. Prepare data

Make a directory named "svhn" in repo root, download and unzip the train.zip and test.zip in it, arange like the below structure.

P.S. The data used in this repo is provided by the course VRDL.

```text
repo_root
  +- svhn
  | + train
  | | + 1.png
  | | + ...
  | | + 33402.png
  | | + digitStruct.mat
  | + test
  | | + 1.png
  | | + ...
  | | + 13068.png
  +- data
  +- models
  +- utils
  ...... etc
```

### 2. Preprocess

```shell script
cd svhn_yolov5
python3 preprocess.py
```

## Reference

- preprocess.py is revised from [here](https://github.com/072jiajia/CVDL_HW2/blob/main/prepare.py). 
- yolo v5: https://github.com/ultralytics/yolov5    
- SVHN: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. (http://ufldl.stanford.edu/housenumbers)


## TODO

- [ ] Add dataset reference
