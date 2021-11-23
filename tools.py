import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from typing import *
from tqdm import tqdm

import copy

os.chdir('/mnt/nfs/work/liaohuang/od/Mine')


def visualize(json_path, img_id, raw=False, show=True, return_arr=False,
              colors: List[tuple] = None, font_size: float = 0.18, display_thresh=0.,
              save_path='tmp.png'):
    """
    :param save_path:
    :param json_path: prediction path
    :param img_id: image id
    :param raw: if True, bbox and txt won't be shown
    :param show: whether plt show img
    :param return_arr: whether return image array
    :param colors: color of bbox and txt (array of 10 tuple of RGB)
    :param font_size: size of text
    :param display_thresh: bbox won't display if its score is lower than this value
    :return:
    """
    with open(json_path, 'r') as f:
        img_pred = json.load(f)
    img = cv2.imread(f'data/test/{img_id}.png')
    for ii in img_pred:
        if ii['image_id'] != img_id:
            continue
        if ii['score'] < display_thresh:
            continue
        x, y, w, h = ii['bbox']
        # print(x, y, w, h, ii['score'])
        # cv2.rectangle(img, (y, x), (h+y, w+x), (0, 0, 255), 2)
        if colors is not None:
            color = colors[ii['category_id'] % 10]
        else:
            color = (0, 255, 0)
        if raw is not True:
            img = cv2.rectangle(img, (int(x), int(y)), (int(w + x), int(h + y)), color, 1)
            # img = cv2.putText(img, f'{ii["category_id"] % 10}:{int(float(ii["score"]) * 10 % 10):01}',
            #                   (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
            img = cv2.putText(img, f'{ii["category_id"] % 10}',
                              (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
    cv2.imwrite(save_path, img)
    if show is True:
        plt.imshow(np.array(Image.open('tmp.png')))
        plt.show()
    if return_arr is True:
        return img


def visualize_many(json_name, resize: tuple = (100, 150), thresh=0.):
    shown_id = [101130, 101138, 101146, 101179, 101194, 10124,
                101334, 101380, 101412, 101541, 101569, 1005,
                100000, 100009, 100059, 100113, 100190, 100216]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
              (0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
              (127, 127, 255), (127, 255, 0), ]
    all_t: list = []
    for img_id in shown_id:
        arr: np.ndarray = visualize(json_name, img_id=img_id, raw=False, show=False, return_arr=True,
                                    colors=colors, font_size=0.3,
                                    display_thresh=thresh)
        t = torch.tensor(arr.transpose(2, 0, 1))

        trans = transforms.Resize(resize)
        t = trans(t)
        all_t.append(t)
    all_t: torch.Tensor = torch.stack(all_t)
    print(all_t.shape)
    all_t.size()
    grid = make_grid(all_t, nrow=3)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.show()


# img = cv2.imread('data/train/10002.png')
# plt.imshow(img)
# plt.show()
# n = np.load('data/train/10002.npy')
# n
# img.shape
#
# import h5py
#
# dsFile = h5py.File('data/train/digitStruct.mat', 'r')
# d = dsFile['digitStruct']
# ref = d['name'][1][0]
# dsFile[ref][2]


def to_new_pred():
    with open('prediction.json', 'r') as f:
        p = json.load(f)
    new_p = []
    for item in p:
        # if item['score'] < 0.4:
        #     continue
        new_item = {}
        for key in item:
            if key == 'bbox':
                # x1, y1, x2, y2 => x1, y1, w, h => y1, x1, h, w
                old_bbox = item[key]
                new_bbox = [old_bbox[0], old_bbox[1], old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]
                new_bbox = [new_bbox[1], new_bbox[0], new_bbox[3], new_bbox[2]]
                new_item[key] = new_bbox
            else:
                new_item[key] = item[key]
        new_p.append(new_item)

    with open('new_p.json', 'w') as file:
        json.dump(new_p, file)


# to_new_pred()

# visualize_many(json_name='new_p.json', resize=(180, 128))
visualize_many(json_name='prediction.json', resize=(128, 128), thresh=0.)
# visualize_many(json_name='prediction.json', resize=(128, 128), thresh=0.2)
# visualize_many(json_name='thresh_3_pred.json', resize=(128, 128))


def plot_save_testing_img(save_n, save_path, json_path, test_img_path, display_thresh=0., font_size=2.):
    for root, dirs, files in os.walk(test_img_path):
        it = enumerate(files)
        for i, file in it:
            if i >= save_n:
                break
            print(f'{i + 1}/{save_n}', end='\r')
            visualize(json_path=json_path,
                      img_id=int(file.split('.')[-2]),
                      raw=False,
                      show=False,
                      return_arr=False,
                      colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
                              (0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                              (127, 127, 255), (127, 255, 0), ],
                      font_size=font_size,
                      display_thresh=display_thresh,
                      save_path=os.path.join(save_path, file))


plot_save_testing_img(save_n=25,
                      # json_path='thresh_3_pred.json',
                      json_path='prediction.json',
                      save_path='/mnt/nfs/work/liaohuang/od/Mine/with_bbox/',
                      test_img_path='/mnt/nfs/work/liaohuang/od/Mine/data/test',
                      display_thresh=0.,
                      font_size=1.)

# Get all scores
with open('prediction.json', 'r') as f:
    j = json.load(f)
appeared = []
id_scores_dict: Dict[int, List[float]] = {}
for obj in j:
    img_id = int(obj['image_id'])
    score = float(obj['score'])
    if img_id in appeared:
        id_scores_dict[img_id].append(score)
    else:
        id_scores_dict[img_id] = [score]
        appeared.append(img_id)
# Calculate each id's score threshold
# 3 highest scores
n_highest = 5
id_thresh_dict: Dict[int, float] = {}
for img_id in id_scores_dict:
    scores: List = id_scores_dict[img_id]
    scores.sort(reverse=True)
    thresh: float = scores[min(n_highest - 1, len(scores) - 1)]
    id_thresh_dict[img_id] = thresh
# Make a new json
new_j = []
for obj in j:
    img_id = int(obj['image_id'])
    score = float(obj['score'])
    if score < id_thresh_dict[img_id]:
        continue
    else:
        new_j.append(obj)
with open('thresh_3_pred.json', 'w') as f:
    json.dump(new_j, f)
    #

#
# with open('thresh_3_pred.json', 'r') as f:
#     jj = json.load(f)
#
# for ooo in jj:
#     print(ooo['image_id'])

for i in range(10040,10145):
    # a = np.array(Image.open(f'data/train/{i}.png'))
    # plt.imshow(a)
    # plt.show()
    with open(f'../svhn/labels/{i}.txt') as f:
        print(f.read().split(' ')[0])
    # print(np.load(f'../svhn/images/{i}.npy'))