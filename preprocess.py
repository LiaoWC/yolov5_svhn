import h5py
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


class BBox:
    ''' Bounding Box '''

    def __init__(self):
        self.label = ""  # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0


class DigitStruct:
    ''' filename and BBox list '''

    def __init__(self):
        self.name = None  # Image file name
        self.bboxList = None  # List of BBox structs


def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup


def readString(strRef, dsFile):
    ''' Reads a string from the file using its reference '''
    strObj = dsFile[strRef]
    str = ''.join(chr(i[0]) for i in strObj)
    return str


def readInt(intArray, dsFile):
    ''' Reads an integer value from the file '''
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else:  # Assuming value type
        intVal = int(intRef)
    return intVal


def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal


def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList


def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name


def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj


def fetch_train_labels():
    ''' Generate annotations of images
    each xxx.png has a file of annotations named xxx.npy
    '''
    dsFileName = 'svhn/train/digitStruct.mat'
    i = 0
    for dsObj in tqdm(yieldNextDigitStruct(dsFileName), total=33402):
        i += 1
        array = []
        for bbox in dsObj.bboxList:
            array.append([bbox.left, bbox.top, bbox.left +
                          bbox.width, bbox.top + bbox.height, bbox.label % 10])

        array = np.array(array)
        filename = 'svhn/train/' + dsObj.name[:-4] + '.npy'
        np.save(filename, array)


def train_valid_split():
    np.random.seed(6666)  # fix seed
    perm = np.random.permutation(33402)
    train_idices = perm[:33000]
    valid_indices = perm[33000:]

    with open('svhn/train.txt', 'w') as f:
        for idx in train_idices:
            f.write(f'svhn/train/{idx}.png\n')

    with open('svhn/valid.txt', 'w') as f:
        for idx in valid_indices:
            f.write(f'svhn/train/{idx}.png\n')


def make_label_txt():
    os.mkdir('svhn/labels')
    for i in tqdm(range(1, 33402 + 1)):  # for each img
        arr = np.load(f'svhn/train/{i}.npy')
        img_h, img_w = np.array(Image.open(f'svhn/train/{i}.png')).shape[:2]
        with open(f'svhn/labels/{i}.txt', 'w') as f:
            for obj in arr:
                (top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                 category_id) = obj.tolist()

                def clamp(v, min_v, max_v):
                    if v < min_v:
                        return min_v
                    if v > max_v:
                        return max_v
                    else:
                        return v

                top_left_x = clamp(top_left_x, 0, img_w - 1)
                top_left_y = clamp(top_left_y, 0, img_h - 1)
                bottom_right_x = clamp(bottom_right_x, 0, img_w - 1)
                bottom_right_y = clamp(bottom_right_y, 0, img_h - 1)

                x_center = (top_left_x + bottom_right_x) / (2. * img_w)
                y_center = (top_left_y + bottom_right_y) / (2. * img_h)
                width = (bottom_right_x - top_left_x) / img_w
                height = (bottom_right_y - top_left_y) / img_h
                # if i == 12668:
                #     print(obj.tolist())
                #     print(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                #     print(x_center, y_center, width, height, img_w, img_h)
                f.write(f'{category_id % 10} {x_center} {y_center} {width} {height}\n')


def detect_test_filenames():
    os.mkdir('svhn/test/labels')
    with open('svhn/test.txt', 'w') as f:
        for roots, dirs, files in os.walk('svhn/test'):
            for file in files:
                f.write(f'svhn/test/images/{file}\n')
                with open(f'svhn/test/labels/{file.split(".")[0]}.txt', 'w') as ff:
                    ff.write('')


if __name__ == "__main__":
    # print('Fetching training labels ...')
    # fetch_train_labels()
    # print('Train valid split ...')
    # train_valid_split()
    # print('Making annotation file for yolo_v5 ...')
    # make_label_txt()
    print('Detecting test image filenames...')
    detect_test_filenames()
    print('Done!')
