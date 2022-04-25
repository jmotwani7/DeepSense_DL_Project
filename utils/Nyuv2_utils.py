import numpy as np
import matplotlib.pyplot as plt
# import scipy.io as sio
import h5py
import os
from PIL import Image
import random
random.seed(100)

def extraction(mat_dir):
    f = h5py.File(mat_dir)
    images = np.array(f["images"])
    depths = np.array(f["depths"])
    depths = (depths / depths.max()*255).transpose(0,2,1)
    labels = np.array(f["labels"])
    if not (len(images) == len(depths) == len(labels)):
        raise Exception("data corruption! Check or download again!")
    return images, depths, labels

def split(train_p, test_p, images, depths, labels):
    data_index = [i for i in range(len(labels))]
    random.shuffle(data_index)

    images_train = []
    images_test = []
    depths_train = []

    depths_test = []
    labels_train = []
    labels_test = []

    split = []

    data_number = len(images)
    interval = int(data_number * train_p)

    split = {"train": data_index[:interval] , "test": data_index[interval:]}

    for i in data_index[:interval]:
        images_train.append(images[i])
        depths_train.append(depths[i])
        labels_train.append(labels[i])

    for i in data_index[interval:]:
        images_test.append(images[i])
        depths_test.append(depths[i])
        labels_test.append(labels[i])

    return (np.array(images_train),
    np.array(images_test),
    np.array(depths_train),
    np.array(depths_test),
    np.array(labels_train),
    np.array(labels_test),
    split)




def save(root_destination, images, depths, labels,type,index):
    if not os.path.exists(root_destination):
        os.makedirs(root_destination)

    data_number = len(images)

    root = root_destination

    raw_dir = os.path.join(root, type + "_images")
    depth_dir = os.path.join(root, type + "_depths")
    label_dir = os.path.join(root, type + "_labels")

    dir = [raw_dir, depth_dir, label_dir]
    for i in dir:
        if not os.path.exists(i):
            os.makedirs(i)

    for i in range(data_number):
        # raw_pricture extraction to raw_dir
        raw_name = os.path.join(raw_dir , "nyu" + str(index[i]) + ".jpg")
        depth_name = os.path.join(depth_dir, "nyu" + str(index[i]) + ".png")
        label_name = os.path.join(label_dir, "nyu" + str(index[i]) + ".png")

        #extract raw picture
        if not os.path.exists(raw_name):
            raw = images[i]
            raw_r = Image.fromarray(raw[0]).convert("L")
            raw_g = Image.fromarray(raw[1]).convert("L")
            raw_b = Image.fromarray(raw[2]).convert("L")
            raw_img = Image.merge("RGB", (raw_r, raw_g, raw_b)).transpose(Image.ROTATE_270)
            raw_img.save(raw_name, optimize=True)

        #extract depth picture
        if not os.path.exists(depth_name):
            depth_img = Image.fromarray(np.uint8(depths[i]))
            depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
            depth_img.save(depth_name, "PNG", optimize=True)

        #extract label picture
        if not os.path.exists(label_name):
            label_img = Image.fromarray(np.uint8(labels[i])).transpose(Image.ROTATE_270)
            label_img.save(label_name, "PNG", optimize=True)





mat_dir = "../datasets/Nyu_v2/nyu_depth_v2_labeled.mat"
destination_root = "./datasets/Nyu_v2"
images, depths, labels = extraction(mat_dir)
images_train, images_test,depths_train ,depths_test,labels_train ,labels_test,split = split(0.8,0.2,images, depths, labels)

save(os.path.join(destination_root, "train"), images_train,depths_train, labels_train,"train", split["train"])
save(os.path.join(destination_root, "test"), images_test,depths_test, labels_test,"test", split["test"])
