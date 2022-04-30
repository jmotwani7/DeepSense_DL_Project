import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import h5py
import flow_transforms
import random
import json
from pathlib import Path
from augmentations.augmentations import Compose, RandomRotate, Scale


# class NyuGenerator(torch.utils.data.Dataset):
#     def __init__(self, root, split="train", is_transform=False, img_size=(480, 640), augmentations=None, img_norm=True):
#         self.root = root
#         self.is_transform = is_transform
#         self.augmentations = augmentations
#         self.img_size = img_size
#         self.split = split
#         self.files = {}
#         for split in ["train", "test"]:
#             file_list = glob.glob(root + "/" + split + "/" + split + "_images/**")
#             # print(file_list)
#             self.files[split] = file_list
#
#     def __len__(self):
#         # return int(np.ceil(len(self.data) / self.batch_size))
#         return len(self.files[self.split])
#
#     def __getitem__(self, id):
#         img_path = self.files[self.split][id]
#         img_name = os.path.split(self.files[self.split][0])[-1][:-4]
#         label_path = self.root + "/" + self.split + "/" + self.split + "_labels/" + img_name + ".png"
#
#         img = np.asarray(Image.open(img_path))
#
#         label = np.asarray(Image.open(label_path))
#
#         if self.is_transform:
#             img, label = self.transform(img, label)
#
#         if self.augmentations is not None:
#             img, label = self.augmentations(img, label)
#
#         return img, label
#
#     def transform(self, img, label):
#         img = Image.fromarray(img).resize((480, 640), Image.ANTIALIAS)
#         label = Image.fromarray(label).resize((480, 640), Image.ANTIALIAS)
#
#         return np.asarray(img), np.asarray(label)


class NyuDatasetLoader(Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0)
        # img = self.imgs[img_idx]
        dpt = self.dpts[img_idx].transpose(1, 0)
        # dpt = self.dpts[img_idx]

        # image = Image.fromarray(np.uint8(img))
        # depth = Image.fromarray(np.uint8(dpt))

        # image.save('img1.png')

        # input_transform = transforms.Compose([flow_transforms.Scale(228)])
        # input_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
        input_transform = transforms.Compose([flow_transforms.Scale(228),
                                              flow_transforms.ArrayToTensor()])
        # target_depth_transform = transforms.Compose([flow_transforms.Scale(228)])
        # target_depth_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
        target_depth_transform = transforms.Compose([flow_transforms.Scale_Single(228),
                                                     flow_transforms.ArrayToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)

        # image = Image.fromarray(np.uint8(img))
        # image.save('img2.png')

        return img, dpt

    def __len__(self):
        return len(self.lists)


# def get_NYU_trainloader(dataset_root, split='train', batch_size=28, shuffle=True):
#     # root = "datasets/Nyu_v2"
#     augmentations = Compose([Scale(512), RandomRotate(10)])
#     nyu_train = NyuDatasetLoader(dataset_root, split=split, is_transform=True, augmentations=augmentations)
#     return DataLoader(nyu_train, batch_size)


def load_test_train_ids(file_path):
    with open(file_path, 'r') as f:
        ids = json.loads(f.read())
        return ids['train'], ids['test']


def save_test_train_ids(file_path, last_id=1498):
    if not Path(file_path).parent.is_dir():
        Path(file_path).parent.mkdir(exist_ok=True)
    all_ids = [i for i in range(last_id)]
    train_ids = random.sample(all_ids, int(last_id * 0.8))
    test_ids = [i for i in all_ids if i not in train_ids]
    ids_dict = {'train': train_ids, 'test': test_ids}
    with open(file_path, 'w') as f:
        f.write(json.dumps(ids_dict))


def get_nyuv2_test_train_dataloaders(dataset_path, train_ids, test_ids, batch_size=32, last_id=1448):
    # all_ids = [i for i in range(last_id)]
    # train_ids = random.sample(all_ids, int(last_id * 0.8))
    # test_ids = [i for i in all_ids if i not in train_ids]
    train_ds = NyuDatasetLoader(dataset_path, train_ids)
    test_ds = NyuDatasetLoader(dataset_path, test_ids)
    return DataLoader(train_ds, batch_size), DataLoader(test_ds, batch_size)
