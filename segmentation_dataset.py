import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SegDataset(Dataset):

    def __init__(self, parent_dir, image_dir, mask_dir, augmentation=None):
        self.image_list = glob.glob(os.path.join(parent_dir, image_dir, "*"))
        self.image_list.sort()
        self.mask_list = glob.glob(os.path.join(parent_dir, mask_dir, "*"))
        self.mask_list.sort()
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        y = Image.open(self.mask_list[index]).convert('1', dither=Image.NONE)
        y_size = y.size

        x = Image.open(self.image_list[index]).convert('RGB')
        if x.size != y_size:
            x = x.resize(y_size, resample=Image.BILINEAR)
        if self.augmentation is not None:
            y = np.array(y).astype(float)
            transformed = self.augmentation(image=np.array(x), mask=y)
            x = transformed["image"]
            y = transformed["mask"]
        else:
            y = transforms.ToTensor()(y)
            x = transforms.ToTensor()(x)

        return x, y


class SegDatasetV2(Dataset):

    def __init__(self, image_list, mask_list, tiling_factor=(1, 1), augmentation=None):

        self.image_list = image_list
        self.image_list.sort()
        self.mask_list = mask_list
        self.mask_list.sort()
        self.tiling_factor = tiling_factor
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_list) * self.tiling_factor[0] * self.tiling_factor[1]

    def get_tile(self, index):
        num_of_tiles = self.tiling_factor[0] * self.tiling_factor[1]
        image_index = index // num_of_tiles
        tile_index = index % num_of_tiles

        y = Image.open(self.mask_list[image_index]).convert('1', dither=Image.NONE)
        y_size = y.size

        x = Image.open(self.image_list[image_index]).convert('RGB')
        if x.size != y_size:
            x = x.resize(y_size, resample=Image.BILINEAR)

        delta_height = y.size[1] // self.tiling_factor[0]
        crop_height = delta_height * (tile_index // self.tiling_factor[1])
        delta_width = y.size[0] // self.tiling_factor[1]
        crop_width = delta_width * (tile_index % self.tiling_factor[1])

        x = x.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))
        y = y.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))

        return x, y

    def __getitem__(self, index):
        x, y = self.get_tile(index)

        if self.augmentation is not None:
            y = np.array(y).astype(float)
            transformed = self.augmentation(image=np.array(x), mask=y)
            x = transformed["image"]
            y = transformed["mask"]
        else:
            y = transforms.ToTensor()(y)
            x = transforms.ToTensor()(x)

        return x, y


class MulticlassSegDataset(Dataset):

    def __init__(self, image_list, mask_lists, tiling_factor=(1, 1), augmentation=None, background_channel=False):

        self.image_list = image_list
        self.mask_lists = mask_lists
        self.tiling_factor = tiling_factor
        self.augmentation = augmentation
        self.background_channel = background_channel

    def __len__(self):
        return len(self.image_list) * self.tiling_factor[0] * self.tiling_factor[1]

    def get_tile(self, index):
        num_of_tiles = self.tiling_factor[0] * self.tiling_factor[1]
        image_index = index // num_of_tiles
        tile_index = index % num_of_tiles

        ys = tuple([Image.open(l[image_index]).convert('1', dither=Image.NONE) for l in self.mask_lists])
        y_size = ys[0].size

        x = Image.open(self.image_list[image_index]).convert('RGB')
        if x.size != y_size:
            x = x.resize(y_size, resample=Image.BILINEAR)

        delta_height = y_size[1] // self.tiling_factor[0]
        crop_height = delta_height * (tile_index // self.tiling_factor[1])
        delta_width = y_size[0] // self.tiling_factor[1]
        crop_width = delta_width * (tile_index % self.tiling_factor[1])

        x = x.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))
        y = tuple([y.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height)) for y in ys])

        return x, y

    def __getitem__(self, index):
        x, y = self.get_tile(index)

        if self.augmentation is not None:
            y = [np.array(m).astype(float) for m in y]
            transformed = self.augmentation(image=np.array(x), masks=y)
            x = transformed["image"]
            y = tuple(transformed["masks"])
        else:
            y = tuple([transforms.ToTensor()(m) for m in y])
            x = transforms.ToTensor()(x)

        #fix albmumentation bug
        if type(y[0]) == np.ndarray:
            y = tuple([torch.from_numpy(m) for m in y])

        if self.background_channel:
            y = torch.logical_not(torch.logical_or(*y)), *y

        return x, y


class SemisupervisedSegDataset(Dataset):

    def __init__(self, image_folder, mask_folder, masks_list, tiling_factor=(1, 1), augmentation=None):

        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.masks_list = masks_list
        self.tiling_factor = tiling_factor
        self.augmentation = augmentation

    def __len__(self):
                return len(self.masks_list) * self.tiling_factor[0] * self.tiling_factor[1]

    def get_tile(self, index):
        num_of_tiles = self.tiling_factor[0] * self.tiling_factor[1]
        image_index = index // num_of_tiles
        tile_index = index % num_of_tiles

        mask_name = self.masks_list[image_index]
        mask_path = os.path.join(self.mask_folder, mask_name)
        point_path = os.path.join(self.mask_folder, mask_name[:-12] + "point_" + mask_name[-7:])
        image_path = os.path.join(self.image_folder, mask_name[:-13] + ".png")

        y = Image.open(mask_path).convert('1', dither=Image.NONE)
        y_size = y.size

        p = Image.open(point_path).convert('1', dither=Image.NONE)

        assert y.size == p.size

        x = Image.open(image_path).convert('RGB')
        if x.size != y_size:
            x = x.resize(y_size, resample=Image.BILINEAR)

        delta_height = y.size[1] // self.tiling_factor[0]
        crop_height = delta_height * (tile_index // self.tiling_factor[1])
        delta_width = y.size[0] // self.tiling_factor[1]
        crop_width = delta_width * (tile_index % self.tiling_factor[1])

        x = x.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))
        y = y.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))
        p = p.crop(box=(crop_width, crop_height, crop_width + delta_width, crop_height + delta_height))

        return x, y, p

    def __getitem__(self, index):
        x, y, p = self.get_tile(index)

        if self.augmentation is not None:
            y = np.array(y).astype(float)
            p = np.array(p).astype(float)
            transformed = self.augmentation(image=np.array(x), masks=(y, p))
            x = transformed["image"]
            y, p = tuple(transformed["masks"])
        else:
            y = transforms.ToTensor()(y)
            p = transforms.ToTensor()(p)
            x = transforms.ToTensor()(x)

        return x, y, p

