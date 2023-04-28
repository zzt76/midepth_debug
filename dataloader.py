# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DepthDataLoader(object):
    def __init__(self, b_distributed, batch_size, num_threads, mode, dataset, input_height, input_width, filenames_file,
                 do_kb_crop=False, do_random_rotate=False, data_path=None, gt_path=None):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(mode, dataset, input_height, input_width, filenames_file, do_kb_crop, do_random_rotate, data_path, gt_path)
            if b_distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=num_threads,
                                   prefetch_factor=4,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(mode, dataset, input_height, input_width, filenames_file, do_kb_crop, do_random_rotate, data_path, gt_path)
            if b_distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(mode, dataset, input_height, input_width, filenames_file, do_kb_crop, do_random_rotate, data_path, gt_path)
            self.data = DataLoader(self.testing_samples,
                                   batch_size,
                                   shuffle=False,
                                   num_workers=num_threads,
                                   pin_memory=True,
                                   persistent_workers=True)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, mode, dataset, input_height, input_width, filenames_file, do_kb_crop, do_random_rotate, data_path, gt_path):
        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.dataset = dataset
        self.input_height = input_height
        self.input_width = input_width
        self.do_kb_crop = do_kb_crop
        self.do_random_rotate = do_random_rotate
        self.data_path = data_path
        self.gt_path = gt_path
        self.degree = 2.5

        self.normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.mode == 'train':
            # if self.dataset == 'kitti' and self.use_right is True and random.random() > 0.5:
            if self.dataset == 'kitti':
                image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
                name = image_path.split('/')[-4] + '_' + image_path.split('/')[-1].split('.')[0]
            elif self.dataset == 'nyu':
                image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
                name = image_path.split('/')[-1].split('.')[0]
            elif self.dataset == 'mi':
                image_path = sample_path.split(',')[0]
                depth_path = sample_path.split(',')[1].strip('\n')
                name = image_path.split('/')[-1].split('.')[0]

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))
            elif self.dataset == 'mi':
                depth_gt = ImageOps.invert(ImageOps.grayscale(depth_gt))

            if self.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # image.save("./test/image.png")
            # depth_gt.save("./test/depth.png")

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            elif self.dataset == 'mi':
                depth_gt = depth_gt / 255.0 * 10.0
            else:
                depth_gt = depth_gt / 255.0

            image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            # Image.fromarray(np.uint8(image*255.0)).save('./image.png')
            # Image.fromarray(np.uint8(depth_gt[:, :, 0]*255.0)).save('./depth.png')

            image = self.to_tensor(image)
            image = self.normalize_imagenet(image)
            depth_gt = self.to_tensor(depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'name': name}

        else:
            if self.dataset == 'kitti':
                image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
                name = image_path.split('/')[-4] + '_' + image_path.split('/')[-1].split('.')[0]
            elif self.dataset == 'nyu':
                image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
                name = image_path.split('/')[-1].split('.')[0]
            elif self.dataset == 'mi':
                image_path = sample_path.split(',')[0]
                depth_path = sample_path.split(',')[1].strip('\n')
                name = image_path.split('/')[-1].split('.')[0]

            image = Image.open(image_path)

            has_valid_depth = False
            try:
                depth_gt = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = None
                # print('Missing gt for {}'.format(image_path))

            if self.dataset == 'mi':
                depth_gt = ImageOps.invert(ImageOps.grayscale(depth_gt))
            # if self.dataset == 'nyu':
            #     depth_gt = depth_gt.crop((43, 45, 608, 472))
            #     image = image.crop((43, 45, 608, 472))

            image = np.asarray(image, dtype=np.float32) / 255.0
            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                if self.dataset == 'nyu':
                    depth_gt = depth_gt / 1000.0
                elif self.dataset == 'mi':
                    depth_gt = depth_gt / 255.0 * 10.0
                else:
                    depth_gt = depth_gt / 255.0

            if self.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            image = self.to_tensor(image)
            image = self.normalize_imagenet(image)
            if has_valid_depth:
                depth_gt = self.to_tensor(depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth, 'name': name}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def to_tensor(self, image_array):
        return torch.permute(torch.from_numpy(image_array), [2, 0, 1])

    def __len__(self):
        return len(self.filenames)
