import os

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import cv2
from tqdm import tqdm
from dataloader import DepthDataLoader
import model_io
from operator import itemgetter
from utils import RunningAverageDict, compute_errors, norm_to_orig
from models.unet import UNet
from torchvision import transforms


class InferenceHelper:
    def __init__(self, model_class=UNet, ckpt=None, dataset='mi', device='cuda:0'):
        self.dataset = dataset
        self.ckpt = ckpt
        self.device = device
        self.do_kb_crop = False
        self.do_garg_crop = False

        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 255  # used to save in 16 bit
            self.data_path = '../datasets/nyu/official_splits/test/'
            self.gt_path = '../datasets/nyu/official_splits/test/'
            self.filenames_file = 'train_test_inputs/nyudepthv2_test_files_with_gt.txt'
            if ckpt == None:
                self.ckpt = "checkpoints_mi/WeightedUnet_30.pt"
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 255
            self.do_kb_crop = True
            self.do_garg_crop = True
            self.data_path = '../datasets/kitti/raw'
            self.gt_path = '../datasets/kitti/val'
            self.filenames_file = 'train_test_inputs/kitti_eigen_test_files_with_gt.txt'
            if ckpt == None:
                self.ckpt = "checkpoints_kitti/WeightedUnet_30.pt"
        elif dataset == 'mi':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 255
            self.data_path = None
            self.gt_path = None
            self.filenames_file = 'train_test_inputs/mi_test.txt'
            if ckpt == None:
                self.ckpt = "checkpoints_mi/WeightedUnet_30.pt"
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model = model_class(cout=1)
        model.eval()
        model = model_io.load_weights_from_checkpoint(ckpt, model)
        self.model = model.to(device)

    @torch.no_grad()
    def predict_dataloader(self, batch_size=8, num_threads=8):

        result_dict_list = []

        ##### Dataloader ######
        loader = DepthDataLoader(False, batch_size, num_threads, 'test', self.dataset, None, None, self.filenames_file, self.do_kb_crop, False, self.data_path, self.gt_path).data
        for i, batch in tqdm(enumerate(loader), desc=f"Loop: Test", total=len(loader)):

            ##### Get data from batch and infer ######
            names = batch['name']
            image = batch['image'].to(self.device)
            depth_gt = batch['depth'].to(self.device)

            ##### Predict ######
            depth_pred = self.predict(image)

            ##### Get data from batch and infer ######
            for i, _ in enumerate(names):
                result_dict_list.append({'name': names[i],
                                         'im': image[i].cpu().numpy(),
                                         'dp_gt': depth_gt[i][0].cpu().numpy(),
                                         'dp_pred': depth_pred[i][0].cpu().numpy()})

        return result_dict_list

    def post_process(self, save_path, result_dict_list, save_imgs=True):
        ##### Logger ######
        os.makedirs(save_path, exist_ok=True)
        log = open(os.path.join(save_path, "0_result.log"), mode="w")
        log.write(f"Current checkpoint is: {self.ckpt} \n")

        m_dp = RunningAverageDict()

        for dict in tqdm(result_dict_list, desc=f"Saving Images"):
            name, im, dp_gt, dp = \
                itemgetter(*['name', 'im', 'dp_gt', 'dp_pred'])(dict)

            im = norm_to_orig(im)
            valid_mask = np.logical_and(dp_gt > self.min_depth, dp_gt < self.max_depth)

            errors = compute_errors(dp_gt[valid_mask], dp[valid_mask])
            m_dp.update(errors)

            if save_imgs:
                self.save_image(im, os.path.join(save_path, f'{name}_im.png'))
                self.save_image(dp_gt / self.max_depth, os.path.join(save_path, f'{name}_dp_gt.png'), self.saving_factor)
                self.save_image(dp / self.max_depth, os.path.join(save_path, f'{name}_dp.png'), self.saving_factor)
        log.write(f'Depth:     {m_dp.get_value()} \n')

    @torch.no_grad()
    def predict(self, image):
        pred = self.model(image) * self.max_depth
        pred = torch.clip(pred, self.min_depth, self.max_depth)

        # Flip
        image_lr = torch.flip(image, [3])
        pred_lr = self.model(image_lr) * self.max_depth
        pred_lr = torch.flip(pred_lr, [3])
        pred_lr = torch.clip(pred_lr, self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)

        final[torch.isinf(final)] = self.max_depth
        final[torch.isnan(final)] = self.min_depth

        return final

    def save_image(self, im, save_path, saving_factor=None):
        c = im.shape[0]
        if c == 3:
            im = np.uint8(np.transpose(im*255.0, [1, 2, 0]))
        elif c == 1:
            if saving_factor is None or saving_factor == 255:
                im = np.uint8(im[0] * 255.0)
            else:
                im = np.uint16(im[0] * saving_factor)
        elif len(im.shape) == 2:
            im = np.uint8(im * 255.0)
        else:
            assert False, "Image channel error!"
        im = Image.fromarray(im)
        im.save(save_path)

    def cv2_resize(self, im, size, mode=cv2.INTER_AREA):
        c = im.shape[0]
        im = cv2.resize(np.transpose(im, [2, 1, 0]), size, interpolation=mode)
        if c == 1:
            im = np.transpose(im, [1, 0])[np.newaxis, :, :]
        else:
            im = np.transpose(im, [2, 1, 0])
        return im


if __name__ == '__main__':
    pass
