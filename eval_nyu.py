import os
from datetime import datetime

from infer import InferenceHelper

if __name__ == '__main__':
    dataset = 'nyu'
    ckpt = 'checkpoints_ml4_best/WeightedUnet_30.pt'
    save_path = os.path.join('./results', dataset, datetime.now().strftime('%h-%d_%H-%M-%S'))

    from models.unet import UNet

    infer_helper = InferenceHelper(UNet, ckpt, dataset, device='cuda:1')
    result_dict_list = infer_helper.predict_dataloader(batch_size=4, num_threads=8)

    infer_helper.post_process(save_path, result_dict_list, save_imgs=False)
