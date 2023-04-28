import os
from datetime import datetime

from infer import InferenceHelper

if __name__ == '__main__':
    dataset = 'mi'
    ckpt = 'checkpoints_Apr-25_19-51-c7bf79/WeightedUnet_Apr-25_19-51-node_bs12-tep30-lr0.0001-wd0.1-9d3cae_best.pt'
    save_path = os.path.join('./results', dataset, datetime.now().strftime('%h-%d_%H-%M-%S'))

    from models.unet import UNet

    infer_helper = InferenceHelper(UNet, ckpt, dataset, device='cuda:0')
    result_dict_list = infer_helper.predict_dataloader(
        batch_size=8, num_threads=8)

    infer_helper.post_process(save_path, result_dict_list, save_imgs=False)
