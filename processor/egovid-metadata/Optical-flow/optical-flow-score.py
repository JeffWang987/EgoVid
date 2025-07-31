import pandas as pd
import os
from tqdm import tqdm
import copy
import argparse
import torch
import torch.nn as nn
from PIL import Image
# import cv2
import numpy as np
from decord import VideoReader
import sys
from raft import RAFT
from utils.utils import InputPadder

def load_optical_flow_model(model_path, device='cpu'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--ego4d_video_src_path", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos')
    parser.add_argument("--save_root", type=str, default='//mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/opt-flow-forimu')
    args = parser.parse_args()
    args.model = model_path
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(device)
    model.eval()
    return model, args


def get_optical_flow(model, image1, image2, device='cpu'):
    def cart_to_polar_manual(x, y):
        magnitude = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        return magnitude, angle
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    flow_up = flow_up[0,...].permute(1,2,0).cpu().detach().numpy()
    # mag, ang = cv2.cartToPolar(flow_up[...,0], flow_up[...,1])
    mag, ang = cart_to_polar_manual(flow_up[...,0], flow_up[...,1])
    ref_size = np.mean(mag.shape)
    mag_relative = mag / ref_size * 512 # 相当于在512尺寸的图上移动的像素距离
    mag_ratio_0_4 = ((mag_relative>0) & (mag_relative<=5)).sum() / (mag_relative.shape[0]*mag_relative.shape[1])
    mag_ratio_4_8 = ((mag_relative>4) & (mag_relative<=8)).sum() / (mag_relative.shape[0]*mag_relative.shape[1])
    mag_ratio_8_12 = ((mag_relative>8) & (mag_relative<=12)).sum() / (mag_relative.shape[0]*mag_relative.shape[1])
    mag_ratio_12_16 = ((mag_relative>12) & (mag_relative<=16)).sum() / (mag_relative.shape[0]*mag_relative.shape[1])
    mag_ratio_16_ = (mag_relative>16).sum() / (mag_relative.shape[0]*mag_relative.shape[1])
    return mag_relative.mean(), mag_ratio_0_4, mag_ratio_4_8, mag_ratio_8_12, mag_ratio_12_16, mag_ratio_16_


if __name__ == '__main__':

    model, opts = load_optical_flow_model('/mnt/workspace/workgroup/jeff.wang/code/3rdparty/RAFT/ckpts/raft-things.pth', 'cuda')
    root_dir = opts.save_root
    data_root = opts.ego4d_video_src_path
    ann_root = opts.csv
    
    os.makedirs(root_dir, exist_ok=True)

    ann_info = pd.read_csv(ann_root)

    ann_info['flow_mean'] = None
    ann_info['flow_0_4'] = None # 相当于0.3s的视频，512尺寸的图像，运动了1/40
    ann_info['flow_4_8'] = None
    ann_info['flow_8_12'] = None
    ann_info['flow_12_16'] = None
    ann_info['flow_16_'] = None
    STEP = 8

    # 在剩下的数据里计算flow
    offsaet_num = 3000  # 1h 单卡
    all_flow_csv = os.listdir(root_dir)
    all_end_idxes = [int(idx.split('e')[-1].split('.csv')[0]) for idx in all_flow_csv]
    tmp_start_idx = opts.start_idx
    for end_idx in all_end_idxes:
        if end_idx > tmp_start_idx and end_idx < opts.end_idx:
            tmp_start_idx = end_idx
    opts.start_idx = tmp_start_idx
    
    save_path = os.path.join(root_dir, 'ego4d_flows_s{}e{}.csv'.format(opts.start_idx, opts.end_idx))
    
    for index, row in tqdm(ann_info.iterrows()):
        if index < opts.start_idx or index > opts.end_idx:
            continue
        if row['flow_mean'] and row['flow_0_4'] and row['flow_4_8'] and row['flow_8_12'] and row['flow_12_16'] and row['flow_16_']:
            continue
        video_path = os.path.join(data_root, row.video_id)
        try:
            vreader = VideoReader(video_path)
            fps = vreader.get_avg_fps()
            v_length = len(vreader)
            frames = vreader.get_batch([i for i in range(0, v_length, STEP)]).asnumpy().astype(np.uint8)
        except:
            continue

        mag_all = []
        mag_ratio_0_4s = []
        mag_ratio_4_8s = []
        mag_ratio_8_12s = []
        mag_ratio_12_16s = []
        mag_ratio_16_s = []
        for i in range(len(frames)-1):
            mag_mean, mag_ratio_0_4, mag_ratio_4_8, mag_ratio_8_12, mag_ratio_12_16, mag_ratio_16_ = \
                get_optical_flow(model, frames[i,...], frames[i+1,...], device='cuda')
            mag_all.append(mag_mean)
            mag_ratio_0_4s.append(mag_ratio_0_4)
            mag_ratio_4_8s.append(mag_ratio_4_8)
            mag_ratio_8_12s.append(mag_ratio_8_12)
            mag_ratio_12_16s.append(mag_ratio_12_16)
            mag_ratio_16_s.append(mag_ratio_16_)
        
        ann_info.loc[index, 'flow_mean'] = np.mean(mag_all)
        ann_info.loc[index, 'flow_0_4'] = np.mean(mag_ratio_0_4s)
        ann_info.loc[index, 'flow_4_8'] = np.mean(mag_ratio_4_8s)
        ann_info.loc[index, 'flow_8_12'] = np.mean(mag_ratio_8_12s)
        ann_info.loc[index, 'flow_12_16'] = np.mean(mag_ratio_12_16s)
        ann_info.loc[index, 'flow_16_'] = np.mean(mag_ratio_16_s)

        if index % offsaet_num == 0 and index != opts.start_idx:
            last_tmp_save_path = os.path.join(root_dir, 'ego4d_flows_s{}e{}.csv'.format(opts.start_idx, index-offsaet_num))
            tmp_save_path = os.path.join(root_dir, 'ego4d_flows_s{}e{}.csv'.format(opts.start_idx, index))
            ann_info.loc[opts.start_idx:index].to_csv(tmp_save_path, index=False)
            os.system('rm -rf {}'.format(last_tmp_save_path))

    ann_info.loc[opts.start_idx:opts.end_idx].to_csv(save_path, index=False)


        
