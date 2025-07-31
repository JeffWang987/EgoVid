import torch
import pandas as pd
pd.options.mode.chained_assignment = None
import os
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
import json
import numpy as np
import imageio
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

def read_video(path, eval_frames=None):
    vreader = imageio.get_reader(path,  'ffmpeg')
    try:
        vreader.get_data(eval_frames[-1])
    except:
        return None
    frames = np.stack([vreader.get_data(i) for i in eval_frames], axis=0)
    return frames



def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def add_legend(img, text='hello', position=(0, 0), colour=[255, 255, 255], size=14):
    font_path = 'DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, size)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, tuple(colour), font=font)
    return np.array(pil_img)


def remove_gravity(acce, fps, gravity_cutoff_freq=0.1):
    """
    Parameters:
        acce: N x 3 numpy array of acceleration values (xyz) in m/s^2
        fps: The sampling rate of the IMU signals in Hz
        gravity_cutoff_freq: Cutoff frequency in Hz to isolate gravity component (generally low; 0.1-0.5 Hz)
    """
    # 设计低通滤波器以分离重力
    b, a = butter(N=1, Wn=gravity_cutoff_freq / (0.5 * fps), btype='low')
    gravity_component = filtfilt(b, a, acce, axis=0)
    # 移除重力分量，获得“纯”加速度信号
    acce_without_gravity = acce - gravity_component
    return acce_without_gravity

def filter_noise(acce, fps, noise_cutoff_freq=1.0):
    # 设计高通滤波器以去除剩余的高频噪声
    b, a = butter(N=1, Wn=noise_cutoff_freq / (0.5 * fps), btype='high')
    acce_filtered = filtfilt(b, a, acce, axis=0)
    return acce_filtered

if __name__ == '__main__':
    csv = '/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/dynamicrafter_csv/ego4d_moreimu_val_0.05_20.csv'
    data_info = pd.read_csv(csv)
    video_root = '/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos'
    save_vis_path = '/mnt/workspace/workgroup/jeff.wang/misc/tmp/ego4d-imu-vis'
    imu_keys = ['gyro_x', 'gyro_y', 'gyro_z', 'accl_x', 'accl_y', 'accl_z']
    os.makedirs(save_vis_path, exist_ok=True)

    # 先统计imu值域
    this_imu_info = {}
    for imu_key in imu_keys:
        this_imu_info[imu_key] = []
    for idx, sub_path in tqdm(enumerate(data_info.video_id)):
        for imu_key in imu_keys:
            this_imu_info[imu_key].extend(list(map(float, data_info[imu_key][idx][1:-1].split(', '))))
    bar_widths = {}
    for imu_key in imu_keys:
        this_mean = np.mean(this_imu_info[imu_key])
        if 'gyro' in imu_key:
            this_var = np.var(this_imu_info[imu_key]) * 3
        else:
            this_var = np.var(this_imu_info[imu_key])
        bar_width = max(np.abs(this_mean-this_var), np.abs(this_mean+this_var))
        bar_widths[imu_key] = bar_width
    bar_widths['accl_x'] = 3
    bar_widths['accl_y'] = 3
    bar_widths['accl_z'] = 3

    for idx, sub_path in tqdm(enumerate(data_info.video_id)):
        this_src_path = os.path.join(video_root, sub_path)
        this_imu_info = {}
        for imu_key in imu_keys:
            this_imu_info[imu_key] = data_info[imu_key][idx][1:-1].split(', ')
            this_imu_info[imu_key] = list(map(float, this_imu_info[imu_key]))
        # 滤除重力加速度
        accl_xyz = np.stack([this_imu_info['accl_x'], this_imu_info['accl_y'], this_imu_info['accl_z']]).transpose(1,0)
        accl_xyz = remove_gravity(accl_xyz, fps=data_info.fps[idx]).transpose(1,0)
        this_imu_info['accl_x'] = accl_xyz[0].tolist()
        this_imu_info['accl_y'] = accl_xyz[1].tolist()
        this_imu_info['accl_z'] = accl_xyz[2].tolist()
        # 滤除高频噪声
        for imu_key in imu_keys:
            this_imu_info[imu_key] = filter_noise(this_imu_info[imu_key], fps=data_info.fps[idx]).tolist()
        frames = read_video(this_src_path, list(range(len(this_imu_info[imu_key]))))
        height, width = frames.shape[1:3]
        
        vis_frames = []
        for frame_idx, frame in enumerate(frames):
            pad_img = 255 * np.ones((height//2, width, 3)).astype(np.uint8)

            bar_height = height // 18
            bar_width = width * 1//3  # 半边的长度
            # centering_offset = 40
            # width_offset = bar_width + width + 200 + centering_offset
            width_offset = width//2
            last_gauge_height = 10
            gauge_height_itv = 10
            color = (79, 171, 198)
            for imu_key in imu_keys:
                cursor = this_imu_info[imu_key][frame_idx]
                max_value = bar_widths[imu_key]
                if cursor > 0:
                    start = 0
                    end = int(min(cursor / max_value, 1) * bar_width)
                else:
                    start = int(max(cursor / max_value, -1) * bar_width)
                    end = 0

                # fill
                this_gauge_height = last_gauge_height + bar_height
                pad_img[last_gauge_height:this_gauge_height, width_offset + start: width_offset + end] = color
                # contour
                height_slice = slice(last_gauge_height, this_gauge_height)
                width_slice = slice(width_offset - bar_width, width_offset + bar_width)
                pad_img[height_slice, width_slice] = make_contour(pad_img[height_slice, width_slice], colour=[0, 0, 0])

                # Middle gauge
                pad_img[last_gauge_height - 2:this_gauge_height + 2, width_offset:width_offset + 1] = (0, 0, 0)
                # Add labels
                pad_img = add_legend(pad_img, f'{imu_key}:', (width_offset - bar_width*1.4, last_gauge_height), (0,0,0), size=24)
                pad_img = add_legend(pad_img, f'{cursor:.2f}', (width_offset + bar_width + 10, last_gauge_height), (0,0,0), size=24)

                last_gauge_height = this_gauge_height + gauge_height_itv
            frame = np.concatenate([frame, pad_img], axis=0)
            vis_frames.append(frame)
        save_path = os.path.join(save_vis_path, '{}.mp4'.format(idx))    
        imageio.mimwrite(save_path, vis_frames)
            
        
