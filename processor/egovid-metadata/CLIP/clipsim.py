import torch
torch.cuda.current_device()
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
import open_clip
from PIL import Image

def get_text_img_similarity(text, img, clip_model, img_preprocess, text_tokenizer, device='cpu'):
    img = img_preprocess(img).unsqueeze(0)
    text = text_tokenizer(text)
    if device.find('cuda')>=0:
        img = img.to(device)
        text = text.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = 100.0 * image_features @ text_features.T
    if device.find('cuda')>=0:
        similarity = similarity.to('cpu')
    return similarity.item()


def get_img_img_similarity(img1, img2, clip_model, img_preprocess, device='cpu'):
    img1 = img_preprocess(img1).unsqueeze(0)
    img2 = img_preprocess(img2).unsqueeze(0)
    if device.find('cuda')>=0:
        img1 = img1.to(device)
        img2 = img2.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features1 = clip_model.encode_image(img1)
        image_features2 = clip_model.encode_image(img2)
        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)
        similarity = 100.0 * image_features1 @ image_features2.T
    if device.find('cuda')>=0:
        similarity = similarity.to('cpu')
    return similarity.item()



def read_video(path, eval_frames=None):
    vreader = imageio.get_reader(path,  'ffmpeg')
    try:
        vreader.get_data(eval_frames[-1])
    except:
        return None
    frames = np.stack([vreader.get_data(i) for i in eval_frames], axis=0)
    return frames

def get_clip_model(clip_model_name="ViT-L-14", pretrained='openai', device='cpu'):
    # see more open_clip.list_pretrained()
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, cache_dir='/mnt/workspace/workgroup/jeff.wang/code/3rdparty/clip-models')
    model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    return model, preprocess, tokenizer


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10000000)
    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--ego4d_video_src_path", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos_all_narration')
    parser.add_argument("--save_root", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/clipsim')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = Options()
    csv = opts.csv
    video_root = opts.ego4d_video_src_path
    root_dir = opts.root_dir
    os.makedirs(root_dir, exist_ok=True)

    offsaet_num = 5000  # 1h 单卡
    tmp_start_idx1 = opts.start_idx
    all_csv = os.listdir(root_dir)
    all_end_idxes = [int(idx.split('0e')[1].split('.csv')[0]) for idx in all_csv]
    tmp_start_idx = opts.start_idx
    for end_idx in all_end_idxes:
        if end_idx > tmp_start_idx and end_idx < opts.end_idx:
            tmp_start_idx = end_idx
    opts.start_idx = tmp_start_idx

    save_path = os.path.join(root_dir, 'ego4d_clips_s{}e{}.csv'.format(opts.start_idx, opts.end_idx))
    data_info = pd.read_csv(csv)
    simi_model, img_preprocess, text_tokenizer = get_clip_model(device='cuda')

    data_info['ti_sim'] = None
    data_info['ii_sim'] = None

    for index, row in tqdm(data_info.iterrows()):
        if index < opts.start_idx or index > opts.end_idx:
            continue
        video_path = os.path.join(video_root, row.video_id)
        text = data_info.name[index]
        try:
            frames = read_video(video_path, [0, 39, 79, 119])
            ti_sim = []
            ii_sim = []
            for frame in frames:
                ti_sim.append(get_text_img_similarity(text, Image.fromarray(frame), simi_model, img_preprocess, text_tokenizer, device='cuda'))
            for frame in frames[1:]:
                ii_sim.append(get_img_img_similarity(Image.fromarray(frames[0]), Image.fromarray(frame), simi_model, img_preprocess, device='cuda'))
            
            data_info.loc[index, 'ti_sim'] = str(ti_sim)
            data_info.loc[index, 'ii_sim'] = str(ii_sim)
        except Exception as e:
            print(e)
            continue
        if index % offsaet_num == 0 and index != opts.start_idx:
            last_tmp_save_path = os.path.join(root_dir, 'ego4d_clips_s{}e{}.csv'.format(opts.start_idx, index-offsaet_num))
            tmp_save_path = os.path.join(root_dir, 'ego4d_clips_s{}e{}.csv'.format(opts.start_idx, index))
            data_info.loc[opts.start_idx:index].to_csv(tmp_save_path, index=False)
            os.system('rm -rf {}'.format(last_tmp_save_path))

    data_info.loc[opts.start_idx:opts.end_idx].to_csv(save_path, index=False)
