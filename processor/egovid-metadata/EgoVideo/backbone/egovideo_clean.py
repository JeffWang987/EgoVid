import torch
import sys
from model.setup_model import *
import argparse
import pandas as pd
from decord import VideoReader, cpu, gpu
import json
from tqdm import tqdm
import imageio
from PIL import Image
from torchvision import transforms
def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10000000)
    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--video_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos")
    parser.add_argument("--save_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/egovideo_score")
    parser.add_argument("--EgoVideo_ckpt", type=str, default="../ckpt_4frames.pth")
    opts = parser.parse_args()
    return opts


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def read_video(path, eval_frames=None):
    vreader = imageio.get_reader(path,  'ffmpeg')
    try:
        vreader.get_data(eval_frames[-1])
    except:
        return None
    frames = [preprocess(Image.fromarray(vreader.get_data(i))) for i in eval_frames]
    return torch.stack(frames, 0)

if __name__ == '__main__':
    args = Options()
    csv = args.csv
    video_root = args.video_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    model,tokenizer = build_model(ckpt_path=args.EgoVideo_ckpt, num_frames = 4)
    model = model.eval().to('cuda').to(torch.float16)
    csv = pd.read_csv(csv)


    offset_num = 10000  # 1.5h 单卡
    tmp_start_idx1 = args.start_idx
    all_info = os.listdir(save_root)
    all_end_idxes = [int(idx.split('e')[-1].split('.json')[0]) for idx in all_info]
    tmp_start_idx = args.start_idx
    for end_idx in all_end_idxes:
        if end_idx > tmp_start_idx and end_idx < args.end_idx:
            tmp_start_idx = end_idx
    args.start_idx = tmp_start_idx
    json_list = []
    for index, row in tqdm(csv.iterrows()):
        if index < args.start_idx or index > args.end_idx:
            continue
        try:
            video_path = os.path.join(video_root, row.video_id)
            vision_input = read_video(video_path, eval_frames=[0, 30, 60, 90])
            vision_input = vision_input[None].permute(0,2,1,3,4).to('cuda').to(torch.float16)
            #[B,C,T,H,W] (1,3,4,224,224)
            text = row['name']
            text = tokenizer(text,max_length=20,truncation=True,padding = 'max_length',return_tensors = 'pt')
            text_input = text.input_ids.to('cuda')
            mask = text.attention_mask.to('cuda')
            image_features, text_features = model(vision_input,text_input,mask)
            json_info = {
                'idx': index,
                'video_id': row.video_id,
                'score': (image_features @ text_features.T).item(),
            }
            json_list.append(json_info)
        except Exception as e:
            print(e)
            continue

        if index % offset_num == 0 and index != args.start_idx:
            last_tmp_save_path = os.path.join(save_root, 'ego4d_egovideo_s{}e{}.json'.format(args.start_idx, index-offset_num))
            tmp_save_path = os.path.join(save_root, 'ego4d_egovideo_s{}e{}.json'.format(args.start_idx, index))
            with open(tmp_save_path, 'w') as f:
                json.dump(json_list, f)
            os.system('rm -rf {}'.format(last_tmp_save_path))
    save_path = os.path.join(save_root, 'ego4d_egovideo_s{}e{}.json'.format(args.start_idx, index))
    with open(save_path, 'w') as f:
        json.dump(json_list, f)

