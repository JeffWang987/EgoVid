import os
import sys
sys.path.append("/mnt/workspace/workgroup/jeff.wang/code/LLaVA-NeXT")
from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import pandas as pd
import json
import math
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoConfig
import cv2
import base64
import openai
from PIL import Image
import numpy as np
import argparse
import torch
def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10000000)
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/workgroup/jeff.wang/Public-Home/models/huggingface/models--lmms-lab--LLaVA-NeXT-Video-32B-Qwen")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default='qwen_1_5')
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default='Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.') 
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--mm_pooling_position", type=str, default="after")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--ego4d_video_src_path", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos_all_narration')
    parser.add_argument("--save_root", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/caption')
    args = parser.parse_args()
    return args

def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames

def load_model(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
    overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
    overwrite_config["mm_pooling_position"] = args.mm_pooling_position
    overwrite_config["mm_newline_position"] = args.mm_newline_position
    cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)

    question = args.prompt

    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    
    return tokenizer, model, image_processor, input_ids, attention_masks, stopping_criteria, stop_str

def get_response(video_path, tokenizer, model, image_processor, input_ids, attention_masks, stopping_criteria, stop_str):
    if os.path.exists(video_path):
        video = load_video(video_path, args)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        video = [video]
    else:
        return None
    with torch.inference_mode():
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True) #, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs



if __name__ == '__main__':
    args = Options()
    csv = args.csv
    video_root = args.ego4d_video_src_path
    save_root = args.save_root
    csv = pd.read_csv(csv)
    tokenizer, model, image_processor, input_ids, attention_masks, stopping_criteria, stop_str = load_model(args)
    save_path = os.path.join(save_root, 'ego4d_captions_s{}e{}.json'.format(args.start_idx, args.end_idx))
    offset_num = 1000  # 3h 单卡
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
            response = get_response(video_path, tokenizer, model, image_processor, input_ids, attention_masks, stopping_criteria, stop_str)
            if response is None:
                continue
            json_info = {
                'idx': index,
                'video_id': row.video_id,
                'response': response
            }
            json_list.append(json_info)
        except Exception as e:
            print(e)
            continue
        if index % offset_num == 0 and index != args.start_idx:
            last_tmp_save_path = os.path.join(save_root, 'ego4d_captions_s{}e{}.json'.format(args.start_idx, index-offset_num))
            tmp_save_path = os.path.join(save_root, 'ego4d_captions_s{}e{}.json'.format(args.start_idx, index))
            with open(tmp_save_path, 'w') as f:
                json.dump(json_list, f)
            os.system('rm -rf {}'.format(last_tmp_save_path))

