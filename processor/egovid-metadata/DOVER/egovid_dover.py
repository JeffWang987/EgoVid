import torch

import argparse
import pickle as pkl

import decord
import numpy as np
import yaml

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER
from tqdm import tqdm
import json
import os
import pandas as pd

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr


def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--opt", type=str, default="./dover.yml", help="the option file"
    )

    ## can be your own
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="./demo/17734.mp4",
        help="the input video path",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    parser.add_argument(
        "-f", "--fusion", action="store_true",
    )

    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--ego4d_video_src_path", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos_all_narration')
    parser.add_argument("--save_root", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/caption')
    parser.add_argument("--colmap_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/code/particle-sfm/outputs/egovid_alljson")
    parser.add_argument("--video_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos")
    parser.add_argument("--save_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/dover_score_forimu")

    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100000000)

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load DOVER
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=args.device)['state_dict']
    )

    dopt = opt["data"]["val-l1080p"]["args"]

    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    csv = args.csv
    video_root = args.video_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
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
        video_path = os.path.join(video_root, row.video_id)
        try:
            ### View Decomposition
            views, _ = spatial_temporal_view_decomposition(
                video_path, dopt["sample_types"], temporal_samplers
            )

            for k, v in views.items():
                num_clips = dopt["sample_types"][k].get("num_clips", 1)
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(args.device)
                )

            results = [r.mean().item() for r in evaluator(views)]
            json_info = {
                'idx': index,
                'video_id': row.video_id,
                'score': fuse_results(results)
            }
            # print(json_info['score'])
            json_list.append(json_info)
        except Exception as e:
            print(e)
            continue
        if index % offset_num == 0 and index != args.start_idx:
            last_tmp_save_path = os.path.join(save_root, 'ego4d_doverscore_s{}e{}.json'.format(args.start_idx, index-offset_num))
            tmp_save_path = os.path.join(save_root, 'ego4d_doverscore_s{}e{}.json'.format(args.start_idx, index))
            with open(tmp_save_path, 'w') as f:
                json.dump(json_list, f)
            os.system('rm -rf {}'.format(last_tmp_save_path))
    save_path = os.path.join(save_root, 'ego4d_doverscore_s{}e{}.json'.format(args.start_idx, index))
    with open(save_path, 'w') as f:
        json.dump(json_list, f)
