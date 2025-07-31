# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Run the whole pipeline of trajectory-based video sfm from images
images -> optical flow -> point trajectories -> motion seg -> global mapper
"""
import os
import argparse
import shutil
import pdb
import imageio
import glob
from tqdm import tqdm
import pandas as pd
import struct

def read_points3d_bin(file_path):
    points_3d = []
    with open(file_path, "rb") as file:
        # 读取记录数量
        num_points = struct.unpack('Q', file.read(8))[0]  # 假设使用Q来表示无符号长整形
    return num_points

def connect_point_trajectory(args, image_dir, output_dir, skip_exists=False, keep_intermediate=False, raft1=None, raft2=None):
    # set directories in the workspace
    flow_dir = os.path.join(output_dir, "optical_flows")
    traj_dir = os.path.join(output_dir, "trajectories")

    # optical flow (RAFT)
    from third_party.RAFT import compute_raft_custom_folder, compute_raft_custom_folder_stride2
    print("[ParticleSFM] Running pairwise optical flow inference......")
    raft1 = compute_raft_custom_folder(image_dir, flow_dir, skip_exists=skip_exists, model=raft1)
    if not args.skip_path_consistency:
        print("[ParticleSfM] Running pairwise optical flow inference (stride 2)......")
        raft2 = compute_raft_custom_folder_stride2(image_dir, flow_dir, skip_exists=skip_exists, model=raft2)

    # point trajectory (saved in workspace_dir / point_trajectories)
    from point_trajectory import main_connect_point_trajectories
    print("[ParticleSfM] Connecting (optimization {0}) point trajectories from optical flows.......".format("disabled" if args.skip_path_consistency else "enabled"))
    main_connect_point_trajectories(flow_dir, traj_dir, sample_ratio=args.sample_ratio, flow_check_thres=args.flow_check_thres, skip_path_consistency=args.skip_path_consistency, skip_exists=skip_exists)

    if not keep_intermediate:
        # remove optical flows
        shutil.rmtree(os.path.join(output_dir, "optical_flows"))
    return traj_dir, raft1, raft2

def motion_segmentation(args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False, motion_model=None):
    # set directories in the workspace
    depth_dir = os.path.join(output_dir, "midas_depth")
    labeled_traj_dir = traj_dir + "_labeled"

    # monocular depth (MiDaS)
    print("[ParticleSfM] Running per-frame monocular depth estimation........")
    from third_party.MiDaS import run_midas
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    run_midas(image_dir, depth_dir, skip_exists=skip_exists)

    # point trajectory based motion segmentation
    print("[ParticleSfM] Running point trajectory based motion segmentation........")
    from motion_seg import main_motion_segmentation
    motion_model = main_motion_segmentation(image_dir, depth_dir, traj_dir, labeled_traj_dir, window_size=args.window_size, 
                traj_max_num=args.traj_max_num, skip_exists=skip_exists, model=motion_model)
    if os.path.isfile(os.path.join(output_dir, "motion_seg.mp4")):
        os.remove(os.path.join(output_dir, "motion_seg.mp4"))
    shutil.move(os.path.join(labeled_traj_dir, "motion_seg.mp4"), output_dir)

    if not keep_intermediate:
        # remove original point trajectories
        shutil.rmtree(depth_dir)
        shutil.rmtree(traj_dir)
    return labeled_traj_dir, motion_model

def sfm_reconstruction(args, image_dir, output_dir, traj_dir, skip_exists=False, keep_intermediate=False):
    # set directories in the workspace
    sfm_dir = os.path.join(output_dir, "sfm")

    # sfm reconstruction
    from sfm import main_global_sfm, main_incremental_sfm, write_depth_pose_from_colmap_format
    if not args.incremental_sfm:
        print("[ParticleSfM] Running global structure-from-motion........")
        main_global_sfm(sfm_dir, image_dir, traj_dir, remove_dynamic=(not args.assume_static), skip_exists=skip_exists)
    else:
        print("[ParticleSfM] Running incremental structure-from-motion with COLMAP........")
        main_incremental_sfm(sfm_dir, image_dir, traj_dir, remove_dynamic=(not args.assume_static), skip_exists=skip_exists)

    # # write depth and pose files from COLMAP format
    write_depth_pose_from_colmap_format(sfm_dir, os.path.join(output_dir, "colmap_outputs_co" \
    "nverted"))

    if not keep_intermediate:
        # remove labeled point trajectories
        shutil.rmtree(traj_dir)

def particlesfm(args, image_dir, output_dir, skip_exists=False, keep_intermediate=False, raft1=None, raft2=None, motion_model=None):
    """
    Inputs:
    - img_dir: str - The folder containing input images
    - output_dir: str - The workspace directory
    """
    if not os.path.exists(image_dir):
        raise ValueError("Error! The input image directory {0} is not found.".format(image_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # connect point trajectory
    traj_dir, raft1, raft2 = connect_point_trajectory(args, image_dir, output_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate, raft1=raft1, raft2=raft2)

    # motion segmentation
    if not args.assume_static:
        traj_dir, motion_model = motion_segmentation(args, image_dir, output_dir, traj_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate,
                motion_model=motion_model)

    # sfm reconstruction
    if not args.skip_sfm:
        sfm_reconstruction(args, image_dir, output_dir, traj_dir, skip_exists=skip_exists, keep_intermediate=keep_intermediate)

    return raft1, raft2, motion_model
def parse_args():
    parser = argparse.ArgumentParser("Dense point trajectory based colmap reconstruction for videos")
    # point trajectory
    parser.add_argument("--flow_check_thres", type=float, default=1.0, help='the forward-backward flow consistency check threshold')
    parser.add_argument("--sample_ratio", type=int, default=2, help='the sampling ratio for point trajectories')
    parser.add_argument("--traj_min_len", type=int, default=3, help='the minimum length for point trajectories')
    # motion segmentation
    parser.add_argument("--window_size", type=int, default=10, help='the window size for trajectory motion segmentation')
    parser.add_argument("--traj_max_num", type=int, default=100000, help='the maximum number of trajs inside a window')
    # sfm
    parser.add_argument("--incremental_sfm", action='store_true', help='whether to use incremental sfm or not')
    # pipeline control
    parser.add_argument("--skip_path_consistency", action='store_true', help='whether to skip the path consistency optimization or not')
    parser.add_argument("--assume_static", action='store_true', help='whether to skip the motion segmentation or not')
    parser.add_argument("--skip_sfm", action='store_true', help='whether to skip structure-from-motion or not')
    parser.add_argument("--skip_exists", action='store_true', help='whether to skip exists')
    parser.add_argument("--keep_intermediate", action='store_true', help='whether to keep intermediate files such as flows, monocular depths, etc.')

    # input by sequence directory
    # python run_particlesfm.py --image_dir ${PATH_TO_SEQ_FOLDER} --output_dir ${OUTPUT_WORKSPACE}
    parser.add_argument("-i", type=str, default="none", help="path to the video")
    parser.add_argument("-o", "--output_dir", type=str, default="none", help="workspace for output")

    # input by workspace
    # python run_particlesfm.py --workspace_dir ${WORKSPACE_DIR}
    parser.add_argument("--workspace_dir", type=str, default="none", help="input workspace")
    parser.add_argument("--image_folder", type=str, default="images", help="image folder") # also used in the folder option

    # input by folder containing multiple workspaces
    # python run_particlesfm.py --root_dir ${ROOT_DIR}
    # multiple sequences should be with the structure below:
    # - ROOT_DIR
    #    - XXX (sequence 1)
    #        - images
    #            - xxxxxx.png
    #    - XXX (sequence 2)
    parser.add_argument("--root_dir", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos", help='path to to the folder containing workspaces')
    parser.add_argument("--csv_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv")
    parser.add_argument("--tmp_save_dir", type=str, default="/mnt/workspace/workgroup/jeff.wang/code/particle-sfm")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/workgroup/jeff.wang/code/particle-sfm/outputs/egovid_alljson")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100000)
    args = parser.parse_args()
    return args

def read_video(path):
    vreader = imageio.get_reader(path,  'ffmpeg')
    frames = []
    for frame in vreader:
        frames.append(frame)
    return frames



if __name__ == "__main__":
    args = parse_args()
    video_root = args.root_dir
    csv_root = args.csv_root
    csv_info = pd.read_csv(csv_root)
    tmp_save_dir = os.pat.join(args.tmp_save_dir, 'tmp_alljson_start{}_end{}'.format(args.start_idx, args.end_idx))
    output_dir = args.output_dir
    raft1, raft2, motion_model  = None, None, None
    for voideo_idx, video_path in tqdm(enumerate(csv_info.video_id)):
        if voideo_idx < args.start_idx or args.start_idx >= args.end_idx:
            continue
        video_path = os.path.join(video_root, video_path)
        try:
            if voideo_idx >= 100000:
                out_path = os.path.join(output_dir, '{:06d}'.format(voideo_idx))
            else:
                out_path = os.path.join(output_dir, '{:05d}'.format(voideo_idx))

            if os.path.exists(out_path):
                continue
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(tmp_save_dir, exist_ok=True)
            frames = read_video(video_path)
            assert len(frames)>=120
            frames = frames[0:min(120, len(frames)):5]  # 6FPS
            for idx, frame in enumerate(frames):
                imageio.imwrite(os.path.join(tmp_save_dir, '{:05d}.png'.format(idx)), frame)
            raft1, raft2, motion_model = particlesfm(args, tmp_save_dir, out_path, skip_exists=args.skip_exists, keep_intermediate=args.keep_intermediate,
                        raft1=raft1, raft2=raft2, motion_model=motion_model)
            os.system('rm -r ' + tmp_save_dir)
            this_colmap_path = os.path.join(out_path, 'sfm/model')
            this_colmap_pts_path = os.path.join(this_colmap_path, 'points3D.bin')
            num_pts = read_points3d_bin(this_colmap_pts_path)
            if num_pts < 20000:
                os.system('rm -r ' + out_path)
                continue
            else:
                os.system('rm -r ' + os.path.join(out_path, 'colmap_outputs_converted', 'depths'))
                os.system('rm -r ' + os.path.join(out_path, 'sfm'))

            with open(os.path.join(out_path, 'num_pts.txt'), 'w'):
                print(num_pts, file=open(os.path.join(out_path, 'num_pts.txt'), 'w'))
        except Exception as e:
            print(e)
            continue
