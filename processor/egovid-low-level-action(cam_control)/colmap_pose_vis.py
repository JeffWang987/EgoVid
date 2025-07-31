import os
from os import path as osp
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/mnt/workspace/workgroup/jeff.wang/code/InteractWorld/data/processor/ego4d-imu')
from colmap_traj_vis import CameraPoseVisualizer, get_c2w
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
import io
from PIL import Image

def read_video(path):
    vreader = imageio.get_reader(path,  'ffmpeg')
    frames = []
    for frame in vreader:
        frames.append(frame)
    return frames

def ext_interpolation(paths, ratio=5, wirte_txt=False, write_path=None):
    exts = []
    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
        ext = []
        for line in lines:
            ext.append(list(map(float, line.split(' '))))
        exts.append(np.array(ext))
    exts = np.stack(exts)

    # 分离旋转矩阵和平移向量
    rotations = exts[:, :, :3]
    translations = exts[:, :, 3]

    # 将旋转矩阵转换为四元数，以方便插值
    quaternions = R.from_matrix(rotations).as_quat()

    # 创建时间戳
    timestamps_original = np.linspace(0, 1, len(exts))  # 原始24帧的时间戳
    timestamps_interpolated = np.linspace(0, 1, len(exts)*ratio)  # 插值成120帧的时间戳

    # 创建线性插值函数
    translation_interp_func = interp1d(timestamps_original, translations, axis=0, kind='linear')

    # 对平移向量进行线性插值
    interpolated_translations = translation_interp_func(timestamps_interpolated)

    # 创建Slerp插值对象
    slerp = Slerp(timestamps_original, R.from_quat(quaternions))
    # R.from_quat(quaternions).slerp(timestamps_interpolated)

    # 对四元数进行Slerp插值
    interpolated_quaternions = slerp(timestamps_interpolated)

    # 将四元数转换回旋转矩阵
    interpolated_rotations = interpolated_quaternions.as_matrix()

    # 重建外参矩阵
    interpolated_camera_extrinsics = np.empty((len(exts)*ratio, 3, 4))
    interpolated_camera_extrinsics[:, :, :3] = interpolated_rotations
    interpolated_camera_extrinsics[:, :, 3]  = interpolated_translations

    if wirte_txt:
        assert write_path is not None
        with open(write_path, 'w') as f:
            for i in range(len(interpolated_camera_extrinsics)):
                f.write(' '.join(map(str, interpolated_camera_extrinsics[i].flatten()))+'\n')

    return interpolated_camera_extrinsics
        
def transform_extrinsics(colmap_extrinsics):
    # 以第一帧为参考系，获取其逆变换矩阵
    if colmap_extrinsics.shape[-1] != colmap_extrinsics.shape[-2]:
        last_row = np.zeros((colmap_extrinsics.shape[0], 1, 4))
        last_row[..., -1] = 1.0
        colmap_extrinsics = np.concatenate((colmap_extrinsics, last_row), axis=1)

    first_frame_inv = np.linalg.inv(colmap_extrinsics[0])
    updated_extrinsics = np.array([first_frame_inv @ pose for pose in colmap_extrinsics])

    # frames_inv = np.linalg.inv(colmap_extrinsics[1:])  # ci2w
    # updated_extrinsics = np.array([colmap_extrinsics[0] @ pose for pose in frames_inv])

    return updated_extrinsics


if __name__ == '__main__':
    version = 'train'
    if version == 'train':
        csv_root = '/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/dynamicrafter_csv/ego4d_moreimu_train_0.95_20.csv'
    elif version =='val':
        csv_root = '/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/dynamicrafter_csv/ego4d_moreimu_val_0.05_20.csv'

    colmap_root = '/mnt/workspace/workgroup/jeff.wang/code/particle-sfm/outputs/ego4d_alljson_{}'.format(version)
    video_root = '/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos'
    vis_root = '/mnt/workspace/workgroup/jeff.wang/misc/tmp/colmaptmp'
    os.makedirs(vis_root, exist_ok=True)
    csv_info = pd.read_csv(csv_root)

    for voideo_idx, video_path in tqdm(enumerate(csv_info.video_id)):
        video_path = osp.join(video_root, video_path)
        this_colmap_path = osp.join(colmap_root, '{:05d}'.format(voideo_idx))
        pose_root = osp.join(this_colmap_path, 'colmap_outputs_converted', 'poses')
        intr_root = osp.join(this_colmap_path, 'colmap_outputs_converted', 'intrinsics')
        if not os.path.exists(pose_root):
            continue
        try:
            frames = read_video(video_path)
            # TODO
            """
            3. imu融合
            """
            itv = 2
            pose_paths =[osp.join(pose_root, pose_sub_path) for pose_sub_path in os.listdir(pose_root) if not pose_sub_path.startswith('interpolate')]
            w2cs = ext_interpolation(pose_paths, ratio=5, wirte_txt=True, write_path=osp.join(pose_root, 'interpolate_poses.txt'))
            w2cs = w2cs[::itv]
            frames = frames[::itv]
            with open(osp.join(intr_root, '00000.txt'), 'r') as f_int:
                this_intri = f_int.readlines()
            fxs = [float(this_intri[0].split(' ')[0]) / float(this_intri[0].split(' ')[-1]) / 2] * len(w2cs)
            # transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
            # last_row = np.zeros((1, 4))
            # last_row[0, -1] = 1.0
            # w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
            # c2ws = get_c2w(w2cs, transform_matrix, relative_c2w=False)
            # 相当于把第一帧作为世界坐标系起点
            c2ws = transform_extrinsics(w2cs)  # 这里有错，输入的名称应该是c2ws，但是这里写成了w2cs，本质内容是对的，只是名字错了
            boundary = 5
            visualizer = CameraPoseVisualizer([-boundary, boundary], [-boundary, boundary], [-boundary, boundary])
            pose_video = []
            for frame_idx, c2w in enumerate(c2ws):
                visualizer.extrinsic2pyramid(c2w, frame_idx / len(c2ws), hw_ratio=9/16, base_xval=1.0,
                                            zval=(fxs[frame_idx]))
                # visualizer.colorbar(len(c2ws))
                # visualizer.savefig(osp.join(vis_root, '{:06d}_{:06d}.png'.format(voideo_idx, frame_idx)))
                buf = io.BytesIO()
                visualizer.savefig(buf)
                buf.seek(0)
                pil_image = Image.open(buf)
                # 将PIL图像转换为NumPy数组，并确保像素尺寸和其他图像匹配
                this_frame = frames[frame_idx]
                pil_height = pil_image.height
                pil_width = pil_image.width
                plt_image = np.asarray(pil_image.resize((int(this_frame.shape[0] / pil_height * pil_width), this_frame.shape[0]), Image.BILINEAR))
                concatenated_image = np.hstack((this_frame, plt_image[:,:,:3]))
                # 关闭buffer和plt图像窗口
                buf.close()
                pose_video.append(concatenated_image)
                # imageio.imwrite(osp.join(vis_root, '{:06d}_{:06d}.png'.format(voideo_idx, frame_idx)), concatenated_image)
            imageio.mimwrite(osp.join(vis_root, '{:05d}.mp4'.format(voideo_idx)), pose_video, fps=15)
        except Exception as e:
            print(e)
