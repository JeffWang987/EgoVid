import os
from os import path as osp
import numpy as np
import pandas as pd
import sys
from colmap_traj_vis import CameraPoseVisualizer, get_c2w
from colmap_pose_vis import transform_extrinsics
from imu_vis import remove_gravity, filter_noise
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
import io
from PIL import Image
from scipy.optimize import least_squares
import argparse
from matplotlib import pyplot as plt
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
    quaternions = Rot.from_matrix(rotations).as_quat()

    # 创建时间戳
    timestamps_original = np.linspace(0, 1, len(exts))  # 原始24帧的时间戳
    timestamps_interpolated = np.linspace(0, 1, len(exts)*ratio)  # 插值成120帧的时间戳

    # 创建线性插值函数
    translation_interp_func = interp1d(timestamps_original, translations, axis=0, kind='linear')

    # 对平移向量进行线性插值
    interpolated_translations = translation_interp_func(timestamps_interpolated)

    # 创建Slerp插值对象
    slerp = Slerp(timestamps_original, Rot.from_quat(quaternions))
    # Rot.from_quat(quaternions).slerp(timestamps_interpolated)

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
def integrate_imu_data(imu_data, initial_velocity, fps):
    dt = 1/fps
    imu_data = imu_data[:, :3]
    # 初始化列表来保存积分结果
    velocities = [initial_velocity]
    positions = [np.zeros(3)]
    
    # 遍历imu数据进行积分
    for i in range(1, len(imu_data)):
        
        # 积分得到速度
        new_velocity = velocities[-1] + imu_data[i] * dt
        
        # 积分得到位置
        new_position = positions[-1] + velocities[-1] * dt + 0.5 * imu_data[i] * dt**2
        
        velocities.append(new_velocity)
        positions.append(new_position)
    
    return np.array(positions)
def optimize_params(colmap_pose, imu_acc, fps):

    def residuals(params, colmap_pose, imu_acc, fps):
        init_vel = params[:3]
        rotation_vector = params[3:6]
        translation_vector = params[6:9]
        scale_factor = params[9]
        
        # 旋转和平移指的是  imu-->colmap
        imu_pose = integrate_imu_data(imu_acc, init_vel, fps)
        rotation_matrix = Rot.from_rotvec(rotation_vector).as_matrix()
        est_pose = (rotation_matrix @ imu_pose.T + translation_vector.reshape(-1, 1)).T * scale_factor

        pose_residuals = est_pose - colmap_pose
        
        return pose_residuals.flatten()

    # 残差函数的初始参数
    initial_params = np.zeros(10)  # 初始速度，旋转向量，平移向量，最后一个是尺度因子

    # 运行最小二乘法进行优化
    result = least_squares(residuals, initial_params, args=(colmap_pose, imu_acc, fps), loss='cauchy')
    init_vel = result.x[:3]
    rotation_vector = result.x[3:6]
    translation_vector = result.x[6:9]
    scale_factor = result.x[9]
    cost = result.cost

    return init_vel, rotation_vector, translation_vector, scale_factor, cost
def observation_model(x):
    return x
def state_transition(x, u, dt):
    def quat_multiply(q, r):
        """
        Multiply two quaternions.
        """
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            -x0*x1 - y0*y1 - z0*z1 + w0*w1,
            x0*w1 + y0*z1 - z0*y1 + w0*x1,
            -x0*z1 + y0*w1 + z0*x1 + w0*y1,
            x0*y1 - y0*x1 + z0*w1 + w0*z1], dtype=np.float64)

    def quat_derivative(q, omega):
        """
        Compute quaternion derivative given the current quaternion (q) and 3D angular rate vector (omega).
        """
        q_omega = np.concatenate([[0], omega])
        return 0.5 * quat_multiply(q, q_omega)


    # 从x提取位置，四元数，速度
    pos = x[:3]
    quat = x[3:7]
    vel = x[7:]
  
    # 对速度进行积分更新位置
    pos_new = pos + vel * dt
  
    # 从u提取角速度和加速度
    omega = u[3:]  # IMU测量的角速度
    acc = u[:3]  # IMU测量的线加速度
    
    # 根据角速度计算四元数的时间导数
    quat_dot = quat_derivative(quat, omega)
    # 四元数积分更新姿态
    quat_new = quat + quat_dot * dt
    if np.linalg.norm(quat_new) == 0:
        quat_new = quat_new
    else:
        quat_new /= (np.linalg.norm(quat_new))  # 保证四元数归一化
  
    # 加速度积分更新速度
    vel_new = vel + acc * dt
  
    # 更新后的完整状态
    x_new = np.concatenate([pos_new, quat_new, vel_new])
    return x_new
def jacobian_of_f(x, u, dt):
    """
    状态转移函数的雅可比矩阵
    """
    # 你需要根据状态转移函数计算它的雅可比矩阵
    # 这里将返回一个简化模型的雅可比矩阵示例。实际模型将更复杂
    Jf = np.eye(len(x))
    Jf[:3, 7:] = np.eye(3) * dt
    return Jf
def jacobian_of_h(x):
    """
    观测模型的雅可比矩阵, 因为是直接从colmap中读取的, 认为可以直接观测到
    """
    # 初始化观测雅可比矩阵
    Jh = np.zeros((7, len(x)))  # 假设有7个观测值（3个位置和4个姿态）
    # 对于位置，直接观测，因此雅可比是一个单位矩阵
    Jh[:3, :3] = np.eye(3)
    # 对于姿态，假设四元数可以直接被观测到
    Jh[3:7, 3:7] = np.eye(4)
    return Jh
def integrate_imu_rot(init_rot_matrix, angular_velocities, fps):
    rotation_matrix = init_rot_matrix
    dt = 1/fps
    rotation_matrixs = [rotation_matrix]
    for omega in angular_velocities:
        
        # 利用微小旋转近似，计算旋转向量
        theta = omega * dt
        
        # 将旋转向量转换为四元数
        quat = Rot.from_rotvec(theta)
        
        # 将四元数转换为旋转矩阵
        delta_rotation_matrix = quat.as_matrix()
        
        # 更新旋转矩阵
        rotation_matrix = rotation_matrix @ delta_rotation_matrix
        rotation_matrixs.append(rotation_matrix)
    return np.stack(rotation_matrixs)

def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10000000)
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--csv", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument("--ego4d_video_src_path", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos_all_narration')
    parser.add_argument("--save_root", type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/caption')
    parser.add_argument("--colmap_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/code/particle-sfm/outputs/egovid_alljson")
    parser.add_argument("--video_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/videos")
    parser.add_argument("--save_video_root", type=str, default="/mnt/workspace/workgroup/jeff.wang/misc/tmp/colmaptmp")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = Options()
    csv_root = args.csv
    colmap_root = args.colmap_root
    video_root = args.video_root
    vis_root =  args.save_video_root
    imu_keys = ['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']
    os.makedirs(vis_root, exist_ok=True)
    csv_info = pd.read_csv(csv_root)

    for video_idx, video_path in tqdm(enumerate(csv_info.video_id)):
        video_path = osp.join(video_root, video_path)
        this_colmap_path = osp.join(colmap_root, '{:05d}'.format(video_idx))
        pose_root = osp.join(this_colmap_path, 'colmap_outputs_converted', 'poses')
        intr_root = osp.join(this_colmap_path, 'colmap_outputs_converted', 'intrinsics')
        if not os.path.exists(pose_root):
            continue
        try:
            pose_paths =[osp.join(pose_root, pose_sub_path) for pose_sub_path in os.listdir(pose_root) if len(pose_sub_path.split('/')[-1])==9]
            w2cs = ext_interpolation(pose_paths, ratio=5, wirte_txt=True, write_path=osp.join(pose_root, 'interpolate_poses.txt'))
            c2ws = transform_extrinsics(w2cs)
            fps = csv_info.fps[video_idx]
            imu_info = np.stack([list(map(float, csv_info[imu_key][video_idx][1:-1].split(', '))) for imu_key in imu_keys])  # 6 N
            imu_info = imu_info[:, :len(c2ws)]
            imu_info[:3] = remove_gravity(imu_info[:3].transpose(1,0), fps).transpose(1,0)
            for imu_idx in range(len(imu_info)):
                imu_info[imu_idx] = filter_noise(imu_info[imu_idx], fps)
            imu_info = imu_info.transpose(1,0)
            init_vel, rotation_vector, translation_vector, scale_factor, cost = optimize_params(c2ws[:, :3, -1], imu_info, fps)

            if cost > 50:
                continue

            # 将sfm pose进行尺度调整
            # 将imu转换到camera坐标系下
            c2ws[:, :3, -1] /= scale_factor
            rotation_matrix = Rot.from_rotvec(rotation_vector).as_matrix()
            imu_acc = imu_info[:, :3]
            imu_ang_vel = imu_info[:, 3:]
            imu_acc = (rotation_matrix @ imu_acc.T).T
            init_vel = rotation_matrix @ init_vel
            imu_ang_vel = (rotation_matrix @ imu_ang_vel.T).T
            est_pose = integrate_imu_data(imu_acc, init_vel, fps) + translation_vector[None]
            imu_data = np.concatenate([imu_acc, imu_ang_vel], axis=1)

            # 初始化状态和协方差矩阵
            x = np.zeros(10)  # 假设有10个状态量  位置3，四元数4，速度3
            P = np.eye(len(x)) * 0.1  # 初始协方差矩阵
            # 假设系统噪声和观测噪声的协方差
            Q = np.eye(len(x)) * 0.01  # 系统噪声协方差
            R = np.eye(7) * 0.1  # 观测噪声协方差
            kalman_rst = []
            for imu_reading, camera_reading in zip(imu_data, c2ws):
                colmap_quat = Rot.from_matrix(camera_reading[:3,:3]).as_quat()
                colmap_pos = camera_reading[:3, -1]
                colmap_reading = np.concatenate([colmap_pos, colmap_quat])

                # 预测步骤
                x = state_transition(x, imu_reading, 1/fps)
                F = jacobian_of_f(x, imu_reading, 1/fps)
                P = F @ P @ F.T + Q

                # 更新步骤
                z = observation_model(colmap_reading)
                H = jacobian_of_h(x)
                y = z - H @ x  # 残差
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(len(x)) - K @ H) @ P
                kalman_rst.append(x)
            kalman_rst = np.stack(kalman_rst)
            
            # imu转换成矩阵形式
            init_rot_matrix = c2ws[0, :3, :3]
            imu_ang = integrate_imu_rot(init_rot_matrix, imu_ang_vel, fps)[:len(c2ws)]
            c2ws_imu = np.concatenate([imu_ang, est_pose[:,:,None]], axis=2)
            c2ws_imu = np.concatenate([c2ws_imu, c2ws[:,-1:, :]], axis=1)

            # kalman转换成矩阵形式
            kalman_rot_matrix = Rot.from_quat(kalman_rst[:, 3:7]).as_matrix()
            c2ws_kalman = np.concatenate([kalman_rot_matrix, kalman_rst[:, :3, None]], axis=2)
            c2ws_kalman = np.concatenate([c2ws_kalman, c2ws[:,-1:, :]], axis=1)

            # 可视化视频+相机轨迹
            itv = 6
            frames = read_video(video_path)
            frames = frames[::itv]
            c2ws = c2ws[::itv]
            c2ws_imu = c2ws_imu[::itv]
            c2ws_kalman = c2ws_kalman[::itv]
            with open(osp.join(intr_root, '00000.txt'), 'r') as f_int:
                this_intri = f_int.readlines()
            fxs = [float(this_intri[0].split(' ')[0]) / float(this_intri[0].split(' ')[-1]) / 2] * len(w2cs)
            boundaryx = min(0.1, max(c2ws[:,0,-1]))
            boundaryy = min(0.1, max(c2ws[:,1,-1]))
            boundaryz = min(0.1, max(c2ws[:,2,-1]))
            visualizer1 = CameraPoseVisualizer([-boundaryx, boundaryx], [-boundaryy, boundaryy], [-boundaryz, boundaryz])
            visualizer2 = CameraPoseVisualizer([-boundaryx, boundaryx], [-boundaryy, boundaryy], [-boundaryz, boundaryz])
            visualizer3 = CameraPoseVisualizer([-boundaryx, boundaryx], [-boundaryy, boundaryy], [-boundaryz, boundaryz])
            pose_video = []
            for frame_idx, (c2w, c2w_imu, c2w_kalman) in enumerate(zip(c2ws, c2ws_imu, c2ws_kalman)):
                visualizer1.extrinsic2pyramid(c2w, frame_idx / len(c2ws), hw_ratio=9/16, base_xval=1.0/50, zval=-(fxs[frame_idx])/30)
                visualizer2.extrinsic2pyramid(c2w_imu, frame_idx / len(c2ws), hw_ratio=9/16, base_xval=1.0/50, zval=-(fxs[frame_idx])/30)
                visualizer3.extrinsic2pyramid(c2w_kalman, frame_idx / len(c2ws), hw_ratio=9/16, base_xval=1.0/50, zval=-(fxs[frame_idx])/30)
                buf1 = io.BytesIO()
                buf2 = io.BytesIO()
                buf3 = io.BytesIO()
                visualizer1.savefig(buf1)
                visualizer2.savefig(buf2)
                visualizer3.savefig(buf3)
                buf1.seek(0)
                buf2.seek(0)
                buf3.seek(0)
                pil_image1 = Image.open(buf1)
                pil_image2 = Image.open(buf2)
                pil_image3 = Image.open(buf3)
                # 将PIL图像转换为NumPy数组，并确保像素尺寸和其他图像匹配
                this_frame = frames[frame_idx]
                pil_height = pil_image1.height
                pil_width = pil_image1.width
                plt_image1 = np.asarray(pil_image1.resize((int(this_frame.shape[0] / pil_height * pil_width), this_frame.shape[0]), Image.BILINEAR))
                plt_image2 = np.asarray(pil_image2.resize((int(this_frame.shape[0] / pil_height * pil_width), this_frame.shape[0]), Image.BILINEAR))
                plt_image3 = np.asarray(pil_image3.resize((int(this_frame.shape[0] / pil_height * pil_width), this_frame.shape[0]), Image.BILINEAR))
                concatenated_image = np.hstack((this_frame, plt_image1[:,:,:3], plt_image2[:,:,:3], plt_image3[:,:,:3]))
                # 关闭buffer和plt图像窗口
                buf1.close()
                buf2.close()
                buf3.close()
                pose_video.append(concatenated_image)
                # imageio.imwrite(osp.join(vis_root, '{:06d}_{:06d}.png'.format(video_idx, frame_idx)), concatenated_image)
            imageio.mimwrite(osp.join(vis_root, '{:05d}.mp4'.format(video_idx)), pose_video, fps=15)
            
            # 对比imu，colmap，kalman三种pose
            # fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            # off_ = 0.15
            # for i in range(3):
            #     x = np.arange(len(est_pose))
            #     y = est_pose[:, i]
            #     y2 = c2ws[:, i, -1]
            #     y3 = kalman_rst[:, i]
            #     ax[i, 0].plot(x, y)
            #     ax[i, 1].plot(x, y2)
            #     ax[i, 2].plot(x, y3)
            #     ax[i, 0].set_title(f'pred pose')
            #     ax[i, 1].set_title(f'colmap pose')
            #     ax[i, 2].set_title(f'kalman pose')
            #     ax[i, 0].set_xlabel('time')
            #     ax[i, 1].set_xlabel('time')
            #     ax[i, 2].set_xlabel('time')
            #     ax[i, 0].set_ylabel('motion')
            #     ax[i, 1].set_ylabel('motion')
            #     ax[i, 2].set_ylabel('motion')
            #     ax[i, 0].set_xlim(0, np.max(x))
            #     ax[i, 1].set_xlim(0, np.max(x))
            #     ax[i, 2].set_xlim(0, np.max(x))
            #     ax[i, 0].set_ylim(-off_, off_)
            #     ax[i, 1].set_ylim(-off_, off_)
            #     ax[i, 2].set_ylim(-off_, off_)
            # plt.tight_layout()
            # plt.savefig(osp.join(vis_root, '{:05d}.png'.format(video_idx)))

        except Exception as e:
            print(e)
