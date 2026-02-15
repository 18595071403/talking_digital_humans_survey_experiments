import os
import cv2
import numpy as np
import argparse
import torch
import lpips
import face_alignment
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
import shutil
from tqdm import tqdm
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Talking Head Video Evaluation Script")
    parser.add_argument('--type', type=str, required=True, choices=['general', 'specific'], 
                        help="Dataset type: 'general' (e.g. 01.mp4) or 'specific' (e.g. Obama.mp4)")
    parser.add_argument('--project', type=str, required=True, 
                        help="Project name (e.g. wav2lip, AD_NeRF, MakeItTalk)")
    parser.add_argument('--data_root', type=str, default='data', help="Root directory for data")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda or cpu)")
    return parser.parse_args()

def get_video_frames(video_path):
    """读取视频所有帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def calculate_lmd(fa, gt_frame, pred_frame):
    """计算单帧的 Landmark Distance (LMD)"""
    # Face Alignment 接收 RGB
    gt_rgb = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
    
    try:
        gt_lands = fa.get_landmarks(gt_rgb)
        pred_lands = fa.get_landmarks(pred_rgb)
        
        if gt_lands is None or pred_lands is None:
            return None
        
        # 取第一个检测到的人脸
        dist = np.mean(np.linalg.norm(gt_lands[0] - pred_lands[0], axis=1))
        return dist
    except Exception as e:
        return None

def main():
    args = parse_args()
    
    # 1. 路径配置
    gt_dir = os.path.join(args.data_root, 'test', args.type)
    pred_dir = os.path.join(args.data_root, 'output', args.type, args.project)
    
    print(f"=== 开始评估 ===")
    print(f"模式: {args.type} | 项目: {args.project}")
    print(f"GT 路径: {gt_dir}")
    print(f"预测路径: {pred_dir}")
    
    if not os.path.exists(gt_dir):
        print(f"错误：GT 目录不存在 -> {gt_dir}")
        return
    if not os.path.exists(pred_dir):
        print(f"错误：预测结果目录不存在 -> {pred_dir}")
        return

    # 2. 模型初始化
    print("正在加载评估模型 (LPIPS, FaceAlignment)...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # LPIPS
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, face_detector='sfd') # LMD

    # FID 需要临时文件夹存储所有帧
    temp_fid_gt = 'temp_fid_gt'
    temp_fid_pred = 'temp_fid_pred'
    if os.path.exists(temp_fid_gt): shutil.rmtree(temp_fid_gt)
    if os.path.exists(temp_fid_pred): shutil.rmtree(temp_fid_pred)
    os.makedirs(temp_fid_gt, exist_ok=True)
    os.makedirs(temp_fid_pred, exist_ok=True)

    # 3. 匹配文件并开始循环
    # 过滤非视频文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.mp4', '.avi'))]
    # 简单排序，对于 01.mp4, 02.mp4 这种补零格式，字符串排序是正确的
    gt_files = sorted(gt_files)
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_lmd = []
    
    fid_frame_count = 0

    print(f"找到 {len(gt_files)} 个原始视频文件。开始逐个处理...")

    for gt_filename in tqdm(gt_files):
        # --- 核心命名逻辑 ---
        # 规则：生成文件名 = 项目名 + "_" + GT文件名
        # 例：GT="01.mp4", Project="wav2lip" -> Pred="wav2lip_01.mp4"
        name_root, ext = os.path.splitext(gt_filename)
        pred_filename = f"{args.project}_{name_root}{ext}"
        
        gt_path = os.path.join(gt_dir, gt_filename)
        pred_path = os.path.join(pred_dir, pred_filename)
        
        if not os.path.exists(pred_path):
            print(f"\n警告：找不到对应的生成视频 {pred_filename} (期望路径: {pred_path})，跳过。")
            continue
            
        # 读取视频
        gt_frames = get_video_frames(gt_path)
        pred_frames = get_video_frames(pred_path)
        
        # 确保数据非空
        if not gt_frames or not pred_frames:
            print(f"\n警告：视频 {gt_filename} 或其对应的生成视频无法读取帧，跳过。")
            continue

        # 确保帧数一致（取最小值）
        min_len = min(len(gt_frames), len(pred_frames))
        gt_frames = gt_frames[:min_len]
        pred_frames = pred_frames[:min_len]
        
        video_psnr = []
        video_ssim = []
        video_lpips = []
        video_lmd = []
        
        for i in range(min_len):
            gt_img = gt_frames[i]
            pred_img = pred_frames[i]
            
            # 保存帧用于 FID 计算
            cv2.imwrite(os.path.join(temp_fid_gt, f"{fid_frame_count}.png"), gt_img)
            cv2.imwrite(os.path.join(temp_fid_pred, f"{fid_frame_count}.png"), pred_img)
            fid_frame_count += 1
            
            # --- PSNR ---
            video_psnr.append(psnr(gt_img, pred_img, data_range=255))
            
            # --- SSIM ---
            gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            gray_pred = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
            video_ssim.append(ssim(gray_gt, gray_pred))
            
            # --- LPIPS ---
            img0 = lpips.im2tensor(gt_img).to(device)
            img1 = lpips.im2tensor(pred_img).to(device)
            with torch.no_grad():
                video_lpips.append(loss_fn_alex(img0, img1).item())
                
            # --- LMD ---
            dist = calculate_lmd(fa, gt_img, pred_img)
            if dist is not None:
                video_lmd.append(dist)
        
        # 汇总单个视频的指标
        if video_psnr: all_psnr.append(np.mean(video_psnr))
        if video_ssim: all_ssim.append(np.mean(video_ssim))
        if video_lpips: all_lpips.append(np.mean(video_lpips))
        if video_lmd: all_lmd.append(np.mean(video_lmd))

    # 4. 计算 FID
    print("正在计算 FID (这可能需要一些时间)...")
    if fid_frame_count > 0:
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [temp_fid_gt, temp_fid_pred], 
                batch_size=50, 
                device=device, 
                dims=2048
            )
        except Exception as e:
            print(f"FID 计算失败: {e}")
            fid_value = -1
    else:
        print("警告：没有提取到任何帧，无法计算 FID。")
        fid_value = -1

    # 清理临时文件
    if os.path.exists(temp_fid_gt): shutil.rmtree(temp_fid_gt)
    if os.path.exists(temp_fid_pred): shutil.rmtree(temp_fid_pred)

    # 5. 输出最终结果
    print("\n" + "="*40)
    print(f"评估报告: {args.project} ({args.type})")
    print("="*40)
    print(f"PSNR : {np.mean(all_psnr) if all_psnr else 0:.4f}")
    print(f"SSIM : {np.mean(all_ssim) if all_ssim else 0:.4f}")
    print(f"LPIPS: {np.mean(all_lpips) if all_lpips else 0:.4f}")
    print(f"LMD  : {np.mean(all_lmd) if all_lmd else 0:.4f}")
    print(f"FID  : {fid_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
