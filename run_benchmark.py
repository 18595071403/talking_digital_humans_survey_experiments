import os
import sys
import subprocess
import time
import cv2  # 需要 pip install opencv-python
from pathlib import Path

# ================= 配置区域 =================

PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"

# 定义模型配置
MODEL_CONFIGS = {
    # === 模型 1: Wav2Lip ===
    "Wav2lip": {
        "category": "general",
        # 请确保路径正确
        "python_path": r"C:\Users\Lenovo\anaconda3\envs\wav2lip\python.exe",
        "script_path": MODELS_ROOT / "Wav2lip/inference.py",
        "checkpoint": "checkpoints/wav2lip.pth",
        "cmd_template": (
            "--checkpoint_path {checkpoint} "
            "--face {video} "
            "--audio {audio} "
            "--outfile {out} "
            "--pads 0 10 0 0 --nosmooth"
        ),
    },
    
    # === 模型 2: AD-NeRF ===
    "AD-NeRF": {
        "category": "specific",
        # 请确保路径正确
        "python_path": r"C:\Users\Lenovo\anaconda3\envs\adnerf\python.exe",
        "script_path": MODELS_ROOT / "AD-NeRF/test.py",
        "checkpoint": "checkpoints/obama.pth",
        "cmd_template": (
            "--config {checkpoint} "
            "--video {video} "
            "--audio {audio} "
            "--save_path {out}"
        ),
    }
    # 在此处添加更多模型...
}

# ================= 辅助函数 =================

def get_video_frame_count(video_path):
    """读取视频文件的总帧数"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def run_model(model_name, config):
    print(f"\n[{model_name}] 正在准备推理...")
    
    category = config["category"]
    
    # 定义输入输出目录
    input_video_dir = DATA_ROOT / "test" / category
    output_video_dir = DATA_ROOT / "output" / category / model_name
    
    # 确保输出目录存在
    os.makedirs(output_video_dir, exist_ok=True)
    
    # 获取测试视频列表 (这里以 HDTF.txt 或文件夹遍历为例，简化为遍历文件夹)
    # 假设 test/general 下面就是 01.mp4, 02.mp4...
    video_files = sorted([f for f in os.listdir(input_video_dir) if f.endswith('.mp4')])
    
    # 为了演示，假设音频和视频是同一个文件，或者你有专门的音频目录
    # 这里默认使用视频文件作为音频输入 (很多Talking Head项目支持这样做)
    
    total_fps = 0
    count = 0
    
    print(f"找到 {len(video_files)} 个测试视频，开始处理...")

    for video_file in video_files:
        video_path = input_video_dir / video_file
        # 输出文件名规则: 模型名_原文件名.mp4
        out_name = f"{model_name}_{video_file}"
        out_path = output_video_dir / out_name
        
        # 构建命令
        # 注意：这里假设 cmd_template 里的参数名匹配
        cmd_str = config["cmd_template"].format(
            checkpoint=config["checkpoint"],
            video=str(video_path),
            audio=str(video_path), # 简单起见，音频使用视频源
            out=str(out_path)
        )
        
        full_cmd = f'"{config["python_path"]}" "{config["script_path"]}" {cmd_str}'
        
        print(f"正在处理: {video_file} ...")
        
        # === 核心：FPS 测量开始 ===
        # 使用 perf_counter 计时更精确
        start_time = time.perf_counter()
        
        try:
            # 执行命令
            subprocess.run(full_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"出错: {e}")
            continue
            
        # === 核心：FPS 测量结束 ===
        end_time = time.perf_counter()
        
        # 计算耗时
        duration = end_time - start_time
        
        # 获取生成视频的帧数来计算 FPS
        if os.path.exists(out_path):
            frames = get_video_frame_count(out_path)
            if frames > 0 and duration > 0:
                fps = frames / duration
                print(f"  -> 耗时: {duration:.2f}s | 帧数: {frames} | FPS: {fps:.2f}")
                total_fps += fps
                count += 1
            else:
                print("  -> 生成视频无效或时长为0")
        else:
            print("  -> 未找到生成文件")

    # 输出该模型的平均 FPS
    if count > 0:
        avg_fps = total_fps / count
        print(f"\n=== {model_name} 评测完成 ===")
        print(f"平均推理速度 (End-to-End): {avg_fps:.2f} FPS")
        print("注意：此FPS包含模型加载时间，纯GPU推理速度会更快。")
    else:
        print(f"\n{model_name} 没有成功处理任何视频。")

# ================= 主程序 =================

if __name__ == "__main__":
    # 遍历配置中的所有模型进行评测
    for model_name, config in MODEL_CONFIGS.items():
        run_model(model_name, config)
