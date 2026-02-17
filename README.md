# 虚拟说话数字人视频生成：代表性工作评测实验报告

## 1. 实验对象与分类

本实验共选取 **13项** 代表性工作，根据技术路线分为两大类：

### 1.1 基于GAN与扩散模型的通用模型（共7项）
此类模型主要针对任意人脸进行驱动 (**General   一般**)：

**GAN类 (4项)：**
*   [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
*   [MakeItTalk](https://github.com/AdobeResearch/MakeItTalk)
*   [PC-AVS](https://github.com/Hangz-nju-cuhk/PC-AVS)
*   [VideoReTalking](https://github.com/vinthony/video-retalking)

**扩散模型类 (3项)：**
*   [DreamTalk](https://github.com/ali-vilab/dreamtalk)
*   [Hallo](https://github.com/fudan-generative-vision/hallo)
*   [EchoMimic](https://github.com/BadToBest/EchoMimic)

### 1.2 基于NeRF与3DGS的特定人脸模型（共6项）
此类模型通常需要针对特定人物进行训练 (**Specific**)：

**NeRF类 (4项)：**
*   [AD-NeRF](https://github.com/YudongGuo/AD-NeRF)
*   [GeneFace](https://github.com/Yerfor/GeneFace)
*   [ER-NeRF](https://github.com/Fictionarry/ER-NeRF)
*   [SyncTalk](https://github.com/ziqibai/SyncTalk)

**3D高斯溅射（3DGS）：**
*   [GaussianTalker](https://github.com/KU-CVLAB/GaussianTalker)
*   [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian)

---

## 2. 实验设计与控制变量

为了确保定性结果的公平性和可复现性，严格定义以下实验设置：

### 2.1 硬件与环境
*   **统一硬件平台**：所有模型的训练、测试与推理均在单张 **NVIDIA RTX 4090** 显卡上完成。

### 2.2 模型配置策略
*   **通用模型**：严格遵循原论文官方代码及推荐超参数，仅使用官方预训练权重，不进行微调，直接在统一数据集上进行推理。
*   **特定人脸模型**：沿用原论文官方代码及推荐超参数，在统一数据集上重新训练后进行测试。
*   **音频特征统一**：为消除音频处理差异，将音频特征提取器统一为 **DeepSpeech**。

### 2.3 数据集选择与处理
*   **通用模型测试集**：
    *   **来源**：[HDTF 数据集](https://huggingface.co/datasets/global-optima-research/HDTF/tree/main)
    *   **数量**：随机选取 50 个视频片段（具体列表见 `.\data\test\general\HDTF.txt`）。
*   **特定人脸模型训练/测试集**：
    *   **对象**：选取 4 位人物（Macron, Lieu, May, Shaheen），覆盖不同肤色、性别与国家。
    *   **预处理**：统一处理为 **25 FPS**，分辨率统一为 **512×512**（Obama 案例为 450×450），单条时长 3–5 分钟。
*   **输出规范**：所有模型生成的视频必须与 Ground Truth (GT) 帧数对齐、分辨率一致。

### 2.4 评价指标体系
*   **生成速度 (FPS)、生成质量 (PSNR, LPIPS, SSIM, FID)、精准度 (LMD)**：均在单张 NVIDIA RTX 4090 显卡上使用统一的评价指标体系评测。
*   **口型同步 (LSE-D, LSE-C)**：使用 [Wav2Lip 官方仓库](https://huggingface.co/camenduru/Wav2Lip/tree/main/checkpoints) 提供的评测代码进行评估。

---

## 3. 实验全流程操作指南

### 3.1 训练与测试环境搭建

#### (1) 数据集准备
1.  从 [Hugging Face](https://huggingface.co/datasets/global-optima-research/HDTF/tree/main) 下载 `clips.zip`。
2.  根据 `\data\test\general\HDTF.txt` 挑选出 50 个目标视频。
3.  按顺序重命名后，复制到 `data/test/general/` 文件夹下。

#### (2) 自动化推理脚本配置 (`run_benchmark.py`)
修改 `run_benchmark.py` 中的 `MODEL_CONFIGS` 部分，确保路径指向正确的环境（Python解释器路径需通过 `where python` 获取）。

**配置示例代码：**

```python   ”“python
# ================= 配置区域 (请修改这里) =================
from pathlib import Path   从pathlib导入路径

# 1. 自动获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute() PROJECT_ROOT = Path(__file__).parent.absolute（）
MODELS_ROOT = PROJECT_ROOT / "models"MODELS_ROOT = PROJECT_ROOT / "models"；

MODEL_CONFIGS = {   Model_configs = {MODEL_CONFIGS = {MODEL_CONFIGS = {
    # === 模型 1: Wav2Lip ===
    # 项目链接: https://github.com/Rudrabha/Wav2Lip
    "Wav2lip": {    "Wav2lip": {
        "category": "general",    "category": "general",
        # 【重要】修改为你本地 Wav2Lip 环境的 python.exe 绝对路径
        "python_path": r"C:\Users\Lenovo\anaconda3\envs\wav2lip\python.exe", "python_path": r"C:\Users\Lenovo\anaconda3\envs\wav2lip\python.exe",
        "script_path": MODELS_ROOT / "Wav2lip/inference.py", "script_path": MODELS_ROOT / "Wav2lip/inference.py",
        "checkpoint": "checkpoints/wav2lip.pth"," checkpoint": "牌号/ wav2lip.pth"
        "cmd_template": (   " cmd_template":(
            "--checkpoint_path {checkpoint} "——checkpoint_path {checkpoint} "；
            "--face {video} "   ——face {video} "；
            "--audio {audio} "   ——音频{音频}"；
            "--outfile {out} "   ——outfile {out} "；
            "--pads 0 10 0 0 --nosmooth"——pads 0 0 0 0 0——nosmooth"；
        ),
    },
    
    # === 模型 2: AD-NeRF ===
    # 项目链接: https://github.com/YudongGuo/AD-NeRF
    "AD-NeRF": {    "AD-NeRF": {
        "category": "specific",    "category": "specific",
        "python_path": r"C:\Users\Lenovo\anaconda3\envs\adnerf\python.exe","python_path": r"C:\Users\Lenovo\anaconda3\envs\adnerf\python.exe",
        "script_path": MODELS_ROOT / "AD-NeRF/test.py",script_path": MODELS_ROOT / "AD-NeRF/test.py"；
        "checkpoint": "checkpoints/obama.pth"," checkpoint": "牌号/ obama.pth"
        "cmd_template": (   " cmd_template":(
            "--config {checkpoint} "   ——config {checkpoint} "；
            "--video {video} "   ——视频{视频}"；
            "--audio {audio} "   ——音频{音频}"；
            "--save_path {out}"   “——save_path}{出来“"--save_path {out}"   “——save_path}{出来“
        ),
    }
}
```

#### (3) 运行推理
在根目录下执行：
```bash   ”“bash
python run_benchmark.py
```
**检查结果**：对应视频将生成在 `data/output/general/[模型名]/` 或 `data/output/specific/[模型名]/` 下。

### 3.2 评测指标环境搭建
为了统一评测标准，需建立独立的评估环境。

1.  **创建环境** (推荐 Python 3.10 + CUDA 12.1)：
    ```bash   ”“bash
    conda create -n digital_human_eval python=3.10 -yConda create -n digital_human_eval python=3.10 -y
    conda activate digital_human_eval激活digital_human_eval
    ```
2.  **安装 PyTorch**：
    ```bash   ”“bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121PIP安装torch torchvision torchaudio——index-url https://download.pytorch.org/whl/cu121
    ```
3.  **安装依赖**：
    ```bash   ”“bash
    pip install -r requirements.txtPIP install -r requirements.txt   “bash”;“bash
    ```

### 3.3 目录结构与文件命名规范
为配合评测脚本，请严格保持以下目录结构。将推理生成的视频按 `[项目名]_[原文件名].mp4` 格式重命名并放入对应文件夹。

```text   ' ' '文本
Project_Root/
├── evaluation.py             # 评测脚本
├── run_benchmark.py          # 自动化推理脚本
└── data/
    ├── test/                 # 原始真实视频 (GT)
    │   ├── general/          # 通用测试集 (01.mp4 ... 50.mp4)
    │   └── specific/         # 特定人测试集 (Obama.mp4, Macron.mp4)
    │
    └── output/               # 推理生成的视频
        ├── general/                    # 通用模型推理生成视频 
        │   ├── Wav2lip/      
        │   │   ├── Wav2lip_01.mp4
        │   │   └── ...…
        │   └── MakeItTalk/   
        │       ├── MakeItTalk_01.mp4
        │       └── ...│美国…
        │
        └── specific/                    # 特定数字人模型推理生成视频
            └── AD_NeRF/    
                ├── AD_NeRF_Obama.mp4
                └── ...
```

### 3.4 执行评测
所有指标采用多视频平均值。

**A：图像质量与精准度评测 (PSNR, SSIM, LPIPS, FID, LMD)**

```bash   ”“bash   “bash”;“bash
# 评测通用模型 (如 Wav2Lip)
python evaluation.py --type general --project wav2lipPython evaluate .py——type general——project wav2lippython evaluate .py—— —project wav2lip

# 评测特定模型 (如 AD-NeRF)
python evaluation.py --type specific --project AD_NeRFpython evaluation.py——type specific——project AD_NeRF .pypython evaluate .py—— —project AD_NeRF .py
```

**B：口型同步评测 (LSE-D, LSE-C)**
需使用 Wav2Lip 官方仓库提供的评估代码：
1.  参考 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) 下载 checkpoints。
2.  编写批量循环脚本调用其 `evaluation/scores.py` 进行打分。

**C：推理速度**
在 `run_benchmark.py` 文件推理过程中评测。

---

## 4. 局限性与后续计划

1.  **数据处理冗余**：由于各项目对输入数据格式要求不一（如FPS、裁切方式），当前复现过程中存在数据处理冗余。后续将整理“最大公约数”数据处理流，发布统一的数据预处理方案。
