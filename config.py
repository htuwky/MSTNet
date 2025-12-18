import os

# ================= 1. 路径配置 (Path Configuration) =================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# [修改] 原始输入数据路径
# 注意：Windows路径前加 r 可以防止转义字符问题
CSV_DIR = r"D:\CodeProjects\Python\MSTNet\dataset\eye_data"

# [注意] 请确认您的帧文件(frame_00001.png)是否在这里，如果不是请修改
# 根据您之前的终端记录推测可能在 dataset/video 或 dataset/Frames
FRAME_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'Frames')

# 处理输出数据
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'output')
VIDEO_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'video_features.npy')

# ================= 2. 视频流核心参数 (Video Stream Core) =================
# 基于 FFmpeg 分析结果
VIDEO_FPS = 23.0       # 视频帧率
VIDEO_W = 960          # 视频宽度
VIDEO_H = 544          # 视频高度

# ================= 3. 特征提取配置 =================
CLIP_MODEL_NAME = "ViT-B/32"
CROP_SIZE = 224
EXTRACT_BATCH_SIZE = 64

# ================= 4. 模型配置 =================
INPUT_DIM = 512
PHYSIO_DIM = 2
HIDDEN_DIM = 128

# ================= 5. 训练配置 =================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cuda"
SEED = 42