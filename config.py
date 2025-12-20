import os
import torch

# ================= 1. 路径配置 (Path Config) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入路径
CSV_DIR = os.path.join(BASE_DIR, 'dataset', 'eye_data')
FRAME_DIR = os.path.join(BASE_DIR, 'dataset', 'frames')

# 输出路径 (只保留分片文件夹，删除了大文件路径)
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')
TEMP_FEATURE_DIR = os.path.join(OUTPUT_DIR, 'temp_features')

# ================= 2. 数据与模拟配置 (Data & Simulation) =================
# [关键] 模拟数据的源头控制，修改这里会影响 eye_data_product.py
NUM_SIMULATED_PEOPLE = 400 # 模拟人数
VIDEO_FPS = 23.0           # 视频帧率
EYE_SAMPLING_RATE = 60.0   # 眼动采样率
VIDEO_DURATION = 343.0     # 视频总时长(秒)，用于时间戳归一化

CLIP_MODEL_NAME = "ViT-B/32"
CROP_SIZE = 224
VIDEO_W = 960
VIDEO_H = 544

# ================= 3. 模型架构配置 (Model Architecture) =================
# 输入维度
CLIP_EMBED_DIM = 512      # Visual Dim
PHYSIO_INPUT_DIM = 3      # Physio Dim: (x, y, t)

# 瓶颈层与融合
USE_BOTTLENECK = True     # 开启 MSTNet 核心策略
BOTTLENECK_DIM = 64       # 压缩到 64 维
BOTTLENECK_DIM_MOTION = 16
BOTTLENECK_DIM_GNN = 32
GNN_NODE_DIM = 12
HIDDEN_DIM = 128          # Transformer 内部维度

# Transformer 细节
NUM_LAYERS = 2            # 层数
NUM_HEADS = 4             # 头数
FFN_EXPANSION_FACTOR = 4  # [新增] FFN膨胀系数 (128 -> 512 -> 128)
DROPOUT = 0.5             # 强力 Dropout
FOURIER_SCALE = 10        # 傅里叶编码缩放

# 序列与分类
MAX_SEQ_LEN = 512         # 序列长度
NUM_CLASSES = 2           # 二分类

# ================= 4. 运行参数 (Hyperparameters) =================
# 训练参数
BATCH_SIZE = 64           # 训练用 (显存占用大)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

# 特征提取参数
EXTRACT_BATCH_SIZE = 256  # [新增] 提取特征用 (显存占用小，可以大一点)

# 系统参数
NUM_WORKERS = 0           # Windows 建议 0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_config():
    print("="*60)
    print("🔧 MSTNet Configuration (Decoupled)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Simulated People: {NUM_SIMULATED_PEOPLE}")
    print(f"Video Duration: {VIDEO_DURATION}s")
    print(f"Batch Sizes -> Train: {BATCH_SIZE} | Extract: {EXTRACT_BATCH_SIZE}")
    print("="*60)