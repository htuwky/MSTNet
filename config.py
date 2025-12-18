import os
import torch

# ================= 1. 路径配置 (Path Config) =================
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据输入目录
CSV_DIR = os.path.join(BASE_DIR, 'dataset', 'Simulated_Eye_Data') # CSV路径
FRAME_DIR = os.path.join(BASE_DIR, 'dataset', 'frames')           # 视频帧路径

# 数据输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset', 'output')
VIDEO_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'video_features.npy') # 最终特征文件
TEMP_FEATURE_DIR = os.path.join(OUTPUT_DIR, 'temp_features')         # 临时分片文件夹

# ================= 2. 数据处理配置 (Data Process) =================
VIDEO_FPS = 23.0         # 视频帧率
EYE_SAMPLING_RATE = 60.0 # 眼动仪采样率

CLIP_MODEL_NAME = "ViT-B/32"
CROP_SIZE = 224
VIDEO_W = 960
VIDEO_H = 544

# ================= 3. 模型架构配置 (Model Architecture) =================
# [关键] 所有的维度定义都在这里，修改此处即可改变模型大小

# 输入源维度
CLIP_EMBED_DIM = 512      # CLIP 输出的维度 (Local/Global)
PHYSIO_INPUT_DIM = 3      # 生理特征输入维度: (x, y, t) 或 (x, y, t, dx, dy)

# 瓶颈层配置 (MSTNet 核心升级)
USE_BOTTLENECK = True     # 是否启用"先降维后拼接"策略
BOTTLENECK_DIM = 64       # 每个分支降维后的宽度 (针对小样本建议 64)

# 融合后的 Transformer 配置
HIDDEN_DIM = 128          # 融合后的内部维度 (不要太大，128足够)
NUM_LAYERS = 2            # Transformer 层数 (小样本用 2 层防过拟合)
NUM_HEADS = 4             # 多头注意力的头数
DROPOUT = 0.5             # [关键] 小样本强力 Dropout，推荐 0.5
FOURIER_SCALE = 10        # 傅里叶位置编码的频率缩放因子

# 序列长度
MAX_SEQ_LEN = 512         # 对应约 8.5 秒视频，捕捉长时程病理特征

# 分类头
NUM_CLASSES = 2           # 0: Healthy, 1: SZ

# ================= 4. 训练超参数 (Training Hyperparams) =================
BATCH_SIZE = 64           # 16GB 显存推荐 64
LEARNING_RATE = 1e-4      # 初始学习率
WEIGHT_DECAY = 1e-4       # 权重衰减 (L2正则化)
NUM_EPOCHS = 100          # 训练轮数
NUM_WORKERS = 0           # Windows下建议设为0，Linux可设为4

# ================= 5. 系统配置 (System) =================
SEED = 42                 # 固定随机种子，保证结果可复现
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_config():
    print("="*60)
    print("🔧 MSTNet Configuration")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Seq Len: {MAX_SEQ_LEN} | Batch Size: {BATCH_SIZE}")
    print(f"Bottleneck: {BOTTLENECK_DIM} | Hidden Dim: {HIDDEN_DIM}")
    print(f"Dropout: {DROPOUT} | Layers: {NUM_LAYERS}")
    print("="*60)