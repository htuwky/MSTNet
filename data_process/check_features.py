import os
import numpy as np
import torch
import sys

# 路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.temporal_stream import TemporalStream
from models.motion_stream import MotionStream
from models.gnn_stream import GNNStream


def check_all_streams_pure():
    temp_dir = config.TEMP_FEATURE_DIR
    visual_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy') and '_motion' not in f])

    if not visual_files:
        print("❌ 错误：未发现特征文件。")
        return

    device = config.DEVICE
    print(f"🚀 MSTNet 全流纯净检查启动 | 设备: {device}")

    # 1. 实例化模型 (仅用于观察中间层对数值的反应)
    t_stream = TemporalStream().to(device).eval()
    m_stream = MotionStream().to(device).eval()
    g_stream = GNNStream().to(device).eval()

    # 2. 读取样本数据 (只读操作)
    fname = visual_files[0]
    subject_id = fname.replace('.npy', '')
    v_data = np.load(os.path.join(temp_dir, fname), allow_pickle=True).item()
    m_data = np.load(os.path.join(temp_dir, f"{subject_id}_motion.npy"), allow_pickle=True).item()

    # 转换为 Tensor
    local_in = torch.from_numpy(v_data['local']).float().unsqueeze(0).to(device)  # [1, S, 512]
    global_in = torch.from_numpy(v_data['global']).float().unsqueeze(0).to(device)  # [1, S, 512]
    motion_in = torch.from_numpy(m_data['motion']).float().unsqueeze(0).to(device)  # [1, S, 6]
    physio_in = torch.from_numpy(m_data['physio']).float().unsqueeze(0).to(device)  # [1, S, 3]

    # ========================== [PART 1: 原始输入数值分布] ==========================
    print("\n" + "=" * 30 + " [1. 原始输入数值分布 (硬盘原始值)] " + "=" * 30)

    # 物理坐标与原始时间戳
    p_names = ['Gaze_X (x)', 'Gaze_Y (y)', 'Timestamp (Raw t)']
    for i, name in enumerate(p_names):
        val = physio_in[0, :, i].cpu().numpy()
        print(f"📍 {name:<18} | 范围: [{val.min():.4f}, {val.max():.4f}] | 均值: {val.mean():.4f}")

    # 视觉特征
    print(
        f"\n🖼️  Local 视觉向量 (CLIP) | 范围: [{local_in.min():.4f}, {local_in.max():.4f}] | 均值: {local_in.mean():.4f}")
    print(
        f"🌍 Global 视觉向量 (CLIP)| 范围: [{global_in.min():.4f}, {global_in.max():.4f}] | 均值: {global_in.mean():.4f}")

    # 运动向量分量
    print(f"\n🏃 Motion 运动分量拆解:")
    m_names = ['u_local', 'v_local', 'u_global', 'v_global', 'v_eye_x', 'v_eye_y']
    for i, name in enumerate(m_names):
        val = motion_in[0, :, i].cpu().numpy()
        print(f"   - {name:<12} | 范围: [{val.min():.4f}, {val.max():.4f}] | 均值: {val.mean():.4f}")

    # ========================== [PART 2: 模型内部中间层响应] ==========================
    print("\n" + "=" * 30 + " [2. 模型内部中间层 (Bottleneck) 响应] " + "=" * 30)

    with torch.no_grad():
        # A. 时序流内部变换
        x_local_t = t_stream.local_proj(local_in)
        # 仅在内存中模拟归一化给模型看，不改数据
        physio_temp = physio_in.clone()
        physio_temp[:, :, 2] /= config.VIDEO_DURATION
        x_physio_t = t_stream.physio_mapper(physio_temp)
        x_fused_t = t_stream.fusion_proj(torch.cat([x_local_t, t_stream.global_proj(global_in), x_physio_t], dim=-1))

        print(
            f"🕒 Temporal -> Transformer 输入层 | 范围: [{x_fused_t.min():.4f}, {x_fused_t.max():.4f}] | 均值: {x_fused_t.mean():.4f}")

        # B. 运动流内部变换
        m_hidden = m_stream.input_proj(motion_in)
        m_bottleneck = m_stream.bottleneck(m_stream.res_block2(m_stream.res_block1(m_hidden)))
        print(
            f"🚀 Motion   -> Bottleneck 输出层   | 范围: [{m_bottleneck.min():.4f}, {m_bottleneck.max():.4f}] | 均值: {m_bottleneck.mean():.4f}")

        # C. GNN 流内部变换
        g_node = g_stream.node_encoder(local_in)
        print(
            f"🧩 GNN      -> 节点编码后量级     | 范围: [{g_node.min():.4f}, {g_node.max():.4f}] | 均值: {g_node.mean():.4f}")

    print("\n" + "=" * 80)
    print("✅ 纯净检查完成。所有硬盘数据未受影响。")


if __name__ == "__main__":
    check_all_streams_pure()