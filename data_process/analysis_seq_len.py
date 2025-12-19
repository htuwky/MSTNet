import numpy as np
import os
import sys
import glob

# 1. 路径设置 (确保能导入 config)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def analyze_sequence_lengths():
    # === 第一部分：数据扫描与统计 ===

    # [更新] 直接从 config 读取临时文件夹路径
    temp_dir = config.TEMP_FEATURE_DIR
    print(f"🚀 [Check] 正在扫描特征文件夹: {temp_dir} ...")

    if not os.path.exists(temp_dir):
        print("❌ 错误: 找不到 temp_features 文件夹！")
        return

    # 获取所有 .npy 文件
    npy_files = glob.glob(os.path.join(temp_dir, "*.npy"))

    if len(npy_files) == 0:
        print("❌ 文件夹为空！(请检查您的特征提取是否真的完成了)")
        return

    print(f"✅ 成功找到 {len(npy_files)} 个特征文件！开始分析...\n")

    seq_lengths = []
    fps_list = []

    # 遍历文件 (不使用 tqdm 也可以，反正很快)
    for f_path in npy_files:
        try:
            content = np.load(f_path, allow_pickle=True).item()

            # content 结构: {'local': ..., 'global': ..., 'timestamp': ...}

            # 1. 检查序列长度
            seq_len = content['local'].shape[0]
            seq_lengths.append(seq_len)

            # 2. 检查真实采样率
            timestamps = content['timestamp']
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    real_fps = len(timestamps) / duration
                    fps_list.append(real_fps)

        except Exception as e:
            print(f"⚠️ 文件损坏: {os.path.basename(f_path)} - {e}")

    if not seq_lengths:
        print("❌ 未提取到有效数据。")
        return

    seq_lengths = np.array(seq_lengths)
    avg_fps = np.mean(fps_list) if fps_list else 60.0

    print("=" * 50)
    print(f"📊 [数据验收报告]")
    print("=" * 50)
    print(f"有效样本数: {len(seq_lengths)} (预期: {config.NUM_SIMULATED_PEOPLE})")
    print(f"数据长度: {np.min(seq_lengths)} ~ {np.max(seq_lengths)} 点")
    print(f"平均长度: {np.mean(seq_lengths):.0f} 点")
    print(f"真实采样率: {avg_fps:.2f} Hz")
    print("=" * 50)

    # === 第二部分：Transformer 窗口决策 ===

    print("\n" + "=" * 110)
    print("💡 [Transformer 窗口长度决策建议 (工程优化版)]")
    print("=" * 110)

    headers = ["推荐 Seq_Len", "是否 2^n?", "对应时长(s)", "覆盖帧数", "评价"]
    row_fmt = "{:<14} | {:<16} | {:<12} | {:<12} | {}"

    print(row_fmt.format(*headers))
    print("-" * 110)

    recommendations = [
        (128, "极速模式。虽比64帧短一点，但计算最快，适合快速实验。"),
        (160, "精准对齐模式。最接近您想要的“64帧窗口”，且符合 32 倍数优化。"),
        (256, "性能/效果平衡。显存占用低，上下文比 64 帧更丰富。"),
        (320, "长序列模式。接近“128帧”窗口。"),
        (512, "大模型模式。16GB 显卡毫无压力，适合捕捉长距离依赖（如回视）。")
    ]

    for seq_len, comment in recommendations:
        is_power_of_2 = (seq_len & (seq_len - 1) == 0) and seq_len > 0
        power_str = f"✅ 是 (2^{int(np.log2(seq_len))})" if is_power_of_2 else "❌ (32倍数)"

        duration = seq_len / avg_fps
        video_frames = duration * config.VIDEO_FPS

        print(row_fmt.format(
            str(seq_len),
            power_str,
            f"{duration:.2f}",
            f"{int(video_frames)}",
            comment
        ))

    print("-" * 110)


if __name__ == "__main__":
    analyze_sequence_lengths()