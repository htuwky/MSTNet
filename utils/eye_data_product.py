import pandas as pd
import numpy as np
import random
import os
import math

# === 1. 设置参数 ===
NUM_PARTICIPANTS = 400  # 生成400人
FPS = 60  # 采样率 60Hz
VIDEO_SECONDS = 343  # 5分43秒
OUTPUT_DIR = "dataset/eye_data"

# 创建输出文件夹
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"正在生成 {NUM_PARTICIPANTS} 人的纯净眼动数据，采样率 {FPS}Hz...")


# === 2. 定义核心生成函数 ===

def get_ideal_gaze(t):
    """
    剧本逻辑：定义正常人应该看哪里的 (x, y)
    """
    x, y = 0.5, 0.5

    # 场景1: Tom & Jerry (0-36s)
    if 0 <= t < 36:
        x = 0.5 + 0.4 * math.sin(2 * math.pi * 0.3 * t)
        y = 0.7 + 0.1 * math.sin(2 * math.pi * 0.5 * t)

    # 场景2: 文字指令 (36-49s)
    elif 36 <= t < 49:
        cycle_time = (t - 36) % 4
        row_index = int((t - 36) / 4) % 3
        x = 0.2 + 0.6 * (cycle_time / 4.0)
        y = 0.3 + row_index * 0.15
        if cycle_time < 0.2: x = 0.2

    # 场景3: 疯狂原始人 (50-90s)
    elif 50 <= t < 90:
        x = 0.7 + 0.05 * math.sin(t)
        y = 0.3 + 0.05 * math.cos(t)

    # 场景4: 交互对话 (其他时间)
    else:
        switch = int(t / 5) % 2
        if switch == 0:
            x = 0.3
        else:
            x = 0.7
        y = 0.4

    return x, y


def generate_one_person(pid, label):
    """
    生成一个人的所有数据 (不包含 Image_File)
    """
    timestamps = np.arange(0, VIDEO_SECONDS, 1 / FPS)
    data_rows = []

    personal_bias_x = random.uniform(-0.05, 0.05)
    personal_bias_y = random.uniform(-0.05, 0.05)

    if label == 0:  # 健康
        noise_level = 0.002
        saccade_prob = 0.005
        reaction_lag = 0
        attention_span = 1.0
    else:  # 患病 (SZ)
        noise_level = 0.015
        saccade_prob = 0.05
        reaction_lag = 10
        attention_span = 0.8

    ideal_path = [get_ideal_gaze(t) for t in timestamps]
    current_x, current_y = 0.5, 0.5

    for i, t in enumerate(timestamps):
        # 1. 计算目标
        target_idx = i
        if label == 1 and i > reaction_lag:
            target_idx = i - reaction_lag

        target_x, target_y = ideal_path[target_idx]

        # 2. 模拟走神
        if label == 1 and random.random() > attention_span:
            target_x = 0.5 + random.uniform(-0.4, 0.4)
            target_y = 0.5 + random.uniform(-0.4, 0.4)

        # 3. 平滑移动
        step_size = 0.2 if label == 0 else 0.1
        current_x += (target_x - current_x) * step_size
        current_y += (target_y - current_y) * step_size

        # 4. 添加噪声
        final_x = current_x + np.random.normal(0, noise_level) + personal_bias_x
        final_y = current_y + np.random.normal(0, noise_level) + personal_bias_y

        # 5. 模拟跳变
        if random.random() < saccade_prob:
            final_x += random.uniform(-0.1, 0.1)
            final_y += random.uniform(-0.1, 0.1)

        # 6. 限制范围
        final_x = np.clip(final_x, 0.0, 1.0)
        final_y = np.clip(final_y, 0.0, 1.0)

        # === 写入数据 (已移除 Image_File) ===
        data_rows.append({
            "Timestamp": round(t, 3),
            "Gaze_X": round(final_x, 4),
            "Gaze_Y": round(final_y, 4),
            "Label": label
        })

    return pd.DataFrame(data_rows)


# === 3. 执行生成循环 ===

for i in range(1, NUM_PARTICIPANTS + 1):
    label = 0 if i <= NUM_PARTICIPANTS // 2 else 1
    label_str = "Healthy" if label == 0 else "SZ"

    filename = f"{i:03d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if i % 20 == 0:
        print(f"进度: 正在生成 {filename} ...")

    df = generate_one_person(i, label)
    df.to_csv(filepath, index=False)

print("\n=== 全部生成完毕！ ===")
print(f"CSV文件已保存在: {os.path.abspath(OUTPUT_DIR)}")