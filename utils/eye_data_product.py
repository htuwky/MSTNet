import pandas as pd
import numpy as np
import random
import os
import math
import sys

# 路径修复，确保能导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# === 1. 从 Config 读取参数 ===
NUM_PARTICIPANTS = config.NUM_SIMULATED_PEOPLE
FPS = config.EYE_SAMPLING_RATE
VIDEO_SECONDS = config.VIDEO_DURATION
OUTPUT_DIR = config.CSV_DIR

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🚀 [Config Loaded] 生成人数: {NUM_PARTICIPANTS} | 时长: {VIDEO_SECONDS}s")


# === 2. 核心逻辑 (保持不变，省略部分数学公式以节省篇幅) ===
def get_ideal_gaze(t):
    # ... (剧本逻辑保持不变，不需要动) ...
    # 这里省略中间长长的数学公式，和您原文件一致
    x, y = 0.5, 0.5
    if 0 <= t < 36:
        x = 0.5 + 0.4 * math.sin(2 * math.pi * 0.3 * t)
        y = 0.7 + 0.1 * math.sin(2 * math.pi * 0.5 * t)
    elif 36 <= t < 49:
        cycle_time = (t - 36) % 4
        row_index = int((t - 36) / 4) % 3
        x = 0.2 + 0.6 * (cycle_time / 4.0)
        y = 0.3 + row_index * 0.15
        if cycle_time < 0.2: x = 0.2
    elif 50 <= t < 90:
        x = 0.7 + 0.05 * math.sin(t)
        y = 0.3 + 0.05 * math.cos(t)
    else:
        switch = int(t / 5) % 2
        if switch == 0:
            x = 0.3
        else:
            x = 0.7
        y = 0.4
    return x, y


def generate_one_person(pid, label):
    # 使用 config.VIDEO_DURATION
    timestamps = np.arange(0, VIDEO_SECONDS, 1 / FPS)
    data_rows = []

    personal_bias_x = random.uniform(-0.05, 0.05)
    personal_bias_y = random.uniform(-0.05, 0.05)

    if label == 0:
        noise_level = 0.002;
        saccade_prob = 0.005;
        reaction_lag = 0;
        attention_span = 1.0
    else:
        noise_level = 0.015;
        saccade_prob = 0.05;
        reaction_lag = 10;
        attention_span = 0.8

    ideal_path = [get_ideal_gaze(t) for t in timestamps]
    current_x, current_y = 0.5, 0.5

    for i, t in enumerate(timestamps):
        target_idx = i
        if label == 1 and i > reaction_lag: target_idx = i - reaction_lag
        if target_idx >= len(ideal_path): target_idx = len(ideal_path) - 1

        target_x, target_y = ideal_path[target_idx]

        if label == 1 and random.random() > attention_span:
            target_x = 0.5 + random.uniform(-0.4, 0.4)
            target_y = 0.5 + random.uniform(-0.4, 0.4)

        step_size = 0.2 if label == 0 else 0.1
        current_x += (target_x - current_x) * step_size
        current_y += (target_y - current_y) * step_size

        final_x = np.clip(current_x + np.random.normal(0, noise_level) + personal_bias_x, 0, 1)
        final_y = np.clip(current_y + np.random.normal(0, noise_level) + personal_bias_y, 0, 1)

        data_rows.append({
            "Timestamp": round(t, 3),
            "Gaze_X": round(final_x, 4),
            "Gaze_Y": round(final_y, 4),
            "Label": label
        })

    return pd.DataFrame(data_rows)


# === 3. 执行 ===
for i in range(1, NUM_PARTICIPANTS + 1):
    label = 0 if i <= NUM_PARTICIPANTS // 2 else 1
    filename = f"{i:03d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if i % 50 == 0: print(f"生成中: {i}/{NUM_PARTICIPANTS} ...")
    generate_one_person(i, label).to_csv(filepath, index=False)

print("✅ 数据生成完毕！")