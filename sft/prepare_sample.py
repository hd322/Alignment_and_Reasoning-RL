import json
import os
import random

# --- 配置 ---
BASE_TRAIN_PATH = "/work/nvme/bfdu/mdong1/assignment5/data/math/train.jsonl"
OUTPUT_ROOT = "/work/nvme/bfdu/mdong1/assignment5/data/math/subsets"
SAMPLE_SIZES = [128, 256, 512, 1024]

def prepare_subsets():
    # 1. 加载原始数据
    if not os.path.exists(BASE_TRAIN_PATH):
        print(f"❌ 找不到原始文件: {BASE_TRAIN_PATH}")
        return

    with open(BASE_TRAIN_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"统计：原始数据共有 {len(all_data)} 条。")

    # 2. 随机打乱 (设置随机种子保证可复现)
    random.seed(42)
    random.shuffle(all_data)

    # 3. 抽取并保存
    for size in SAMPLE_SIZES:
        subset_dir = os.path.join(OUTPUT_ROOT, f"sample_{size}")
        os.makedirs(subset_dir, exist_ok=True)
        
        subset_file = os.path.join(subset_dir, "train.jsonl")
        
        # 抽取前 size 个样本
        subset_data = all_data[:size]
        
        with open(subset_file, 'w', encoding='utf-8') as f:
            for item in subset_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ 已生成 sample_{size} 目录，包含 {len(subset_data)} 条样本。")
        print(f"   路径: {subset_file}")

if __name__ == "__main__":
    prepare_subsets()