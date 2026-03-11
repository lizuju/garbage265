#!/bin/bash
# --- 导出 PATH 并初始化 Conda ---
set -e
export PATH=/root/miniconda3/bin:/root/anaconda3/bin:/opt/conda/bin:$PATH:/usr/local/bin
[ -f "/root/miniconda3/etc/profile.d/conda.sh" ] && source "/root/miniconda3/etc/profile.d/conda.sh"
[ -f "/opt/conda/etc/profile.d/conda.sh" ] && source "/opt/conda/etc/profile.d/conda.sh"

# --- 磁盘空间紧急清理 ---
echo "🧹 正在准备清洁的运行环境..."
rm -rf ~/.cache/pip ~/.cache/huggingface ~/.cache/torch/hub
# 清理 Conda 包缓存
if command -v conda &> /dev/null; then
    conda clean --all -y &>/dev/null || true
fi

# --- 强制清理残留进程 (精准打击，避免自杀) ---
echo "⚔️  正在清理残留后台进程 (不影响当前安装)..."
# 仅杀掉真正的训练进程，排除掉当前正在运行的 shell 脚本及其父进程
CUR_PID=$$
PPID_VAL=$PPID
ps aux | grep -E 'python|py_cmd' | grep -E 'train.py|modelscope|train_classify.py' | grep -v 'grep' | awk -v cur=$CUR_PID -v parent=$PPID_VAL '$2 != cur && $2 != parent {print $2}' | xargs kill -9 2>/dev/null || true

# --- 磁盘空间发现 ---
DATA_DISK=""
for d in "/autodl-tmp" "/root/autodl-tmp" "/mnt/workspace" "/data"; do
    if [ -d "$d" ] && [ -w "$d" ]; then
        DATA_DISK="$d"
        break
    fi
done
if [ -z "$DATA_DISK" ]; then
    DATA_DISK=$(df -h | grep '^/dev/' | sort -k4 -hr | awk '{print $6}' | while read m; do if [ -w "$m" ] && [ "$m" != "/" ]; then echo "$m"; break; fi; done)
fi
if [ ! -z "$DATA_DISK" ] && [ "$DATA_DISK" != "/" ]; then
    echo "💎 数据存储盘: $DATA_DISK"
    export MODELSCOPE_CACHE="$DATA_DISK/modelscope_cache"
else
    export MODELSCOPE_CACHE="$(pwd)/modelscope_cache"
fi
mkdir -p "$MODELSCOPE_CACHE"

# --- Python 探测 ---
PYTHON_CMD=$(which python3.10 || which python3.9 || which python3)
ln -sf "$PYTHON_CMD" py_cmd

# --- Step 1: YOLOv5 仓库 ---
if [ ! -f "yolov5/requirements.txt" ]; then
    echo "📂 正在拉取 YOLOv5 源码..."
    rm -rf yolov5
    git clone --depth 1 https://ghp.ci/https://github.com/ultralytics/yolov5.git || git clone --depth 1 https://github.com/ultralytics/yolov5.git
    
    # 自动应用我们的环境或架构分类补丁
    if [ -f "apply_patches.sh" ]; then
        bash apply_patches.sh
    else
        echo "⚠️  未找到 apply_patches.sh，跳过自定义补丁应用"
    fi
fi

# --- Step 2: 依赖安装 ---
echo "📦 正在检查并同步依赖..."
$PYTHON_CMD -m pip install --no-cache-dir Pillow tqdm matplotlib seaborn pandas oss2 addict datasets==3.6.0 "modelscope[dataset]" albumentations==1.3.1 &>/dev/null

# --- Step 3: GPU 验证 ---
./py_cmd - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"✅ GPU 就绪: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("⚠️  警告: 未发现 CUDA GPU")
EOF

# --- Step 4: 数据集下载与软链接 ---
./py_cmd - <<'EOF'
import os, shutil, sys
from pathlib import Path
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

def prepare():
    if Path("train").exists() and any(Path("train").iterdir()):
        print("✅ 数据集物理链接已存在，跳过。")
        return
    
    cache_dir = os.environ.get('MODELSCOPE_CACHE')
    print("📂 正在确认/下载 13GB 数据集...")
    MsDataset.load('garbage265', namespace='tany0699', subset_name='default', split='train', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    MsDataset.load('garbage265', namespace='tany0699', subset_name='default', split='validation', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    
    root = Path(cache_dir)
    train_paths = list(root.glob("**/train"))
    val_paths = list(root.glob("**/val"))
    
    if train_paths and val_paths:
        t = max(train_paths, key=lambda p: len(str(p)))
        v = max(val_paths, key=lambda p: len(str(p)))
        if os.path.exists("train"): os.remove("train") if os.path.islink("train") else shutil.rmtree("train")
        if os.path.exists("val"): os.remove("val") if os.path.islink("val") else shutil.rmtree("val")
        os.symlink(t.absolute(), "train")
        os.symlink(v.absolute(), "val")
        print("🔗 物理文件夹链接建立成功。")

prepare()
EOF

# --- Step 5: 源码热补丁 (补丁总汇) ---
echo "🔧 正在应用鲁棒性与性能补丁..."
# A. Dataloader 防崩溃补丁 (覆盖 Detection & Classification)
(cd yolov5 && git checkout utils/dataloaders.py 2>/dev/null) || true
./py_cmd - <<'EOF'
from pathlib import Path
f = Path("yolov5/utils/dataloaders.py")
if f.exists():
    c = f.read_text()
    # 1. 保护目标检测 Dataloader
    old1 = '        if self.album_transforms:'
    new1 = '        if getattr(im, "size", 0) == 0: im = __import__("numpy").zeros((224, 224, 3), dtype=__import__("numpy").uint8)\n        if self.album_transforms:'
    if old1 in c and 'getattr(im,' not in c:
        c = c.replace(old1, new1)
        
    # 2. 保护图像分类 Dataloader (解决 cv2.cvtColor 崩溃)
    old2 = 'cv2.cvtColor(im, cv2.COLOR_BGR2RGB)'
    new2 = 'cv2.cvtColor(im if getattr(im, "size", 0) > 0 else __import__("numpy").zeros((224, 224, 3), dtype=__import__("numpy").uint8), cv2.COLOR_BGR2RGB)'
    if old2 in c:
        c = c.replace(old2, new2)
        
    f.write_text(c)
    print("   - [OK] Dataloader 空图片保护 (全面拦截 CVD Error)")
EOF

# B. 验证频率优化补丁 (每 5 轮验一次)
# [注意] 此处已由 Hierarchical Patch 替代，不再在 setup.sh 中硬编码修改
# (cd yolov5 && git checkout classify/train.py 2>/dev/null) || true
# ./py_cmd - <<'EOF'
# ... (相关代码已注释)
# EOF

# C. 恢复训练 Epoch 强制重写 (已禁用，以支持自定义 Epoch 训练)
# ./py_cmd - <<'EOF'
# ... (相关代码已注释)
# EOF

# --- Step 6: 深度图片体检 ---
echo "🔍 正在进行深度图片损坏扫描 (确保训练中途不崩溃)..."
./py_cmd - <<'EOF'
import os, cv2, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
bad = 0
for s in ['train', 'val']:
    p = Path(s)
    if not p.exists(): continue
    imgs = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        imgs.extend(list(p.rglob(ext)))
    for f in tqdm(imgs, desc=f"Scanning {s}"):
        try:
            with Image.open(f) as i: i.verify()
            im = cv2.imread(str(f))
            if im is None or im.size == 0: raise ValueError
        except:
            os.remove(f); bad += 1
if bad > 0: print(f"✅ 已清理 {bad} 张损坏图片。")
else: print("✨ 数据集状态完美。")
EOF

echo ""
echo "🚀 到位！环境已全副武装，准备起飞。"
