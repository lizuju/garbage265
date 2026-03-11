#!/bin/bash
# ============================================================
#  garbage265 — 远程 GPU 服务器自动化部署脚本
# ============================================================

# 服务器配置
HOST="YOUR_SERVER_IP_OR_DOMAIN"
PORT="22"
USER="YOUR_USERNAME"
# 🚨 关键修改：将工程移到数据盘，防止系统盘 (30GB) 爆满
REMOTE_DIR="/root/autodl-tmp/garbage265"

# --- 待上传的文件列表 ---

# A. 基础运行文件 (直接传到根目录)
BASE_FILES=(
    "setup.sh" 
    "train_classify.py" 
    "predict.py" 
    "classname.txt" 
    "diagnose_dataset.py" 
    "camera_predict.py" 
    "README.md"
)

# B. 核心架构补丁 (传到 patches/ 目录，防止被 setup.sh 覆盖)
PATCH_FILES=(
    "yolov5/models/common.py"
    "yolov5/models/yolo.py"
    "yolov5/classify/train.py"
    "yolov5/classify/val.py"
    "yolov5/export.py"
    "yolov5/utils/hierarchical.py"
)

echo "========================================"
echo "  🚀 任务：同步代码补丁并起飞"
echo "  目标：$HOST:$PORT"
echo "========================================"

# 1. 在远程创建目录结构
ssh -p $PORT $USER@$HOST "mkdir -p $REMOTE_DIR/patches/yolov5/models $REMOTE_DIR/patches/yolov5/classify $REMOTE_DIR/patches/yolov5/utils"

# 2. 上传基础文件
echo "📤 上传基础运行文件..."
for file in "${BASE_FILES[@]}"; do
    if [ -f "$file" ]; then
        scp -P $PORT "$file" $USER@$HOST:$REMOTE_DIR/
    fi
done

# 3. 上传核心补丁 (存入 patches 备份)
echo "🔧 同步核心架构补丁..."
for file in "${PATCH_FILES[@]}"; do
    if [ -f "$file" ]; then
        scp -P $PORT "$file" $USER@$HOST:$REMOTE_DIR/patches/$file
    fi
done

echo ""
echo "========================================"
echo "  🛠️  拉起远程任务 (带原子补丁保护)"
echo "========================================"

# 我们使用 nohup 后台运行。关键点：在 setup.sh 运行完后，强制用 patches 覆盖一次源码
ssh -p $PORT $USER@$HOST "cd $REMOTE_DIR && nohup bash -c '
    set -e
    echo \"--- 1. 执行基础环境准备 --- \"
    bash setup.sh 2>&1
    
    echo \"--- 2. 应用层级架构原子补丁 --- \"
    # 将 patches 目录下的文件强行覆盖到当前目录，确保 survive setup.sh 的清理
    cp -rv patches/* . 2>&1
    
    echo \"--- 3. 启动高清层级模型训练 --- \"
    ./py_cmd train_classify.py 2>&1
' &> train.log < /dev/null &"

echo ""
echo "🚀 远程任务已挂起并加速运行！"
echo "========================================"
echo "你可以通过以下命令监控实时进度（含补丁日志）："
echo "ssh -p $PORT $USER@$HOST 'tail -f $REMOTE_DIR/train.log'"
echo "========================================"
