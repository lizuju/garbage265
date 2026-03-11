#!/bin/bash
# ============================================================
#  apply_patches.sh — 将自定义层级分类补丁覆盖到 yolov5/ 目录
# ============================================================
#  用法: bash apply_patches.sh
#  说明: setup.sh 会在 clone YOLOv5 后自动调用本脚本
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches/yolov5"
YOLO_DIR="$SCRIPT_DIR/yolov5"

if [ ! -d "$YOLO_DIR" ]; then
    echo "❌ 未找到 yolov5/ 目录，请先运行 setup.sh 克隆 YOLOv5"
    exit 1
fi

if [ ! -d "$PATCH_DIR" ]; then
    echo "❌ 未找到 patches/yolov5/ 目录"
    exit 1
fi

echo "🔧 正在应用层级分类补丁..."

PATCH_FILES=(
    "classify/train.py"
    "classify/val.py"
    "models/common.py"
    "models/yolo.py"
    "export.py"
    "utils/hierarchical.py"
)

for file in "${PATCH_FILES[@]}"; do
    if [ -f "$PATCH_DIR/$file" ]; then
        cp "$PATCH_DIR/$file" "$YOLO_DIR/$file"
        echo "   ✅ $file"
    else
        echo "   ⚠️  跳过 $file (补丁文件不存在)"
    fi
done

echo "🎉 补丁应用完成！"
