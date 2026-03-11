import argparse
import sys
import os
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# 添加 yolov5 路径到 sys.path
YOLO_DIR = Path(__file__).parent / "yolov5"
if str(YOLO_DIR) not in sys.path:
    sys.path.append(str(YOLO_DIR))

from models.common import DetectMultiBackend


def load_classnames(class_file: Path, model_names=None, data_dir: Path | None = None) -> list:
    """
    优先使用 checkpoint 内置的 model.names 来构建 idx->中文名映射，
    避免因目录扫描排序/路径权限导致标签错位。
    """
    if not class_file.exists():
        print(f"❌ 找不到类名文件: {class_file}")
        sys.exit(1)
        
    with open(class_file, encoding="utf-8") as f:
        raw_names = [line.strip() for line in f if line.strip()]

    sorted_subdirs = []
    if data_dir and data_dir.exists():
        sorted_subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    # 最可靠：按训练时固化在权重里的 names 顺序映射
    if model_names:
        mapped_names = []
        for i in range(len(model_names)):
            raw = str(model_names[i]) if isinstance(model_names, dict) else str(model_names[i])
            idx = None
            if raw.isdigit():
                idx = int(raw)
            else:
                m = re.fullmatch(r"class(\d+)", raw.lower())
                if m:
                    # CoreML 常见 classN: N 是输出通道索引，不一定是原始目录ID
                    pos = int(m.group(1))
                    if sorted_subdirs and pos < len(sorted_subdirs):
                        idx = int(sorted_subdirs[pos])
                    else:
                        idx = pos
            mapped_names.append(raw_names[idx] if idx is not None and idx < len(raw_names) else raw)
        return mapped_names

    # 兜底：按 ImageFolder 的字典序目录顺序映射
    if sorted_subdirs:
            mapped_names = []
            for s in sorted_subdirs:
                idx = int(s)
                mapped_names.append(raw_names[idx] if idx < len(raw_names) else f"类别{idx}")
            return mapped_names
            
    return raw_names


def predict(img_path: str, weights: str, top_k: int, img_size: int):
    # 确认权重文件存在
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"❌ 找不到权重文件: {weights_path}")
        return

    # 确认图片存在
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"❌ 找不到图片: {img_path}")
        return

    print(f"🔍 模型加载中: {weights_path}...")
    device = torch.device("cpu") # CoreML 在本地 Mac 通常使用 CPU/Neural Engine
    
    # 使用 DetectMultiBackend 自动支持 .pt, .mlpackage, .onnx
    model = DetectMultiBackend(weights_path, device=device, fuse=True)
    model.eval()
    print(f"✅ 模型加载完成 ({model.__class__.__name__})")

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(tensor)
        
        # 处理多后端返回格式差异
        if isinstance(output, tuple):
            logits = output[0] # 层级多头 .pt 格式 (取 sub-class)
        elif isinstance(output, list):
            logits = torch.from_numpy(output[0]) if isinstance(output[0], np.ndarray) else output[0]
        elif isinstance(output, dict):
            # CoreML 返回可能是字典
            if 'sub_class' in output:
                logits = torch.from_numpy(output['sub_class'])
            else:
                logits = torch.from_numpy(list(output.values())[0])
        else:
            logits = output

        if hasattr(logits, "logits"): logits = logits.logits
        probs = torch.softmax(logits, dim=1)[0].cpu()

    # 加载类别名
    data_dir = Path(__file__).parent
    if not (data_dir / "train").exists():
        try:
            potential_dir = weights_path.parents[3]
            if (potential_dir / "train").exists():
                data_dir = potential_dir
        except IndexError:
            pass
        
    classnames = load_classnames(
        Path(__file__).parent / "classname.txt",
        model_names=model.names,
        data_dir=data_dir / "train",
    )

    # 取 Top-K
    topk_probs, topk_idxs = probs.topk(min(top_k, len(classnames)))

    print()
    print("=" * 50)
    print(f"  🖼️  图片: {img_path.name}")
    print("=" * 50)
    for rank, (prob, idx) in enumerate(zip(topk_probs, topk_idxs), 1):
        idx_int = idx.item()
        name    = classnames[idx_int] if idx_int < len(classnames) else f"Unknown({idx_int})"
        bar     = "█" * int(prob.item() * 30)
        print(f"  #{rank}  {name:<20}  {prob.item()*100:5.1f}%  {bar}")
    print("=" * 50)

    # 最终判断
    best_idx  = topk_idxs[0].item()
    best_name = classnames[best_idx] if best_idx < len(classnames) else f"Unknown({best_idx})"
    print(f"\n  🏆 预测结果: 【{best_name}】  (置信度 {topk_probs[0].item()*100:.1f}%)\n")

    # 判断大类
    major_categories = {
        "厨余垃圾": "🍳 厨余垃圾（湿垃圾）",
        "可回收物": "♻️  可回收物",
        "其他垃圾": "🗑️  其他垃圾（干垃圾）",
        "有害垃圾": "☠️  有害垃圾",
    }
    for key, label in major_categories.items():
        if best_name.startswith(key):
            print(f"  📌 垃圾大类: {label}\n")
            break


def main():
    base = Path(__file__).parent
    
    # 默认优先使用 .pt（层级分类在 CoreML 导出后可能存在输出漂移）
    default_weights = base / "garbage265_hierarchical" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = base / "garbage265_hierarchical" / "weights" / "best.mlpackage"

    parser = argparse.ArgumentParser(description="garbage265 垃圾分类推理")
    parser.add_argument("--img",     required=True,               help="待预测图片路径")
    parser.add_argument("--weights", default=str(default_weights), help="模型权重路径")
    parser.add_argument("--top",     type=int, default=5,         help="显示 Top-N 结果")
    parser.add_argument("--img-size",type=int, default=448,       help="输入图片尺寸")
    args = parser.parse_args()

    predict(args.img, args.weights, args.top, args.img_size)


if __name__ == "__main__":
    main()
