import argparse
import os
import sys
from pathlib import Path
from collections import Counter
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def load_classnames(data_dir: Path, class_file: Path) -> list:
    """参考 predict.py，按 YOLOv5 字母序逻辑对齐类名"""
    if not class_file.exists():
        print(f"❌ 找不到类名文件: {class_file}")
        sys.exit(1)
        
    with open(class_file, encoding="utf-8") as f:
        raw_names = [line.strip() for line in f if line.strip()]

    subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not subdirs:
        return raw_names

    mapped_names = []
    for s in subdirs:
        idx = int(s)
        mapped_names.append(raw_names[idx] if idx < len(raw_names) else f"类别{idx}")
    return mapped_names

def diagnose(data_dir: str, weights: str, target_classes: list, img_size: int, top_n_errors: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = Path(weights)
    yolo_dir = Path(__file__).parent / "yolov5"
    
    # 加载模型
    print(f"🔍 正在加载模型: {weights_path} (设备: {device})")
    model = torch.hub.load(str(yolo_dir), "custom", path=str(weights_path), source="local", verbose=False)
    model.eval().to(device)
    
    # 修改：对齐字母序类名
    classnames = load_classnames(Path(data_dir), Path(__file__).parent / "classname.txt")
    
    # 建立 反向映射：从文件夹数字名(str) 到 它在字母序排序列表里的索引(int)
    # 这就是 YOLOv5 训练时分配给该文件夹的真实 Label
    subdirs_sorted = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir() and d.name.isdigit()])
    folder_to_idx = {name: i for i, name in enumerate(subdirs_sorted)}
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {}

    for cls_folder in target_classes:
        cls_path = Path(data_dir) / cls_folder
        if not cls_path.exists():
            print(f"⚠️ 找不到类别目录: {cls_path}，跳过。")
            continue
            
        # 尝试从 classname 获取中文名
        try:
            real_name = classnames[int(cls_folder)]
        except:
            real_name = f"类别{cls_folder}"
            
        print(f"\n📂 正在诊断 【{real_name}】 (目录: {cls_folder})...")
        
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        if not images:
            print("   (无图片)")
            continue
            
        correct = 0
        mistakes = []
        
        for img_p in tqdm(images, desc=f"扫描 {real_name}", leave=False):
            try:
                img = Image.open(img_p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(tensor)
                    if hasattr(logits, "logits"): logits = logits.logits
                    probs = torch.softmax(logits, dim=1)[0]
                    
                conf, pred_idx = torch.max(probs, dim=0)
                pred_idx = pred_idx.item()
                conf = conf.item()
                
                # 判断是否匹配 (数字文件夹名映射后的索引 vs 预测索引)
                if pred_idx == folder_to_idx.get(cls_folder, -1):
                    correct += 1
                else:
                    mistakes.append({
                        "path": img_p.name,
                        "pred": classnames[pred_idx] if pred_idx < len(classnames) else f"类别{pred_idx}",
                        "conf": conf
                    })
            except Exception as e:
                pass

        total = len(images)
        acc = (correct / total) * 100 if total > 0 else 0
        
        print(f"📊 结果: 准确率 {acc:.1f}% ({correct}/{total})")
        
        if mistakes:
            # 统计常被错认成谁
            wrong_counts = Counter([m['pred'] for m in mistakes])
            top_wrong = wrong_counts.most_common(3)
            print(f"   ⚠️ 最常被误认为: " + ", ".join([f"{n}({c}张)" for n, c in top_wrong]))
            
            # 找出置信度最高的错误（混淆最严重的图）
            mistakes.sort(key=lambda x: x['conf'], reverse=True)
            print(f"   🚨 典型错误案例 (模型极度自信但认错的):")
            for i, m in enumerate(mistakes[:top_n_errors]):
                print(f"      - {m['path']}: 被误认为【{m['pred']}】 (置信度 {m['conf']*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="数据集诊断脚本: 扫描特定类别的误报情况")
    parser.add_argument("--data", default=".", help="数据集根目录 (包含数字文件夹的路径)")
    parser.add_argument("--weights", required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--classes", required=True, help="要诊断的文件夹名(类别编号)，逗号分隔，如: 3,12,31")
    parser.add_argument("--top", type=int, default=3, help="显示的典型错误案例数量")
    args = parser.parse_args()
    
    target_list = [c.strip() for c in args.classes.split(",")]
    diagnose(args.data, args.weights, target_list, 224, args.top)

if __name__ == "__main__":
    main()
