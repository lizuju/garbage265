import os
import sys
import time
import re
import cv2
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
    优先使用 checkpoint 内置的 model.names，避免标签映射错位。
    """
    if not class_file.exists():
        print(f"❌ 找不到类名文件: {class_file}")
        sys.exit(1)
        
    with open(class_file, encoding="utf-8") as f:
        raw_names = [line.strip() for line in f if line.strip()]

    sorted_subdirs = []
    if data_dir and data_dir.exists():
        sorted_subdirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])

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
                    pos = int(m.group(1))
                    if sorted_subdirs and pos < len(sorted_subdirs):
                        idx = int(sorted_subdirs[pos])
                    else:
                        idx = pos
            mapped_names.append(raw_names[idx] if idx is not None and idx < len(raw_names) else raw)
        return mapped_names

    # 兜底：从训练目录映射
    if sorted_subdirs:
            mapped_names = []
            for s in sorted_subdirs:
                idx = int(s)
                mapped_names.append(raw_names[idx] if idx < len(raw_names) else f"类别{idx}")
            return mapped_names
    
    return raw_names

def main():
    # --- 配置 ---
    weights = "garbage265_hierarchical/weights/best.pt"  # 默认优先使用 .pt
    if not Path(weights).exists():
        weights = "garbage265_hierarchical/weights/best.mlpackage"
    
    class_file = "classname.txt"
    img_size = 448 # 训练分辨率为 448
    
    # --- 模型加载 ---
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"❌ 找不到权重文件: {weights_path}, 请确认路径是否正确。")
        return

    print(f"🔍 正在加载模型: {weights_path} (加速引擎准备中...)")
    device = torch.device("cpu") # CoreML 在 Mac 上通常在 CPU/Neural Engine 调度
    
    # 使用 DetectMultiBackend 自动识别格式 (.pt, .mlpackage, .onnx)
    model = DetectMultiBackend(weights_path, device=device, fuse=True)
    model.eval()
    
    print(f"✅ 模型加载完成 (Backend: {model.__class__.__name__})")

    # --- 类名加载 ---
    data_dir = Path(__file__).parent
    if not (data_dir / "train").exists():
        try:
            potential_dir = weights_path.parents[3]
            if (potential_dir / "train").exists():
                data_dir = potential_dir
        except IndexError:
            pass
            
    classnames = load_classnames(
        Path(class_file),
        model_names=model.names,
        data_dir=data_dir / "train",
    )

    # --- 预处理 ---
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 启动摄像头 ---
    cap = cv2.VideoCapture(1) # 通常 Mac 自带摄像头是 0
    if not cap.isOpened():
        cap = cv2.VideoCapture(1) # 尝试外接摄像头
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return

    print("\n🚀 实时检测已启动！(M2 CoreML 加速已开启)")
    print("💡 提示：按 'q' 键退出预览。")
    
    # --- 稳定性算法配置 ---
    from collections import deque
    WINDOW_SIZE = 3      # 减少滞后，提升响应速度
    CONF_THRESHOLD = 0.20 # 0.4 对 265 类过高，容易频繁“无法识别”
    SHOW_THRESHOLD = 0.08 # 低于此值才显示“识别中”
    history = deque(maxlen=WINDOW_SIZE)

    # --- macOS 中文字体支持 ---
    from PIL import ImageDraw, ImageFont
    font_path = "/System/Library/Fonts/PingFang.ttc"
    if not os.path.exists(font_path):
        font_path = "/Library/Fonts/Arial Unicode.ttf"
    try:
        font = ImageFont.truetype(font_path, 32)
        font_small = ImageFont.truetype(font_path, 18)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 PIL 图片进行预处理
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = transform(img_pil).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = model(tensor)
            
            # 处理多后端返回格式差异
            if isinstance(output, tuple):
                logits = output[0] # 层级多头 .pt 格式
            elif isinstance(output, list):
                logits = torch.from_numpy(output[0]) if isinstance(output[0], np.ndarray) else output[0]
            elif isinstance(output, dict):
                # CoreML 返回可能是字典 {'sub_class': ..., 'major_class': ...} 或 {'output0': ...}
                if 'sub_class' in output:
                    logits = torch.from_numpy(output['sub_class'])
                else:
                    logits = torch.from_numpy(list(output.values())[0])
            else:
                logits = output

            if hasattr(logits, "logits"): logits = logits.logits
            probs = torch.softmax(logits, dim=1)[0].cpu()

        # --- 稳定性核心逻辑：时序平滑 ---
        history.append(probs)
        # 计算过去 N 帧的平均概率分布
        avg_probs = torch.stack(list(history)).mean(dim=0)
        
        # 获取平均概率最大的 Top 1
        conf, idx = torch.max(avg_probs, dim=0)
        conf = conf.item()
        
        # 确定显示内容
        if conf >= CONF_THRESHOLD:
            name = classnames[idx.item()] if idx.item() < len(classnames) else f"Unknown({idx.item()})"
            color = (0, 255, 0) # 绿色
        elif conf >= SHOW_THRESHOLD:
            # 低置信度时仍展示当前最可能类别，避免“完全无法识别”的体验
            name = classnames[idx.item()] if idx.item() < len(classnames) else f"Unknown({idx.item()})"
            color = (0, 220, 255) # 黄
        else:
            name = "正在扫视/识别中..."
            color = (200, 200, 200) # 灰色
        
        # 计算 FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()

        # --- 使用 PIL 绘制中文 ---
        draw = ImageDraw.Draw(img_pil)
        # 绘制黑边背景条
        draw.rectangle([0, 0, 640, 60], fill=(0, 0, 0))
        
        display_text = f"{name} ({conf*100:.1f}%)" if conf >= CONF_THRESHOLD else name
        draw.text((20, 10), display_text, font=font, fill=color)
        draw.text((520, 20), f"FPS: {fps:.1f}", font=font_small, fill=(255, 255, 255))

        # 转回 BGR 给 OpenCV 显示
        frame_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Garbage 265 [CoreML M2 Accelerated]", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
