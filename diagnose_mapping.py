import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms

YOLO_DIR = Path(__file__).parent / "yolov5"
if str(YOLO_DIR) not in sys.path:
    sys.path.append(str(YOLO_DIR))

from models.common import DetectMultiBackend

def test_mapping():
    base = Path(__file__).parent
    weights = str(base / "garbage265_hierarchical" / "weights" / "best.pt")
    device = torch.device("cpu")
    model = DetectMultiBackend(weights, device=device, fuse=True)
    model.eval()

    img_size = 448
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_samples = [
        "0", "1", "10", "100", "149", "254", "264"
    ]

    base_dir = base / "train"
    
    print(f"{'Folder':<10} | {'Pred Index':<10} | {'Confidence':<10}")
    print("-" * 35)

    for folder in test_samples:
        p = base_dir / folder
        if not p.exists():
            print(f"{folder:<10} | {'NOT FOUND':<10}")
            continue
        
        # Pick the first image in the folder
        img_file = next(p.glob("*.jpg"), None)
        if not img_file:
            print(f"{folder:<10} | {'NO IMAGE':<10}")
            continue
            
        img = Image.open(img_file).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, tuple):
                logits = output[0]  # Take sub-class head
            else:
                logits = output
                
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            print(f"{folder:<10} | {idx.item():<10} | {conf.item()*100:>8.1f}%")

if __name__ == "__main__":
    test_mapping()
