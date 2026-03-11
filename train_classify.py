"""
garbage265 垃圾分类 — YOLOv5 Classification 训练脚本
------------------------------------------------------
硬件   : RTX 5090 / 32 GB VRAM
数据集 : tany0699/garbage265  (265 类, ~147k 张图)
模型   : yolov5x-cls  (ImageNet 预训练, 最大/最准确版本)
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# 配置区（可根据需要修改）
# ─────────────────────────────────────────────
CONFIG = {
    # 模型选择: yolov5n-cls / yolov5s-cls / yolov5m-cls / yolov5l-cls / yolov5x-cls
    "model"  : "yolov5x-cls.pt",

    # 输入图片尺寸（Phase 3/4 保持 448）
    "img"    : 448,

    # Batch Size
    "batch"  : 128,

    # 训练轮数
    "epochs" : 80,

    # GPU 设备编号
    "device" : "0",

    # DataLoader 工作进程数
    "workers": 8,

    # 实验名称（Hierarchical 版本使用相应后缀）
    "name"   : "garbage265_hierarchical",

    # 学习率
    "lr0"    : 0.001,

    # 标签平滑
    "label_smoothing": 0.1,

    # 层级架构配置 (Phase 4 新增)
    "nc_major"    : 4,    # 大类数量 (厨余, 可回收, 其他, 有害)
    "major_weight": 0.5,  # 大类损失权重
    "val_period"  : 5,    # 每隔多少轮验证一次

    # 数据集根目录
    "data"   : str(Path(__file__).parent),
}
# ─────────────────────────────────────────────


def check_yolov5():
    """确保 yolov5 仓库存在"""
    yolo_path = Path(__file__).parent / "yolov5"
    if not yolo_path.exists():
        print("❌ 未找到 yolov5/ 目录，请先运行: bash setup.sh")
        sys.exit(1)
    return yolo_path


def check_dataset(data_dir: Path):
    """验证 train/ val/ 目录结构"""
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists():
        print(f"❌ 找不到训练集目录: {train_dir}")
        sys.exit(1)
    if not val_dir.exists():
        print(f"❌ 找不到验证集目录: {val_dir}")
        sys.exit(1)

    # 统计类别数（排除非目录条目）
    n_train = sum(1 for p in train_dir.iterdir() if p.is_dir())
    n_val   = sum(1 for p in val_dir.iterdir()   if p.is_dir())

    print(f"📂 数据集目录  : {data_dir}")
    print(f"📊 train/ 类别 : {n_train}")
    print(f"📊 val/   类别 : {n_val}")

    if n_train != 265 or n_val != 265:
        print(f"⚠️  警告: 期望 265 个类别，实际 train={n_train}, val={n_val}")

    return n_train


def build_command(yolo_path: Path, cfg: dict) -> list:
    """构造 YOLOv5 classify/train.py 的命令行参数"""
    train_script = yolo_path / "classify" / "train.py"

    # 自动探测是否可以断点续传
    last_ckpt = Path(__file__).parent / "yolov5" / "runs" / "train-cls" / cfg["name"] / "weights" / "last.pt"
    
    cmd = [
        sys.executable, str(train_script),
        "--model",      cfg["model"],
        "--data",       cfg["data"],
        "--epochs",     str(cfg["epochs"]),
        "--batch-size", str(cfg["batch"]),
        "--imgsz",      str(cfg["img"]),
        "--device",     cfg["device"],
        "--workers",    str(cfg["workers"]),
        "--name",       cfg["name"],
        "--lr0",        str(cfg["lr0"]),
        "--label-smoothing", str(cfg.get("label_smoothing", 0.1)),
        "--exist-ok",
    ]
    
    if cfg.get("nc_major"):
        cmd.extend(["--nc-major", str(cfg["nc_major"])])
        cmd.extend(["--major-weight", str(cfg.get("major_weight", 0.5))])
    
    # 验证频率 (Phase 4.1 新增)
    if cfg.get("val_period"):
        cmd.extend(["--val-period", str(cfg["val_period"])])
    
    if last_ckpt.exists():
        print(f"🔄 发现之前的检查点 {last_ckpt}, 已启用 --resume 断点续传...")
        cmd.append("--resume")
        
    return cmd


def print_banner(cfg: dict):
    print("=" * 60)
    print("  🗑️  garbage265 垃圾分类 — YOLOv5 训练启动")
    print("=" * 60)
    print(f"  模型    : {cfg['model']}")
    print(f"  图片尺寸: {cfg['img']}x{cfg['img']}")
    print(f"  Batch   : {cfg['batch']}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  设备    : GPU {cfg['device']}")
    print(f"  结果目录: yolov5/runs/train-cls/{cfg['name']}/")
    print("=" * 60)

    # 检查 GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU     : {name}  ({mem:.0f} GB)")
        else:
            print("  ⚠️  未检测到 CUDA GPU，将使用 CPU（速度很慢）")
    except ImportError:
        print("  ⚠️  torch 未安装，请先运行 bash setup.sh")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="garbage265 YOLOv5 分类训练")
    parser.add_argument("--model",   default=CONFIG["model"],   help="模型权重")
    parser.add_argument("--epochs",  type=int, default=CONFIG["epochs"], help="训练轮数")
    parser.add_argument("--batch",   type=int, default=CONFIG["batch"],  help="Batch size")
    parser.add_argument("--img",     type=int, default=CONFIG["img"],    help="输入图片尺寸")
    parser.add_argument("--device",  default=CONFIG["device"],  help="GPU 设备号，如 0 或 0,1")
    parser.add_argument("--workers", type=int, default=CONFIG["workers"], help="DataLoader 进程数")
    parser.add_argument("--name",    default=CONFIG["name"],    help="实验名称")
    parser.add_argument("--lr0",     type=float, default=CONFIG["lr0"], help="初始学习率")
    parser.add_argument("--label-smoothing", type=float, default=CONFIG["label_smoothing"], help="标签平滑度")
    parser.add_argument("--nc-major", type=int, default=CONFIG["nc_major"], help="大类数量")
    parser.add_argument("--major-weight", type=float, default=CONFIG["major_weight"], help="大类损失权重")
    parser.add_argument("--val-period", type=int, default=CONFIG["val_period"], help="验证频率")
    parser.add_argument("--data",    default=CONFIG["data"],    help="数据集根目录")
    args = parser.parse_args()

    # 用命令行参数覆盖默认配置
    cfg = vars(args)

    print_banner(cfg)

    yolo_path  = check_yolov5()
    data_dir   = Path(cfg["data"])
    check_dataset(data_dir)

    cmd = build_command(yolo_path, cfg)

    print("🚀 执行命令:")
    print("   " + " ".join(cmd))
    print()
    print("📝 训练日志实时输出，按 Ctrl+C 可中断训练")
    print("-" * 60)

    # 启动训练
    result = subprocess.run(cmd, cwd=str(yolo_path.parent))

    if result.returncode == 0:
        best_weights = (
            Path(yolo_path.parent)
            / "yolov5" / "runs" / "train-cls" / cfg["name"] / "weights" / "best.pt"
        )
        print()
        print("=" * 60)
        print("✅ 训练完成！")
        print(f"   最优权重保存在: {best_weights}")
        print(f"   推理命令: python predict.py --img <图片路径>")
        print("=" * 60)
    else:
        print()
        print(f"❌ 训练异常结束（返回码: {result.returncode}）")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
