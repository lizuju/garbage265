# Dataset Download
import time
import threading
import sys
import logging
from modelscope.msdatasets import MsDataset

# 只显示 WARNING 以上级别的 modelscope 日志，避免刷屏
logging.getLogger('modelscope').setLevel(logging.WARNING)

# --- 等待动画 ---
_stop_spinner = False

def spinner(msg):
    frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    start = time.time()
    while not _stop_spinner:
        elapsed = time.time() - start
        print(f'\r{frames[i % len(frames)]}  {msg}  (已等待 {elapsed:.0f}s)', end='', flush=True)
        time.sleep(0.1)
        i += 1
    print()  # 换行

# --- 开始 ---
print("=" * 55)
print("  开始加载数据集: tany0699/garbage265")
print("=" * 55)

start_time = time.time()

# 启动等待动画线程
t = threading.Thread(target=spinner, args=("正在下载数据集，请耐心等待...",), daemon=True)
t.start()

# 加载数据集（这一步会下载，可能需要几分钟）
ds = MsDataset.load(
    'tany0699/garbage265',
    subset_name='default',
    split='train'
)

# 停止动画
_stop_spinner = True
t.join()

elapsed = time.time() - start_time
print(f"\n✅ 数据集加载完成！共耗时 {elapsed:.1f} 秒")
print(f"📦 训练集大小: {len(ds)} 条样本")
print("\n前 3 条样本预览：")
for i, sample in enumerate(ds):
    print(f"  [{i+1}] {sample}")
    if i >= 2:
        break
print("=" * 55)