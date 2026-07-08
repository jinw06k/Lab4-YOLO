"""
test_ultralytics.py -- does the ultralytics (PyTorch) YOLO path work on this Pi?

The lab normally runs the model as ONNX via onnxruntime (lean, no torch). This
script checks whether the heavier ultralytics + PyTorch path is viable instead
-- the open question on a 1 GB Pi 5 is whether torch + the model fit in RAM and
run at a usable speed, or whether it OOMs / swaps to a crawl.

It does NOT touch the camera -- it feeds a synthetic frame so the only thing
under test is ultralytics/torch. If this PASSES, an ultralytics version of
partF.py is worth trying. If it FAILS (or the FPS is unusably low), stick with
the ONNX path.

Setup on the Pi (needs internet -- MWireless is fine):
    pip install ultralytics          # pulls in torch; large download
    python test_ultralytics.py       # downloads yolo26n.pt on first run

    python test_ultralytics.py yolo26n.pt   # or point at a specific model file
"""

import sys
import time
import resource                       # Linux: ru_maxrss is peak RSS in kilobytes

import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo26n.pt"
IMGSZ = 320                           # match the lab's export size
N_RUNS = 5                            # a few inferences to get past warmup


def peak_ram_mb():
    # On Linux ru_maxrss is in KB; this is the whole process's peak resident set.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    print("=" * 60)
    print("Testing the ultralytics / PyTorch YOLO path")
    print("=" * 60)

    # ---- 1. Import (torch is the memory-heavy part) ------------------------
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        print(f"[FAIL] could not import ultralytics/torch: {e}")
        print("       -> ultralytics not usable here. Stick with the ONNX path.")
        return 1
    print(f"[ OK ] imported  torch {torch.__version__}")
    print(f"[INFO] peak RAM after imports: {peak_ram_mb():.0f} MB")

    # ---- 2. Load the model (allocates the weights) -------------------------
    try:
        t0 = time.time()
        model = YOLO(MODEL)
        print(f"[ OK ] loaded model '{MODEL}' in {time.time() - t0:.1f}s")
        print(f"[INFO] peak RAM after load:    {peak_ram_mb():.0f} MB")
    except Exception as e:
        print(f"[FAIL] could not load model '{MODEL}': {e}")
        print("       -> likely out of memory. Stick with the ONNX path.")
        return 1

    # ---- 3. Run inference on a synthetic frame -----------------------------
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        times = []
        for i in range(N_RUNS):
            t0 = time.time()
            model.predict(dummy, imgsz=IMGSZ, verbose=False)
            times.append(time.time() - t0)
            print(f"[INFO] inference {i + 1}/{N_RUNS}: {times[-1] * 1000:.0f} ms")
    except Exception as e:
        print(f"[FAIL] inference crashed (likely OOM): {e}")
        print("       -> ultralytics not usable here. Stick with the ONNX path.")
        return 1

    # ---- Verdict -----------------------------------------------------------
    steady = min(times)               # best (non-warmup) time
    fps = 1.0 / steady if steady > 0 else 0.0
    print("-" * 60)
    print(f"[PASS] ultralytics runs here.")
    print(f"       best inference: {steady * 1000:.0f} ms  (~{fps:.2f} FPS)")
    print(f"       peak RAM:       {peak_ram_mb():.0f} MB   (this Pi has ~1024 MB)")
    print("       If peak RAM is near 1024 MB or FPS is much worse than the")
    print("       ONNX path (~1 Hz), prefer ONNX. Otherwise ultralytics is fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
