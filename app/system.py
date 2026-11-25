import psutil
from .pipeline import LIVE

def snapshot():
    vm = psutil.virtual_memory()
    d = {"cpu_percent": psutil.cpu_percent(interval=None),
         "ram_used_gb": round(vm.used / 1e9, 2), "ram_total_gb": round(vm.total / 1e9, 2),
         "processing_fps": LIVE["fps"], "gpu": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u, m, n = pynvml.nvmlDeviceGetUtilizationRates(h), pynvml.nvmlDeviceGetMemoryInfo(h), pynvml.nvmlDeviceGetName(h)
        d["gpu"] = {"name": n.decode() if isinstance(n, bytes) else n, "util_percent": u.gpu,
                    "vram_used_gb": round(m.used / 1e9, 2), "vram_total_gb": round(m.total / 1e9, 2)}
    except Exception:
        pass  # no NVIDIA GPU/driver: gpu stays null, UI shows UNAVAILABLE
    return d
