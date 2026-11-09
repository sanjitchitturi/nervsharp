import platform
import os


def detect_hardware() -> dict:
    info = {
        "device_name": "cpu",
        "has_gpu": False,
        "has_npu": False,
        "cpu_count": os.cpu_count() or 4,
        "arch": platform.machine(),
        "os": platform.system(),
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["has_gpu"] = True
            info["device_name"] = "Apple MPS"
    except ImportError:
        pass

    if info["device_name"] == "cpu":
        try:
            import cpuinfo
            info["device_name"] = cpuinfo.get_cpu_info().get("brand_raw", "cpu")
        except ImportError:
            info["device_name"] = platform.processor() or "cpu"

    return info

