"""
Utility functions for Google Play Store NLP Analytics.
Includes reproducibility helper, device auto-detection, model persistence, and time formatting.
"""

import os
import random
import time
import numpy as np
import joblib

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, and torch (including CUDA if available).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device():
    """
    Auto-detect CUDA availability and log device details, total VRAM, precision capabilities,
    and recommended execution strategy.
    
    Returns:
        torch.device or str: Detected torch device.
    """
    if torch is not None and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        fp16_supported = True

        print(f"[GPU Mode Activated]")
        print(f"   Device         : {device} ({gpu_name})")
        print(f"   Total VRAM     : {total_mem_gb:.2f} GB")
        print(f"   Precision      : {'BF16' if bf16_supported else ('FP16' if fp16_supported else 'FP32')}")
        print(f"   Execution Strategy: Full dataset (37K samples)")
    else:
        device = torch.device("cpu") if torch is not None else "cpu"
        print(f"[CPU Mode Activated]")
        print(f"   Device         : {device}")
        print(f"   Precision      : FP32")
        print(f"   Execution Strategy: Stratified subset (5K samples by default)")

    return device


def save_model(model, path: str, model_type: str = "sklearn"):
    """
    Save trained model or vectorizer to path.
    
    Args:
        model: Trained model object or transformer model/tokenizer.
        path (str): Destination file path or directory.
        model_type (str): 'sklearn' or 'transformer'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_type == "transformer":
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
        else:
            raise AttributeError("Transformer model/tokenizer object does not have 'save_pretrained' method.")
    else:
        joblib.dump(model, path)
    print(f"Model saved successfully to '{path}'.")


def load_model(path: str, model_type: str = "sklearn", transformer_cls=None):
    """
    Load a trained model or vectorizer from path.
    
    Args:
        path (str): File path or directory to load from.
        model_type (str): 'sklearn' or 'transformer'.
        transformer_cls: Class with `from_pretrained` (required if model_type == 'transformer').
    
    Returns:
        Loaded model object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path '{path}' does not exist.")

    if model_type == "transformer":
        if transformer_cls is None:
            raise ValueError("transformer_cls must be specified when loading a transformer model.")
        return transformer_cls.from_pretrained(path)
    else:
        return joblib.load(path)


def format_time(seconds: float) -> str:
    """
    Convert time in seconds to human-readable string (hh:mm:ss or mm:ss).
    """
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"
