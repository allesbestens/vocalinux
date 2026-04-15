"""
Whisper.cpp model information and hardware detection for Vocalinux.

This module provides model metadata and hardware acceleration detection
for whisper.cpp, supporting Vulkan, CUDA, and CPU backends.
"""

import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Whisper.cpp model information
# Models are downloaded from Hugging Face (ggml format)
WHISPERCPP_MODEL_INFO = {
    "tiny": {
        "size_mb": 39,
        "params": "39M",
        "desc": "Fastest, lowest accuracy",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    },
    "base": {
        "size_mb": 74,
        "params": "74M",
        "desc": "Fast, good for basic use",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    "small": {
        "size_mb": 244,
        "params": "244M",
        "desc": "Balanced speed/accuracy",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    },
    "medium": {
        "size_mb": 769,
        "params": "769M",
        "desc": "High accuracy, slower",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
    "large": {
        "size_mb": 1550,
        "params": "1550M",
        "desc": "Highest accuracy, slowest",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    },
}

# Available models list
AVAILABLE_MODELS = list(WHISPERCPP_MODEL_INFO.keys())


# Compute backend types
class ComputeBackend:
    """Compute backend options for whisper.cpp."""

    VULKAN = "vulkan"
    CUDA = "cuda"
    CPU = "cpu"


def _normalize_gpu_name(name: str) -> str:
    """Normalize GPU names for loose matching across backends."""
    return re.sub(r"\s+", " ", name.strip()).casefold()


def list_vulkan_devices() -> list[tuple[int, str]]:
    """
    Enumerate Vulkan devices in reported order.

    Returns:
        List of ``(index, device_name)`` tuples.
    """
    devices: list[tuple[int, str]] = []

    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return devices

        for line in result.stdout.splitlines():
            match = re.search(r"deviceName\s*[:=]\s*(.+)$", line)
            if not match:
                continue

            device_name = match.group(1).strip()
            if device_name:
                devices.append((len(devices), device_name))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Vulkan device enumeration failed: {e}")

    return devices


def list_cuda_devices() -> list[tuple[int, str, Optional[int]]]:
    """
    Enumerate CUDA devices with approximate VRAM in MiB.

    Returns:
        List of ``(index, device_name, memory_mib)`` tuples.
    """
    devices: list[tuple[int, str, Optional[int]]] = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return devices

        for line in result.stdout.splitlines():
            raw_line = line.strip()
            if not raw_line:
                continue

            parts = [part.strip() for part in raw_line.split(",", 2)]
            if len(parts) < 2:
                continue

            try:
                device_index = int(parts[0])
            except ValueError:
                continue

            memory_mib = None
            if len(parts) > 2:
                try:
                    memory_mib = int(float(parts[2]))
                except ValueError:
                    memory_mib = None

            devices.append((device_index, parts[1], memory_mib))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"CUDA device enumeration failed: {e}")

    return devices


def select_preferred_vulkan_device() -> Optional[tuple[int, str]]:
    """
    Select the most suitable Vulkan device for automatic whisper.cpp use.

    Preference order:
    1. Vulkan device matching the highest-VRAM CUDA device
    2. First enumerated Vulkan device
    """
    vulkan_devices = list_vulkan_devices()
    if not vulkan_devices:
        return None

    cuda_devices = list_cuda_devices()
    if cuda_devices:
        best_cuda = max(cuda_devices, key=lambda item: (item[2] or -1, -item[0]))
        best_cuda_name = _normalize_gpu_name(best_cuda[1])

        exact_matches = [
            (device_index, device_name)
            for device_index, device_name in vulkan_devices
            if _normalize_gpu_name(device_name) == best_cuda_name
        ]
        if exact_matches:
            return exact_matches[0]

        partial_matches = [
            (device_index, device_name)
            for device_index, device_name in vulkan_devices
            if best_cuda_name in _normalize_gpu_name(device_name)
            or _normalize_gpu_name(device_name) in best_cuda_name
        ]
        if partial_matches:
            return partial_matches[0]

    return vulkan_devices[0]


def detect_vulkan_support() -> tuple[bool, Optional[str]]:
    """
    Detect if Vulkan is available and get device info.

    Returns:
        Tuple of (is_available, device_name)
    """
    try:
        preferred_device = select_preferred_vulkan_device()
        if preferred_device:
            _, device_name = preferred_device
            logger.info(f"Vulkan support detected: {device_name}")
            return True, device_name
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Vulkan detection failed: {e}")

    return False, None


def detect_cuda_support() -> tuple[bool, Optional[str]]:
    """
    Detect if NVIDIA CUDA is available and get device info.

    Returns:
        Tuple of (is_available, device_info)
    """
    try:
        cuda_devices = list_cuda_devices()
        if cuda_devices:
            _, gpu_name, memory_mib = max(cuda_devices, key=lambda item: (item[2] or -1, -item[0]))
            gpu_memory = f"{memory_mib} MiB" if memory_mib is not None else "unknown"
            logger.info(f"CUDA support detected: {gpu_name} ({gpu_memory})")
            return True, f"{gpu_name} ({gpu_memory})"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"CUDA detection failed: {e}")

    return False, None


def detect_compute_backend() -> tuple[str, str]:
    """
    Detect the best available compute backend.

    Priority order: Vulkan > CUDA > CPU

    Returns:
        Tuple of (backend_type, backend_info)
    """
    # Try Vulkan first (supports AMD, Intel, NVIDIA)
    has_vulkan, vulkan_info = detect_vulkan_support()
    if has_vulkan and vulkan_info:
        return ComputeBackend.VULKAN, vulkan_info

    # Try CUDA next (NVIDIA only)
    has_cuda, cuda_info = detect_cuda_support()
    if has_cuda and cuda_info:
        return ComputeBackend.CUDA, cuda_info

    # Fall back to CPU
    cpu_info = detect_cpu_info()
    return ComputeBackend.CPU, cpu_info


def detect_cpu_info() -> str:
    """
    Detect CPU information for CPU backend.

    Returns:
        CPU info string
    """
    try:
        # Try to get CPU model from /proc/cpuinfo
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":")[1].strip()
                    return cpu_name
    except Exception as e:
        logger.debug(f"Could not read CPU info: {e}")

    # Fallback to nproc
    try:
        result = subprocess.run(
            ["nproc"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            cpu_count = result.stdout.strip()
            return f"{cpu_count} cores"
    except Exception:
        pass

    return "CPU"


def get_recommended_model() -> tuple[str, str]:
    """
    Get the recommended whisper.cpp model based on system configuration.

    Returns:
        Tuple of (model_name, reason)
    """
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total // (1024**3)

        # Detect available compute backends
        backend, backend_info = detect_compute_backend()

        if backend == ComputeBackend.VULKAN:
            # Vulkan can handle larger models efficiently
            if ram_gb >= 8:
                return "small", f"Vulkan GPU with {ram_gb}GB RAM"
            else:
                return "base", f"Vulkan GPU with {ram_gb}GB RAM"
        elif backend == ComputeBackend.CUDA:
            # CUDA has more VRAM typically
            if "GB" in backend_info:
                try:
                    vram_gb = int(backend_info.split("GB")[0].split("(")[-1].strip())
                    if vram_gb >= 8:
                        return "medium", f"CUDA GPU with {vram_gb}GB VRAM"
                    elif vram_gb >= 4:
                        return "small", f"CUDA GPU with {vram_gb}GB VRAM"
                    else:
                        return "base", f"CUDA GPU with limited VRAM"
                except (ValueError, IndexError):
                    pass
            return "small", f"CUDA GPU detected"
        else:
            # CPU-only recommendations based on RAM
            if ram_gb >= 16:
                return "base", f"{ram_gb}GB RAM - CPU inference"
            elif ram_gb >= 8:
                return "tiny", f"{ram_gb}GB RAM - optimized for speed"
            else:
                return "tiny", f"Limited RAM ({ram_gb}GB) - fastest model"

    except ImportError:
        logger.debug("psutil not available for system detection")

    # Default recommendation
    return "tiny", "Default recommendation"


def get_model_path(model_name: str) -> str:
    """
    Get the path where a model should be stored.

    Args:
        model_name: Name of the model (tiny, base, small, medium, large)

    Returns:
        Path to the model file
    """
    models_dir = os.path.expanduser("~/.local/share/vocalinux/models/whispercpp")
    os.makedirs(models_dir, exist_ok=True)

    if model_name == "large":
        # Large model uses v3 variant
        return os.path.join(models_dir, "ggml-large-v3.bin")
    else:
        return os.path.join(models_dir, f"ggml-{model_name}.bin")


def is_model_downloaded(model_name: str) -> bool:
    """
    Check if a whisper.cpp model is downloaded.

    Args:
        model_name: Name of the model

    Returns:
        True if model exists, False otherwise
    """
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)


def get_backend_display_name(backend: str) -> str:
    """
    Get a user-friendly display name for a compute backend.

    Args:
        backend: Backend type (vulkan, cuda, cpu)

    Returns:
        Display name string
    """
    names = {
        ComputeBackend.VULKAN: "Vulkan GPU",
        ComputeBackend.CUDA: "NVIDIA CUDA",
        ComputeBackend.CPU: "CPU",
    }
    return names.get(backend, backend.upper())
