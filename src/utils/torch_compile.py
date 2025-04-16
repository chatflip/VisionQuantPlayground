import torch


def get_device_compute_capability() -> float | None:
    """GPUのCompute Capabilityを取得する

    Returns:
        float | None: GPUのCompute Capability
    """
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return float(f"{major}.{minor}")
