import torch


def select_device(priority=["cuda", "mps", "cpu"]):
    """
    Selects the device based on the given priority list.

    Parameters:
        - priority (list): List of strings representing device priorities.

    Returns:
        - torch.device: Device selected based on priority.
    """

    if "cuda" in priority and torch.cuda.is_available():
        return torch.device("cuda")
    if "mps" in priority and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if "cpu" in priority:
        return torch.device("cpu")

    raise ValueError("No valid device found in priority list.")
