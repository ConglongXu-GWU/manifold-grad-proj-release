
import torch


def _svd_input_for_device(a: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return the tensor used for SVD and whether CPU routing was used."""
    if a.device.type == "mps":
        return a.cpu(), True
    return a, False


def initial_lr(a: torch.Tensor, k: int):
    """
    Initialize low-rank factors via truncated SVD (PyTorch).

    For MPS inputs, SVD is run explicitly on CPU and the factors are moved
    back to the original device. PyTorch 2.12 does not provide a native MPS
    SVD kernel, and explicit CPU routing avoids the implicit fallback warning.

    Inputs:
      a : torch.Tensor of shape [m, n]
      k : int, rank constraint

    Outputs:
      U : torch.Tensor of shape [m, k]
      x : torch.Tensor of shape [k]
      V : torch.Tensor of shape [n, k]
    """
    # Reduced SVD (economy SVD)
    # torch.linalg.svd returns U, S, Vh directly
    svd_input, _ = _svd_input_for_device(a)
    U, x, Vh = torch.linalg.svd(svd_input, full_matrices=False)

    # Truncate to rank k
    U = U[:, :k]
    x = x[:k]
    V = Vh[:k, :].transpose(0, 1)

    return (
        U.to(device=a.device, dtype=a.dtype),
        x.to(device=a.device, dtype=a.dtype),
        V.to(device=a.device, dtype=a.dtype),
    )


def initialization_method_for_input(a: torch.Tensor) -> str:
    """Return the SVD initialization method label for metadata."""
    _, uses_cpu_for_mps = _svd_input_for_device(a)
    return "svd_cpu_for_mps" if uses_cpu_for_mps else "svd_native"
