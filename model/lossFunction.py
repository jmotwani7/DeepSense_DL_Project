import torch
import torchvision


def inverseHuberLoss(out, target):
    """

    Parameters
    ----------
    out
    target

    Returns
    -------

    """
    diff = torch.abs(out - target)
    Coff = 0.2 * torch.max(diff).item()
    return torch.mean(torch.where(diff < Coff, diff, (diff *diff + Coff *Coff) / (2 * Coff)))

