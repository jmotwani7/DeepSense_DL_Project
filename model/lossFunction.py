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
    # diff = torch.abs(out - target)
    # Coff = 0.2 * torch.max(diff).item()
    # return torch.mean(torch.where(diff < Coff, diff, (diff *diff + Coff ^ 2) / (2 * Coff)))
    absdiff = torch.abs(out - target)
    C = 0.2 * torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff, (absdiff * absdiff + C * C) / (2 * C)))
