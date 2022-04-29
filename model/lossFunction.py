import torch
import torchvision


def inverseHuberLoss(target, out):
    diff = torch.abs(out - target)
    Coff = 0.2 * torch.max(diff).item()
    return torch.mean(torch.where(diff < Coff, diff, (diff ^ 2 + Coff ^ 2) / (2 * Coff)))

