import torch
import torch.nn as nn
import torch.nn.functional as F

# def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
# 	from .cuda.chamfer_distance import ChamferDistance
# 	cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
# 	cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
# 	cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
# 	chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
# 	return chamfer_loss

def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
    """
    Compute the Chamfer distance between two point clouds using PyTorch.
    """
    # Compute pairwise distances between points in each point cloud
    cost_p0_p1 = torch.cdist(template, source)
    cost_p1_p0 = torch.cdist(source, template)

    # Compute the nearest neighbor distances for each point in each point cloud
    min_cost_p0_p1, _ = torch.min(cost_p0_p1, dim=1)
    min_cost_p1_p0, _ = torch.min(cost_p1_p0, dim=1)

    # Average the nearest neighbor distances for each point cloud
    chamfer_loss = torch.mean(min_cost_p0_p1) + torch.mean(min_cost_p1_p0)

    return chamfer_loss


class ChamferDistanceLoss(nn.Module):
	def __init__(self):
		super(ChamferDistanceLoss, self).__init__()

	def forward(self, template, source):
		return chamfer_distance(template, source)