import torch
import torch.nn.functional as F
import torchvision.models.detection._utils as  det_utils


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
	"""
	Computes the loss for Faster R-CNN.

	Arguments:
		class_logits (Tensor)
		box_regression (Tensor)
		labels (list[BoxList])
		regression_targets (Tensor)

	Returns:
		classification_loss (Tensor)
		box_loss (Tensor)
	"""

	labels = torch.cat(labels, dim=0)
	regression_targets = torch.cat(regression_targets, dim=0)

	classification_loss = F.cross_entropy(class_logits, labels)

	# get indices that correspond to the regression targets for
	# the corresponding ground truth labels, to be used with
	# advanced indexing
	sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
	labels_pos = labels[sampled_pos_inds_subset]
	N, num_classes = class_logits.shape
	box_regression = box_regression.reshape(N, -1, 4)

	box_loss = det_utils.smooth_l1_loss(
		box_regression[sampled_pos_inds_subset, labels_pos],
		regression_targets[sampled_pos_inds_subset],
		beta=1 / 9,
		size_average=False,
	)
	box_loss = box_loss / labels.numel()

	return classification_loss, box_loss


def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False):
    device = prd_cls_scores.device
    prd_labels = torch.cat(prd_labels_int32, 0).to(device)
    # prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).to(device)
    loss_cls_prd = F.cross_entropy(prd_cls_scores, prd_labels)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd

