from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

# # Define FPN
# fpn = resnet_fpn_backbone(
#     backbone_name='resnet18', pretrained=True, trainable_layers=5)


# print(gt_boxes_in_image[:, 0, :].shape)
# print(gt_sbj_boxes_in_image.shape)
# print(gt_boxes_in_image[:, 1, :].shape)
# print(gt_obj_boxes_in_image.shape)

# print(gt_boxes_in_image[:, 0, :])
# print("***")
# print(sbj_boxes

# sbj_matched_idxs_in_image = sbj_matched_idxs_in_image.clamp(
    #     min=0)
    # obj_matched_idxs_in_image = obj_matched_idxs_in_image.clamp(
    #     min=0)

    # a = torch.where(torch.all(x == torch.tensor([[1,2,3]]), dim=1)) 

# sbj_indices = [torch.where(torch.all(gt_boxes_in_image[:, 0, :] == x, dim=1))[0].item() for x in sbj_boxes]  
# obj_indices = [torch.where(torch.all(gt_boxes_in_image[:, 1, :] == x, dim=1))[0].item() for x in obj_boxes] 
# sbj_indices = torch.tensor(sbj_indices)
# obj_indices = torch.tensor(obj_indices)

# mask = sbj_indices == obj_indices
# sbj_indices[~mask]=0
# labels_in_image = gt_preds_in_image[sbj_indices]

# sbj_matched_idxs_in_image[sbj_matched_idxs_in_image !=
#                           obj_matched_idxs_in_image] = -1
# clamped_sbj_matched_idxs_in_image = sbj_matched_idxs_in_image.clamp(
#     min=0)

# labels_in_image = gt_preds_in_image[clamped_sbj_matched_idxs_in_image]
# bg_inds = sbj_matched_idxs_in_image == -1
# labels_in_image[bg_inds] = 0



# x = torch.Tensor(1,3,512,512)
# outputs = fpn(x)

# # print(type(outputs))

# for i in outputs.items():
#     print(i[1].shape)

#     # >>> # returns
#     #     >>>   [('0', torch.Size([1, 256, 16, 16])),
#     #     >>>    ('1', torch.Size([1, 256, 8, 8])),
#     #     >>>    ('2', torch.Size([1, 256, 4, 4])),
#     #     >>>    ('3', torch.Size([1, 256, 2, 2])),
#     #     >>>    ('pool', torch.Size([1, 256, 1, 1]))]

import numpy as np
gt_sbj = torch.tensor([[1,2,3],
                  [1,2,3],
                  [14,2,3]])


gt_obj = torch.tensor([[11,12,13],
                  [14,15,16],
                  [14,15,16]])




sbj_boxes = torch.tensor([[1,2,3], [7,8,9]])
obj_boxes = torch.tensor([[14,15,16], [11,12,13]])


# a = torch.tensor([torch.where(torch.all(gt_sbj == x, dim=1)) for x in sbj_boxes])  
a = torch.where(torch.all(gt_sbj == sbj_boxes[0], dim=1))
b = torch.where(torch.all(gt_obj == obj_boxes[0], dim=1))

# b = torch.tensor([torch.where(torch.all(gt_obj == x, dim=1))[0].item() for x in obj_boxes]) 
print(a)
print(b)

a = np.intersect1d(a[0],b[0])
print(a)

if a.any():
    print(a[0])

# print(b)

# mask = a==b

# print(~mask)
# print(mask.to(dtype=torch.int64))

# sbj = torch.tensor([[1,2,3],
#                   [1,2,3],
#                   [7,8,9]])

# print(torch.unique(sbj, dim=0))

