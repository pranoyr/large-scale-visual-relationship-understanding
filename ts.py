import torch
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops
import time


# initalize the database
db_dict = {}
results = []


def write(ts_str):
	""" Writes the timestamp to a file.
	"""
	with open("sample.txt", "a") as f:
		f.write(str(ts_str))


def get_ts(frame_no, fps):
	""" Returns the timestamp for a given frame number.
	"""
	sec = frame_no / fps
	return time.strftime('%H:%M:%S', time.gmtime(sec))


def display_ts(predictions, frame_no, fps, th=10):
	for (key, db_dict_of_object) in db_dict.items():
		if key not in predictions.keys():
			db_dict.pop(key)
			continue

		predictions_tensor =  torch.tensor(predictions[key])

		match_quality_matrix = box_ops.box_iou(predictions_tensor.type(torch.float32), db_dict_of_object["box"].type(torch.float32))
		proposal_matcher = det_utils.Matcher(
			0.5,
			0.5,
			allow_low_quality_matches=False)
		matched_idxs_in_image = proposal_matcher(match_quality_matrix)
		clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)


		fill = predictions_tensor[clamped_matched_idxs_in_image]
		count = db_dict_of_object["count"]
		mask = fill == 0
		mask = mask[:, 0]
		db_tensor = fill[~mask]
		db_count = count[~mask]

		diff = predictions_tensor[~predictions_tensor.unsqueeze(1).eq(db_tensor).all(-1).any(-1)][1:]
		db_tensor = torch.cat((db_tensor, diff))
		db_count = torch.cat((db_count, torch.zeros(len(diff))))

		db_count += 1
		count_mask = db_count == th

		db_dict[key]["count"] = db_count
		db_dict[key]["box"] = db_tensor
		
		if count_mask.any():
			results.append((get_ts(frame_no, fps), key))
			write((get_ts(frame_no, fps), key))

	# update the database
	for (key, box) in predictions.items():
		if key not in db_dict.keys():
			db_dict[key] = {"box": [], "count": []}
			db_dict[key]["box"] = torch.tensor(box)
			db_dict[key]["count"] = torch.tensor([0]*len(box))
