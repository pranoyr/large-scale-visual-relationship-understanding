import torch
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops
import time
import cv2


# initalize the database
db_dict = {}
results = []


def set_text(draw, results):
	x, y = 10, 20
	for (timestamp, label) in results:
		font = cv2.FONT_HERSHEY_SIMPLEX
		lineThickness = 1
		font_size = 0.5
		# set some text
		# get the width and height of the text box
		(text_width, text_height) = cv2.getTextSize(f"{label}, time: {timestamp}", font, font_size, lineThickness)[0]
		
		text_offset_x, text_offset_y = int(x), int(y)
		# make the coords of the box with a small padding of two pixels
		box_coords = ((text_offset_x, text_offset_y), (text_offset_x +text_width + 2, text_offset_y - text_height - 10))
		cv2.rectangle(draw, box_coords[0], box_coords[1], (255, 0, 0), cv2.FILLED)
		cv2.putText(draw, f"{label}, time: {timestamp}", (text_offset_x, text_offset_y-5), font,font_size, (255, 255, 255), lineThickness, cv2.LINE_AA)
		# x+=10
		y+=30

def write(ts_str):
	""" Writes the timestamp to a file.
	"""
	with open("sample.txt", "a") as f:
		f.write(str(ts_str))
		f.write("\n")


def get_ts(frame_no, fps):
	""" Returns the timestamp for a given frame number.
	"""
	sec = frame_no / fps
	return time.strftime('%H:%M:%S', time.gmtime(sec))


def display_ts(draw, predictions, frame_no, fps, th=10):
	for (key, db_dict_of_object) in db_dict.copy().items():
		if key not in predictions.keys():
			if key not in ["aeroplane", "catering truck arrived", "catering truck attached", "bridge connected", "wheel chocks"]:
    				db_dict.pop(key)
			continue
	
		predictions_tensor = torch.cat([torch.tensor([[0,0,0,0]]), torch.tensor(predictions[key])])

		match_quality_matrix = box_ops.box_iou(predictions_tensor.type(torch.float32), db_dict_of_object["box"].type(torch.float32))
  
		clamped_matched_idxs_in_image = []
		for i in range(match_quality_matrix.shape[1]):
			row_idx = match_quality_matrix[:,i].max(dim=0)[1]
			col_idx = match_quality_matrix[row_idx,:].max(dim=0)[1] 
			if col_idx == i:
				clamped_matched_idxs_in_image.append(row_idx)
			else:
				clamped_matched_idxs_in_image.append(0)
		clamped_matched_idxs_in_image = torch.tensor(clamped_matched_idxs_in_image)

		# match_quality_matrix = box_ops.box_iou(predictions_tensor.type(torch.float32), db_dict_of_object["box"].type(torch.float32))
		# proposal_matcher = det_utils.Matcher(
		# 	0.5,
		# 	0.5,
		# 	allow_low_quality_matches=False)
		# matched_idxs_in_image = proposal_matcher(match_quality_matrix)
		# clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

		fill = predictions_tensor[clamped_matched_idxs_in_image]
		count = db_dict_of_object["count"]
		mask = fill == 0
		mask = mask[:, 0]
		db_tensor = fill[~mask]
		db_count = count[~mask]

		# diff = db_tensor - predictions
		diff = predictions_tensor[~predictions_tensor.unsqueeze(1).eq(db_tensor).all(-1).any(-1)][1:]
		db_tensor = torch.cat((db_tensor, diff))
		db_count = torch.cat((db_count, torch.zeros(len(diff))))

		db_count += 1
		count_mask = db_count == th

		# update the database
		db_dict[key]["count"] = db_count
		db_dict[key]["box"] = db_tensor
		
		if count_mask.any():
			results.append((get_ts(frame_no, fps), key)) # resutls = [(timestamp, "arrived")]
			# set_text(draw, results)
			write((get_ts(frame_no, fps), key))
	
		set_text(draw, results)

	
	# update the database
	for (key, box) in predictions.items():
		if key not in db_dict.keys():
			db_dict[key] = {"box": [], "count": []}
			db_dict[key]["box"] = torch.tensor(box)
			db_dict[key]["count"] = torch.tensor([0]*len(box))


	print("results")
	print(results)

	print("db_dict")
	print(db_dict)

	print("predictions")
	print(predictions)


	print()
	print()
	print()
