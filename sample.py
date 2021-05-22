import json
from utils.boxes import xywh_to_xyxy
# Opening JSON file
f = open('/home/cyberdome/data/vg/relationships.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
print(len(data))

def load_annotations(index):
    for spo in data[index]['relationships']:
        try:
            gt_sbj_label = spo['subject']['name']
        except:
            gt_sbj_label = ''.join(spo['subject']['names'])

        # gt_sbj_bbox = spo['subject']['bbox']
        try:
            gt_obj_label = ''.join(spo['object']['names'][0])
        except:
            gt_obj_label = spo['object']['name']
        # gt_obj_bbox = spo['object']['bbox']
        predicate = spo['predicate']

        print(gt_sbj_label , predicate, gt_obj_label)
        # return(gt_sbj_label , predicate, gt_obj_label)

        # prepare bboxes for subject and object
        # gt_sbj_bbox = y1y2x1x2_to_x1y1x2y2(gt_sbj_bbox)
        # gt_obj_bbox = y1y2x1x2_to_x1y1x2y2(gt_obj_bbox)
        # boxes.append([gt_sbj_bbox, gt_obj_bbox])

        # prepare labels for subject and object
        # map to word
        # gt_sbj_label = self.all_objects[gt_sbj_label]
        # gt_obj_label = self.all_objects[gt_obj_label]
        # predicate = self.predicates[predicate]
        # # map to new index
        # labels.append([self._class_to_ind[gt_sbj_label],
        #                 self._class_to_ind[gt_obj_label]])
        # preds.append(self._preds_to_ind[predicate])


def __getitem__(index):
    img_name = data[index]['image_id']
    load_annotations(index)
    # img_path = self.image_path_from_index(img_name)
    # img = Image.open(img_path)
    # img = self.transform(img)
for i in range(1000):
    __getitem__(i)
	