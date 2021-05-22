import json

# Opening JSON file
f = open('/home/cyberdome/data/vg/relationships.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

def load_annotations(index):
    for spo in data[index]['relationships']:
        gt_sbj_label = spo['subject']['name']
        # gt_sbj_bbox = spo['subject']['bbox']
        gt_obj_label = spo['object']['names'][0]
        # gt_obj_bbox = spo['object']['bbox']
        predicate = spo['predicate']

        print(gt_sbj_label , predicate, gt_obj_label  )

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
    boxes, labels, preds = load_annotations(index)
    # img_path = self.image_path_from_index(img_name)
    # img = Image.open(img_path)
    # img = self.transform(img)


__getitem__(1)
	


