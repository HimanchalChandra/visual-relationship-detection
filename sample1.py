from utils.util import calc_iou


obj_bbox_gt = 1, 2, 3, 4


print(obj_bbox_gt)

iou = calc_iou([1,2,3,4],obj_bbox_gt)
print(iou)