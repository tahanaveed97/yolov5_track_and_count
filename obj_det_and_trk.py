
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import colors
from utils.torch_utils import select_device, time_sync
from sort import *


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def detect():
    weights = 'yolov5n.pt'
    imgsz=(640, 640)
    iou_thres=0.45  
    max_det=1000
    device='cpu'  
    conf_thres = 0.25
    half = False

    memory = {}

    #.... Initialize SORT .... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
    if pt or jit:
        model.model.half() if half else model.model.float()

    cap = cv2.VideoCapture('2.mp4')

    line_position = 240

    roi_line = [(0, line_position), (640, line_position)]
    line = [roi_line[0], roi_line[1]]
    intersect_val = int(line_position)

    in_count = 0
    out_count = 0


    while True:

        ret, im0 = cap.read()
        if im0 is None:
            cap = cv2.VideoCapture('2.mp4')
            continue
        im0 = cv2.resize(im0, (640,640))
        im = torch.from_numpy(im0.transpose(2,0,1)).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, [0], False, max_det=max_det)

        previous = memory.copy()
        boxes, indexIDs, memory = [], [], {}

        for i, det in enumerate(pred):

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                dets_to_sort = np.empty((0,6))

                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)

                for track in tracked_dets:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[8]))
                    memory[indexIDs[-1]] = boxes[-1]

                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
                        cv2.rectangle(im0, (x, y), (w, h), (255, 0, 0), 2)
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                            p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                            im0 = cv2.line(im0, p0, p1, (0,0,255), 4)

                            if intersect(p0, p1, line[0], line[1]):
                                if p0[1] > intersect_val:
                                    in_count+=1

                                elif p0[1] < intersect_val:
                                    out_count+=1

                        text = "{}".format(indexIDs[i])
                        cv2.putText(im0, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.line(im0, line[0], line[1], (255, 49, 255), 5)
        cv2.putText(im0, "In Count" + str(in_count), (20, 100), 2, 1, (0, 0, 255), 2)
        cv2.putText(im0, "Out Count" + str(out_count), (20, 150), 2, 1, (0, 0, 255), 2)

        cv2.imshow('img', im0)
        cv2.waitKey(1)


if __name__ == "__main__":
    detect()
