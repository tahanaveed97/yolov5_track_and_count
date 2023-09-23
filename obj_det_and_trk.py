
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



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)
    return img


def detect():
    weights = 'yolov5n.pt'
    imgsz=(640, 640)
    iou_thres=0.45  
    max_det=1000
    device='cpu'  
    conf_thres = 0.25
    half = False

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


        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                dets_to_sort = np.empty((0,6))

                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
                

                #loop over tracks
                for track in tracks:
                    [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                            (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                            (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 
                
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, None)
        cv2.imshow('img', im0)
        cv2.waitKey(1)


if __name__ == "__main__":
    detect()
