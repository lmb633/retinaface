from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils import cfg_mnet
from utils import PriorBox, py_cpu_nms, decode, decode_landm
import cv2
from models import RetinaFace
import time

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

torch.set_grad_enabled(False)
cfg = cfg_mnet
net = RetinaFace(cfg=cfg, phase='test')
net.load_state_dict(torch.load('weight/mobilenet.pt', map_location=lambda storage, loc: storage), strict=False)
net.eval()
print('Finished loading model!')
cudnn.benchmark = True
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
net = net.to(device)


def detect(image_path):
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = [list(det) for det in dets if det[4] > args.vis_thres]
    print(dets, landms)
    # dets = np.concatenate((dets, landms), axis=1)
    # image_show(img_raw, dets)
    return dets, landms


def image_show(img_raw, dets):
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image
        cv2.imshow('name', img_raw)
        cv2.waitKey(0)


if __name__ == '__main__':
    img_path = 'data/test.jpg'
    start = time.time()
    detect(img_path)
    time
