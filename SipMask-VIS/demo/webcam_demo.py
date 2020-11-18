import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse

import cv2
import torch
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', 0))

    camera = cv2.VideoCapture(0)

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret, img = camera.read()
        cv2.flip(img, 1, img)
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        dets = []
        for i in range(len(result)):
            dets.append(result[i])
        for i in dets[0]:
            bbox = dets[0][i]['bbox']
            obj_class = model.CLASSES[dets[0][i]['label']]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 1)
            image = cv2.putText(img, obj_class, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA) 
        cv2.imshow('Camera frame', img)
        '''show_result(
            img, np.array(dets), model.CLASSES, score_thr=args.score_thr, wait_time=1, show=True)'''


if __name__ == '__main__':
    main()
