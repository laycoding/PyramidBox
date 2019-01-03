import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

from PIL import Image, ImageDraw
from pyramid import build_sfd
from layers import *
import cv2
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.cuda.set_device(0)


print('Loading model..')
ssd_net = build_sfd('test', 640, 2)
net = ssd_net
net.load_state_dict(torch.load('./models/Res50_pyramid.pth'))
net.cuda()
net.eval()
print('Finished loading model!')


def detect_face(image, shrink):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)


    print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123],dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    net.priorbox = PriorBoxLayer(width,height)
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes=[]
    scores = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.01:
            score = detections[0,i,j,0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            boxes.append([pt[0],pt[1],pt[2],pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0] / shrink
    det_ymin = boxes[:,1] / shrink
    det_xmax = boxes[:,2] / shrink
    det_ymax = boxes[:,3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def write_to_txt(f, det):
    f.write('{:s}\n'.format(str(event[0][0].encode('utf-8'))[2:-1] + '/' + im_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


if __name__ == '__main__':
    subset = 'val' # val or test
    if subset is 'val':
        wider_face = sio.loadmat('../data/wider_face_val.mat')    # Val set
    else:
        wider_face = sio.loadmat('../data/wider_face_test.mat')   # Test set
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    Path = '/home/face/zhangweidong/Torch_Facedet/data/WIDER_val/images/'
    save_path = 'test_folder/results' + '_' + subset + '/'


    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        if not os.path.exists(save_path + str(event[0][0].encode('utf-8'))[2:-1] ):
            os.makedirs(save_path + str(event[0][0].encode('utf-8'))[2:-1] )
        for num, file in enumerate(filelist):
            
            im_name = str(file[0][0].encode('utf-8'))[2:-1] 
            Image_Path = Path + str(event[0][0].encode('utf-8'))[2:-1] +'/'+im_name[:] + '.jpg'
            print(Image_Path)
            image = cv2.imread(Image_Path,cv2.IMREAD_COLOR)

            max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
            max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            
            shrink = max_im_shrink if max_im_shrink < 1 else 1

            dets = detect_face(image, 1)  # origin test

            f = open(save_path + str(event[0][0].encode('utf-8'))[2:-1]  + '/' + im_name + '.txt', 'w')
            write_to_txt(f, dets)

            print('event:%d num:%d' % (index + 1, num + 1))

