import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from tqdm import tqdm
import sys
import os
import cv2
from torch.utils.data import DataLoader

from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict
import torchvision.transforms as transforms

from utils.metrics import ConfusionMatrix, ap_per_class
from utils.metrics import box_iou

import pandas


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

## test dataloader ## 
class MyYoloDatasets():
    def __init__(self):
        self.root = "/data1/datasets/great-barrier-reef/"
        self.img_path = "/data1/datasets/great-barrier-reef/val_split1.txt"
        self.imgs = []
        self.labels = []
        self.transform = transforms.ToTensor()      
        self.setup()
    def setup(self):
        with open(self.img_path,'r') as f: 
            while True:
                rel_path = f.readline().strip("\n") 
                # if len(self.imgs) == 100: 
                #     break
                # print(rel_path)
                if rel_path: 
                    self.imgs.append(rel_path)
                    self.labels.append("labels/{}.txt".format(rel_path[9:].split(".")[0]))
                    # print(self.labels[-1])
                else: 
                    break
    def __len__(self):
        return len(self.imgs)                                
    def __getitem__(self, index): 
        img = cv2.imread(os.path.join(self.root, self.imgs[index]))[:, :, ::-1]
        # print(os.path.join(self.root, self.imgs[index]))
        # print(self.imgs[index])
        label = []
        with open(os.path.join(self.root, self.labels[index]), 'r') as f: 
            while True: 
                line = f.readline().strip("\n")
                if line:
                    line = map(float,line.split(" ")) 
                    box = list(line)
                    cls = box[0]
                    xmin = (box[1] - box[3]/2) * 1280
                    ymin = (box[2] - box[4]/2) * 720
                    xmax = (box[1] + box[3]/2) * 1280
                    ymax = (box[2] + box[4]/2) * 720
                    img_l = [cls, xmin, ymin, xmax, ymax] ## to xyxy
                    label.append(img_l)
                else: 
                    break
        # print(label)
        return img, label  

# def f2_score(predict, target):
    
if __name__ == '__main__':
    model = torch.hub.load('./', 
                       'custom', 
                       path='runs/train/5s_split1_imgsize=2560/weights/best.pt',
                       source='local',
                    #    device = 'cuda',
                       force_reload=True)  # local repo
    model.conf_thres = 0.05
    model.iou_thres = 0.45
    print("load ckpt ... ")
            
    iter_test = MyYoloDatasets() 
    print("init dataset ...")        
    # iter_test = DataLoader(test_dataset, batch_size=1, shuffle=False)


    ## predict ##
    stats = []
    iouv = torch.linspace(0.3, 0.8, 10)
    niou = iouv.numel()
    for (image, labels) in tqdm(iter_test):
        # print(image.shape)
        r = model(image, size=3200)
        # print(r.pandas().xyxy[0])    
        detects = []
        for idx, row in r.pandas().xyxy[0].iterrows():            
            # anno += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))
            # print(row.confidence)
            detects.append([row.xmin, row.ymin, row.xmax, row.ymax, row.confidence, 0])
        
        # print("labels: ", labels)
        # print("detects: ", detects)
        labels = torch.Tensor(labels)

        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []
        
        if len(detects) == 0: 
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        
        detects = torch.Tensor(detects)
        predn = detects.clone()
        
        if nl:
            tbox = labels[:, 1:5]
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            correct = process_batch(predn, labelsn, iouv)
        else: 
            correct = torch.zeros(len(detects), niou, dtype=torch.bool)
        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))
        # print("detects:",detects) 

    # compute metrics 
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    print(stats[0].shape, stats[1].shape, stats[2].shape, stats[3].shape)
    # if len(stats) and stats[0].any():
    tp, fp, p, r, f2, ap, ap_class = ap_per_class(*stats, names={0:'starfish'})
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # print(f2.shape)
    # f2 = f2.mean()
    nt = np.bincount(stats[3].astype(np.int64))  # number of targets per class
    # else:
    #     nt = torch.zeros(1)
    print(mp, mr)    
    print(f2.mean())
    import matplotlib.pyplot as plt
    # print(f2[0])
    plt.plot(f2[0])
    plt.savefig("../sahi-val/hub-f2-scores.jpg")    
    