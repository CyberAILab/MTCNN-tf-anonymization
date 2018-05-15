#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import math
import os
import numpy as np

input_path = "./input/"
output_path = "./output/"

test_mode = "ONet"
thresh = [0.5, 0.5, 0.5]
min_face_size = 12
stride = 2
slide_window = False
scale_factor = 0.79
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 16, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, scale_factor = scale_factor, slide_window=slide_window)
gt_imdb = []
for item in os.listdir(input_path):
    gt_imdb.append(os.path.join(input_path,item))
test_data = TestLoader(sorted(gt_imdb))
all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
count = 0
num_face = 0

face_txt = open(os.path.join(output_path, "face.txt"), 'w')

for imagepath in sorted(gt_imdb):
    print imagepath
    image = cv2.imread(imagepath)
    blur = image.copy()
    mask = np.zeros(image.shape, dtype=np.uint8)
    masked = image.copy()
    for bbox, landmark in zip(all_boxes[count], landmarks[count]):
        left = int(bbox[0]) if int(bbox[0]) > 0 else 0
        right = int(bbox[2]) if int(bbox[2]) < image.shape[1] else image.shape[1] - 1
        up = int(bbox[1]) if int(bbox[1]) > 0 else 0
        down = int(bbox[3]) if int(bbox[3]) < image.shape[0] else image.shape[0] - 1
        line = os.path.basename(imagepath) + ' ' + str(left)+ ' ' + str(right)  + ' ' + str(up)+ ' ' + str(down) + '\n'
        face_txt.write(line)
        blur_size = int(float(right - left) * 0.3) * 2 + 1
        blur[up:down, left:right, :] = cv2.GaussianBlur(blur[up:down, left:right, :], (blur_size, blur_size), 0)

        maxx = 0
        minx = image.shape[1]
        maxy = 0
        miny = image.shape[0]
        for i in range(len(landmark)/2):
            #cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
            maxx = max(landmark[2*i], maxx)
            maxy = max(landmark[2*i+1], maxy)
            minx = min(landmark[2*i], minx)
            miny = min(landmark[2*i+1], miny)
        #Calculate the angle based on the eye position
        deltay = landmark[2*1+1] - landmark[2*0+1]
        deltax = landmark[2*1] - landmark[2*0]
        angle = (math.atan(deltay / deltax) * 180.0) / math.pi
        #Calculate the axes based on the dist between the bbox and the nose position
        shortaxe1 = ((bbox[2] + maxx) / 2) - landmark[2*2]
        shortaxe2 = landmark[2*2] - bbox[0] - (minx - bbox[0]) / 2
        longaxe1 = ((bbox[3] + maxy) / 2) - landmark[2*2+1]
        longaxe2 = landmark[2*2+1] - bbox[1] - (miny - bbox[1]) / 2

        #The center of the ellipse is the average between the nose position and the bbox
        centerx = ((bbox[0] + bbox[2]) / 2 + landmark[2*2]) / 2
        centery = landmark[2*2+1]
        if shortaxe2 + centerx > bbox[2]:
            shortaxe2 = bbox[2] - centerx

        if centerx - shortaxe1 < bbox[0] :
            shortaxe1 = centerx - bbox[0]
        cv2.ellipse(mask, (int(centerx), int(centery)), (int(max(shortaxe1, shortaxe2)), int(max(longaxe1, longaxe2))), angle, 0.0, 360.0, (255, 255, 255), -1)


    selected = np.where(mask == 255)
    masked[selected] = blur[selected]

    for bbox, landmark in zip(all_boxes[count], landmarks[count]):
        cv2.putText(masked,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1] - 2)),cv2.FONT_HERSHEY_TRIPLEX,0.5,color=(0,255,0))
        cv2.rectangle(masked, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0))

    count = count + 1
    cv2.imwrite(os.path.join(output_path, os.path.basename(imagepath)), masked)
    cv2.imshow("masked", masked)
    cv2.waitKey(0)
face_txt.close()
