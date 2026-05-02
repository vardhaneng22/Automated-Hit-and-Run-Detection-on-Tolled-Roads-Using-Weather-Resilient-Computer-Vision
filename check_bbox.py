import cv2
import numpy as np
import os, sys

out = 'outputs/repo_test_out_1.jpg'
if not os.path.exists(out):
    print('Output file not found:', out)
    sys.exit(1)
img = cv2.imread(out)
print('Saved image shape:', img.shape)
# Count pure green pixels (BGR)
green_mask = (img[:,:,0]==0) & (img[:,:,1]==255) & (img[:,:,2]==0)
print('Green pixel count:', int(green_mask.sum()))
# Also load detector and print plates
sys.path.insert(0, os.path.join(os.getcwd(), 'ANPR_repo'))
from v8 import ANPR_V8
model = ANPR_V8(os.path.join('ANPR_repo','models','anpr_v8.pt'))
orig = cv2.imread('i3.jpg')
plates, _ = model.detect(orig, threshold=0.3)
print('Detected plates from model on i3.jpg:')
for p in plates:
    print(p)

# show bounding box extents
for p in plates[:5]:
    try:
        x1,y1,x2,y2,conf = p[:5]
        print('box:', int(x1),int(y1),int(x2),int(y2),'conf',conf)
    except:
        print('unexpected plate format', p)
