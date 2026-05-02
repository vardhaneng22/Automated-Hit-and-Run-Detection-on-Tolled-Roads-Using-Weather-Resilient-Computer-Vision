import cv2
import numpy as np
import os, sys
outf='check_bbox_log.txt'
with open(outf,'w') as f:
    out = 'outputs/repo_test_out_1.jpg'
    if not os.path.exists(out):
        f.write('Output file not found: '+out+'\n')
        sys.exit(1)
    img = cv2.imread(out)
    f.write('Saved image shape: '+str(img.shape)+'\n')
    green_mask = (img[:,:,0]==0) & (img[:,:,1]==255) & (img[:,:,2]==0)
    f.write('Green pixel count: '+str(int(green_mask.sum()))+'\n')
    sys.path.insert(0, os.path.join(os.getcwd(), 'ANPR_repo'))
    from v8 import ANPR_V8
    model = ANPR_V8(os.path.join('ANPR_repo','models','anpr_v8.pt'))
    orig = cv2.imread('i3.jpg')
    plates, _ = model.detect(orig, threshold=0.3)
    f.write('Detected plates from model on i3.jpg:\n')
    for p in plates:
        f.write(str(p)+'\n')
    for p in plates[:5]:
        try:
            x1,y1,x2,y2,conf = p[:5]
            f.write('box: %d %d %d %d conf %s\n'%(int(x1),int(y1),int(x2),int(y2),str(conf)))
        except Exception as e:
            f.write('unexpected plate format '+str(p)+' error:'+str(e)+'\n')
print('done')
