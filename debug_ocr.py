import importlib
import new
importlib.reload(new)
import cv2
img = cv2.imread('outputs/plate_crop_i.jpg')
print('img is None?', img is None)
best, cands = new.recognize_text(img)
print('best:', best)
print('candidates:', cands)
