import cv2
import numpy as np

def estimate_impact_source(damage_box, image_height):
    x1,y1,x2,y2 = damage_box
    height = (y1+y2)/2

    relative_height = height / image_height

    if relative_height > 0.6:
        return "Possible Animal Collision"
    elif relative_height > 0.4:
        return "Possible Human Impact"
    else:
        return "Likely Vehicle Collision"


def detect_scratch_direction(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLines(edges,1,np.pi/180,120)

    if lines is None:
        return "Unknown"

    angles=[]

    for line in lines[:20]:
        rho,theta = line[0]
        angles.append(theta)

    mean_angle = np.mean(angles)

    degree = np.degrees(mean_angle)

    if degree < 20 or degree > 160:
        return "Side Swipe Impact"
    elif 70 < degree < 110:
        return "Frontal Collision"
    else:
        return "Angled Collision"


def severity_score(damages, image_shape):

    img_area = image_shape[0]*image_shape[1]
    total_damage=0

    for d in damages:
        x1,y1,x2,y2 = d["bbox"]
        area=(x2-x1)*(y2-y1)
        total_damage+=area

    ratio = total_damage/img_area

    score = min(int(ratio*500),100)

    return score