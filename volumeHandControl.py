import mediapipe as mp
import handTracking_module as htm
import cv2
import numpy as np
import math
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#-------------------------------------------------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
#print(volume_range)
min_vol = volume_range[0]
max_vol = volume_range[1]
#-------------------------------------------------------------------

cap = cv2.VideoCapture(0)
wCam,hCam = 1280,720
cap.set(3,wCam)
cap.set(4,hCam)
prev_time = 0

detector = htm.handDetector(detectionCon=0.7)
while True:
    success,img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img,drawing=False)

    if len(lm_list)!=0:
        x1,y1 = lm_list[4][1],lm_list[4][2]
        x2,y2 = lm_list[8][1],lm_list[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y2)
        print(length)
        vol = np.interp(length,[2,68],[min_vol,max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

    curr_time = time.time()
    fps = abs(1/(curr_time-prev_time))
    prev_time = curr_time
    cv2.putText(img,f'FPS: {str(round(fps))}',(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

