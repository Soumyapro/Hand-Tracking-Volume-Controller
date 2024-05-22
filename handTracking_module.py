#Hand Tracking Module
import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, 
                                         max_num_hands=self.maxHands, 
                                         min_detection_confidence=self.detectionCon, 
                                         min_tracking_confidence=self.trackCon)
        self.draw = mp.solutions.drawing_utils

    def find_hands(self,img,drawing=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                 if drawing:
                    self.draw.draw_landmarks(
                        img, hand, self.mp_hands.HAND_CONNECTIONS,
                        self.draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        self.draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )

        return img

    def find_position(self,img,hands_no=0,drawing=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hands_no]

            for id,lm in enumerate(my_hand.landmark):
                h,w,c = img.shape
                cx,cy = int((lm.x)*w),int((lm.y)*h)
                lm_list.append([id,cx,cy])

                if drawing and id==0:
                    cv2.circle(img,(cx,cy),15,(255,255,255),cv2.FILLED)
        
        return lm_list
    
def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    curr_time = 0
    detector = handDetector()

    while True:
        success,img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list)!=0:
            print(lm_list[4])

        curr_time = time.time()
        fps = abs(1/(curr_time-prev_time))
        prev_time = curr_time

        cv2.putText(img,str(round(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
