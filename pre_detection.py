from ultralytics import YOLO
from PIL import Image
import cv2 as cv

class DetModel:
    def __init__(self):
        self.model=YOLO('detection_best_final.pt')  # load a custom model
        

    def predict(self,img_path):
        results = self.model(source=img_path,show=False,conf=0.6,save=False)  # predict on an image
        
        for r in results:
            self.pred_xyxy=r.boxes.xyxy.tolist()
            print('this are the prediction bboxes:',self.pred_xyxy)
            #print('this are the classes:', [int(cls) for cls in r.boxes.cls.tolist()])
            
            #print('this are the keypoints:',r.keypoints)
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im_resize=cv.resize(im_array,(600,600))
            im_rgb=cv.cvtColor(im_resize,cv.COLOR_BGR2RGB)
            cv.imshow('Predicted',im_rgb)

        
            

#To plot from here
#obj=DetModel()
#obj.predict()

