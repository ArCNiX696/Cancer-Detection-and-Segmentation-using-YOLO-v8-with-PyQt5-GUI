from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import numpy as np

class SegmModel:
    def __init__(self):
        self.model=YOLO('best_segmen_final.pt')  # load a custom model
       

    def predict(self,img_path):
        results = self.model(source=img_path,show=False,conf=0.5,save=False)  # predict on an image
        
        for r in results:
            if r.masks is not None and r.masks.xy is not None:
                self.pred_xyxy = r.masks.xy
                #print('this are the masks:',self.pred_xyxy)
                
            
            else:
                self.pred_xyxy = np.array([[[0, 0], [0, 0]]], dtype=np.float32)
                #print('this are the masks:', self.pred_xyxy)
                
            #print('this are the masks:',self.pred_xyxy)
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im_resize=cv.resize(im_array,(600,600))
            im_rgb=cv.cvtColor(im_resize,cv.COLOR_BGR2RGB)
            cv.imshow('Predicted',im_rgb)
            
#To plot from here
#obj=DetModel()
#obj.predict()
