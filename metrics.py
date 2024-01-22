import cv2
import numpy as np
import os 
import glob
import os.path as osp
import labelme
from pre_detection import *
from pre_segmentation import *
import matplotlib.pyplot as plt
'''
yolov8_coordinates=[np.array([[        397,         224],
       [        396,         225],
       [        396,         246],
       [        397,         247],
       [        397,         248],
       [        408,         259],
       [        409,         259],
       [        411,         261],
       [        412,         261],
       [        413,         262],
       [        421,         262],
       [        423,         260],
       [        424,         260],
       [        428,         256],
       [        428,         255],
       [        431,         252],
       [        431,         250],
       [        432,         249],
       [        432,         247],
       [        433,         246],
       [        433,         245],
       [        432,         244],
       [        432,         241],
       [        431,         240],
       [        431,         237],
       [        427,         233],
       [        427,         232],
       [        422,         227],
       [        421,         227],
       [        418,         224]], dtype=np.float32), np.array([[         79,         228],
       [         78,         229],
       [         73,         229],
       [         73,         231],
       [         72,         232],
       [         72,         250],
       [         74,         252],
       [         83,         252],
       [         84,         251],
       [         86,         251],
       [         90,         247],
       [         91,         247],
       [         92,         246],
       [         92,         245],
       [         94,         243],
       [         94,         242],
       [         95,         241],
       [         95,         239],
       [         94,         238],
       [         94,         235],
       [         93,         234],
       [         93,         233],
       [         91,         231],
       [         91,         230],
       [         90,         229],
       [         89,         229],
       [         88,         228]], dtype=np.float32)]

'''

class MetricsRaiza:
    def __init__(self):
        #self.detect_pre=DetModel()
        self.seg=SegmModel()
        self.json_dir=os.path.join('./dataset/Sorted Data/','Json_files')
        self.json_listdir=[]
        self.actual_json_idx=0
        

#################### PREPROCESSING ########################    
    #Generate a list with the json files paths
    def json_idx(self):
        self.json_files_listdir = os.listdir(self.json_dir)

        for _, filename in enumerate(self.json_files_listdir):
            file_path = os.path.join(self.json_dir, filename)
            self.json_listdir.append(file_path)
            #print(f"Ruta del archivo {idx + 1}: {file_path}")
        #print('this is self.json_listdir: ',self.json_listdir[2])

              
    #Iterate in the json files
    def json_prev(self):
        try:
            if self.json_listdir:
                self.actual_json_idx = (self.actual_json_idx - 1) % len(self.json_listdir)
                print('this is self.actual_json_idx:',self.actual_json_idx)
                #return self.actual_json_idx
                

        except Exception as e:
            print(f"Error: {e}")

    #Iterate in the json files
    def json_next(self):
        try:
            if self.json_listdir:
                self.actual_json_idx = (self.actual_json_idx + 1) % len(self.json_listdir)
                print('this is self.actual_json_idx:',self.actual_json_idx)
                #return self.actual_json_idx
                

        except Exception as e:
            print(f"Error: {e}")
        
#################### IOU ########################
    #Obtain xyxy coors from the current json file
    def GT_xyxybox(self):

        class_names = ('left normal', 'right normal')

        #print('this is self.actual_json_idx:',self.actual_json_idx)
        filename=self.json_listdir[self.actual_json_idx]
             
        #create and objet of labelme of the filename
        label_file = labelme.LabelFile(filename=filename)

        #store image metadata in json files
        labelme.utils.img_data_to_arr(label_file.imageData)
        
        self.bboxes = []
        labels = []
        
        for shape in label_file.shapes:

            class_name_sp = shape["label"]
            #D
            #print('Esta es la clase de jason :',class_name_sp)
            if class_name_sp not in class_names:
                continue
            
            else:
                class_id = class_names.index(class_name_sp)
                #D
                #print('Esta es la clase idx :',class_id)
                    
    
            sortingForObj = np.asarray(shape["points"])
            sort_x = np.argsort(sortingForObj[:,0])#order an arr by ascendent value indx 
            xmin = sortingForObj[sort_x[0]][0]
            xmax = sortingForObj[sort_x[-1]][0]
            
            sort_y = np.argsort(sortingForObj[:,1])
            ymin = sortingForObj[sort_y[0]][1]
            ymax = sortingForObj[sort_y[-1]][1]
            
            # (xmin, ymin), (xmax, ymax) = shape["points"]
            # swap if min is larger than max.
            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            
            self.bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)


        print('this are GT bboxes:',self.bboxes)

        return self.bboxes



        #for box in bboxes:
            #print('this are the bboxes:',box)

        #for label in labels:
            #print('this are the bb labels:',label)


    def calculate_iou(self,bbox_gt, bbox_pred):
        # Calculate coors
        x1_inter = max(bbox_gt[0], bbox_pred[0])
        y1_inter = max(bbox_gt[1], bbox_pred[1])
        x2_inter = min(bbox_gt[2], bbox_pred[2])
        y2_inter = min(bbox_gt[3], bbox_pred[3])

        # Calculate intersection area
        #area_inter = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
        area_inter = abs(x2_inter - x1_inter ) * abs( y2_inter - y1_inter )

        # Calculate the area of the GT and Predicted bounding boxes 
        #area_gt = (bbox_gt[2] - bbox_gt[0] + 1) * (bbox_gt[3] - bbox_gt[1] + 1)
        #area_pred = (bbox_pred[2] - bbox_pred[0] + 1) * (bbox_pred[3] - bbox_pred[1] + 1)
        area_gt = abs(bbox_gt[2] - bbox_gt[0] ) * abs(bbox_gt[3] - bbox_gt[1] )
        area_pred = abs(bbox_pred[2] - bbox_pred[0] ) * abs(bbox_pred[3] - bbox_pred[1] )
      
      

        #Calculate IOU
        iou = area_inter / float(area_gt + area_pred - area_inter)

        return iou
    
    def IOU(self, bboxes_gt, bboxes_pred):
        
        iou = iou_2 = 0.0
        
        if len(bboxes_gt) > 0 and len(bboxes_pred) > 0:
            iou = self.calculate_iou(bboxes_gt[0], bboxes_pred[0])

        if len(bboxes_gt) > 1 and len(bboxes_pred) > 1:
            iou_2 = self.calculate_iou(bboxes_gt[1], bboxes_pred[1])

        final_iou = (iou + iou_2) / 2
        return round(abs(final_iou), 6)

        
    def Run_IOU(self):
        #bbbox=[[351.7725830078125, 173.61370849609375, 463.0155029296875, 271.2545471191406], [43.20277404785156, 176.21890258789062, 128.79232788085938, 262.5395202636719]]
        self.GT_xyxybox()
        #self.IOU(bboxes_gt, bboxes_pred)

#################### DICE COEFFICIENT ########################
    def round_yolo_coors(self,pred_coors):
        
        
        #self.seg.pred_xyxy
        rounded_yolov8_coordinates = [[[int(round(x)), int(round(y))] for x, y in polygon] for polygon in pred_coors]
        #rounded_yolov8_coordinates = [[[int(round(x)), int(round(y))] for x, y in polygon] for polygon in yolov8_coordinates]

        #print('Coordenadas redondeadas de YOLOv8:', rounded_yolov8_coordinates)
        
        zero_mask = np.zeros((512, 512), dtype=np.uint8)
        self.yolo_binary_masks = [] 
        for coordinates in rounded_yolov8_coordinates:
            yolo_binary_mask = cv2.fillPoly(zero_mask, [np.array(coordinates)], 1)
            self.yolo_binary_masks.append(yolo_binary_mask)

        #print('these are yolo_binary_masks:',self.yolo_binary_masks)
        return self.yolo_binary_masks
    
    def GT_masks(self):
        filename=self.json_listdir[self.actual_json_idx]
             
        #create and objet of labelme of the filename
        label_file = labelme.LabelFile(filename=filename)

        #store image metadata in json files
        labelme.utils.img_data_to_arr(label_file.imageData)

        #create a dic with the keys and verify if label_file.shapes keys are in keep_classes
        keep_classes = ['cancer', 'mix', 'warthin']
        #print('this is keep_classes: ',keep_classes) 
        shapes = [shape for shape in label_file.shapes if shape["label"] in keep_classes]
        #print('this is shapes: ',shapes)

        mask_coordinates = [shape['points'] for shape in shapes]
        #print('Coordenadas de las máscaras:', mask_coordinates)

    
        rounded_coordinates = [[[int(round(x)), int(round(y))] for x, y in polygon] for polygon in mask_coordinates]
        #print('Coordenadas redondeadas:', rounded_coordinates)


        zero_mask = np.zeros((512, 512), dtype=np.uint8)
        self.GT_binary_masks = []

        for coordinates in rounded_coordinates:
            GT_binary_mask = cv2.fillPoly(zero_mask, [np.array(coordinates)], 1)
            self.GT_binary_masks.append(GT_binary_mask)


        #print('these are GT_binary_masks:',self.GT_binary_masks)
        return self.GT_binary_masks



    def dice_coefficient(self,mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        dice = (2. * intersection.sum()) / (mask1.sum() + mask2.sum())
        
        return dice
    

    def Run_dice(self,mask1, mask2):
        dice_scores = []

        min_length = min(len(mask1), len(mask2))
        for i in range(min_length):
            dice_score = self.dice_coefficient(mask1[i], mask2[i])
            dice_scores.append(dice_score)

        # Calcular el promedio de los coeficientes de Dice
        average_dice = np.mean(dice_scores)
        #print(f'Coeficientes de Dice individuales: {dice_scores}')
        #print(f'Coeficiente de Dice promedio: {average_dice}')
            
        return round(abs(average_dice), 6)
    


    

#################### ACCUARACY,PRECISION,RECALL ########################
    def create_bbox_mask(self,image_size, bboxes):
        
        if bboxes:
            mask = np.zeros(image_size, dtype=np.uint8)

            for bbox in bboxes:
                bbox = [int(coord) for coord in bbox]  # Convertir las coordenadas a enteros
                #cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1, thickness=cv2.FILLED)
                mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]=1
            #return mask
                #mask=np.expand_dims(mask, axis=-1)    
            return mask
        
        else:
            mask = np.zeros(image_size, dtype=np.uint8)
            return mask
        
    

    def bbox_binary_mask(self,Gt_Bboxes,Pred_Bboxes):
        Gt_Bmask=self.create_bbox_mask((512, 512),Gt_Bboxes)
        Pred_Bmask=self.create_bbox_mask((512, 512),Pred_Bboxes)
        
        print(f"Gt_Bmask Shape: {Gt_Bmask.shape}")
        print(f"Pred_Bmask Shape: {Pred_Bmask.shape}")


        GT_how_many_ones=np.sum(Gt_Bmask==1)
        GT_how_many_zeros=np.sum(Gt_Bmask==0)

        Pred_how_many_ones=np.sum(Pred_Bmask==1)
        Pred_how_many_zeros=np.sum(Pred_Bmask==0)
        print(f"Number of ones in Gt_Bmask: {GT_how_many_ones}")
        print(f"Number of zeros in Gt_Bmask: {GT_how_many_zeros}")

        print(f"Number of ones in Pred_Bmask: {Pred_how_many_ones}")
        print(f"Number of zeros in Pred_Bmask: {Pred_how_many_zeros}")
        
        
        return Gt_Bmask,Pred_Bmask
    
    def calculate_metrics(self,pred_mask, GT_mask):
        # (True Positives)
        TP = np.sum(np.logical_and(pred_mask, GT_mask))

        # (True Negatives)
        TN = np.sum(np.logical_and(~pred_mask, ~GT_mask))

        #(False Positives)
        FP = np.sum(np.logical_and(pred_mask, ~GT_mask))

        #(False Negatives)
        FN = np.sum(np.logical_and(~pred_mask, GT_mask))

        #(Accuracy)
        self.accuracy = (TP + TN) / (TP + TN + FP + FN)

        if TP+FP==0:
            self.precision=0.0
            
        else:
            self.precision=TP/(TP+FP)
            
        
        if TP+FN==0:
            self.recall=0.0
            
        else:
            self.recall=TP/(TP+FN)

        
        print(f"this TP:{TP} , this is TN {TN}, this is FP {FP} and this FN{FN}")
        print('this is accuracy',self.accuracy)
        print('this is precision',self.precision)
        print('this is recall',self.recall)

        self.accuracy=round(abs(self.accuracy),6)
        self.precision=round(abs(self.precision),6)
        self.recall=round(abs(self.recall),6)

        return self.accuracy,self.precision,self.recall
    
    #################### VISUALIZATION ########################
    
    def visualize_Bboxes_masks(self, masks_list, titles):
        num_masks = len(masks_list[0])

        plt.figure(figsize=(12, 4 * num_masks))
        plt.suptitle(titles[0] + ' vs ' + titles[1])

        for i in range(num_masks):
            print(f'Mask {i + 1} shape: {masks_list[0][i].shape}')  # Añadir esta línea
            plt.subplot(num_masks, 3, 3 * i + 1)
            plt.imshow(np.array(masks_list[0][i], dtype=np.uint8).reshape((512, 512)), cmap='gray', vmin=0, vmax=255)
            plt.title(f'Mask {i + 1} - {titles[0]}')

            plt.subplot(num_masks, 3, 3 * i + 2)
            plt.imshow(np.array(masks_list[1][i], dtype=np.uint8).reshape((512, 512)), cmap='gray', vmin=0, vmax=255)
            plt.title(f'Mask {i + 1} - {titles[1]}')

            plt.subplot(num_masks, 3, 3 * i + 3)
            plt.hist(masks_list[0][i].ravel(), bins=[-0.5, 0.5, 1.5], color='black', alpha=0.7)
            plt.title(f'Mask {i + 1} Histogram - {titles[0]}')
            plt.xticks([0, 1])

        plt.show()



    def visualize_masks(masks, title):
        num_masks = len(masks)

        plt.figure(figsize=(12, 4 * num_masks))
        plt.suptitle(title)

        for i in range(num_masks):
            plt.subplot(num_masks, 2, 2 * i + 1)
            plt.imshow(masks[i], cmap='gray')
            plt.title(f'Mask {i + 1}')
            plt.colorbar()

            plt.subplot(num_masks, 2, 2 * i + 2)
            plt.hist(masks[i].ravel(), bins=[-0.5, 0.5, 1.5], color='black', alpha=0.7)  # Cambiado aquí
            plt.title(f'Mask {i + 1} Histogram')
            plt.xticks([0, 1])

        plt.show()


    


if __name__ == "__main__":
    
    obj=MetricsRaiza()
    #obj.Run_IOU()
    #obj.GT_masks()
    #obj.round_yolo_coors()
    #obj.Run_dice(obj.yolo_binary_masks,obj.GT_binary_masks)
    # Visualizar las máscaras de YOLOv8
    #obj.visualize_masks(obj.yolo_binary_masks, 'Máscaras de YOLOv8')

    # Visualizar las máscaras de Ground Truth
    #obj.visualize_masks(obj.GT_binary_masks, 'Máscaras de Ground Truth')
    # Visualizar las máscaras de YOLOv8
    
    #pred_bboxes= [[351.7725830078125, 173.61370849609375, 463.0155029296875, 271.2545471191406], [43.20277404785156, 176.21890258789062, 128.79232788085938, 262.5395202636719]]
    #GT_bboxes= [[181.05263157894737, 348.57894736842104, 280.52631578947364, 464.36842105263156], [172.6315789473684, 41.73684210526315, 264.7368421052631, 122.26315789473682]]
    
    #Gt_Bmask,Pred_Bmask=obj.bbox_binary_mask(GT_bboxes,pred_bboxes)
    #print('this is Gt_Bmask:',Gt_Bmask)
    #print('this is Gt_Bmask:',Pred_Bmask)
    
    #obj.calculate_metrics(Pred_Bmask,Gt_Bmask)
   
