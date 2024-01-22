from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import cv2 as cv

import imgviz
import labelme
import shutil
import numpy as np
try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)



class GtGenerator:
    def __init__(self):
        #self.input_dir=os.path.join('./dataset/','demo_test9/')
        #print('this is dir',self.input_dir)

        self.detect_dir = os.path.join('./dataset/Sorted Data/GT/','detection/') #detection Output
        self.segmetation_dir = os.path.join('./dataset/Sorted Data/GT/','segmentation/') #segmentation Output

    def detect_labelme(self,input_dir):
        shutil.rmtree(self.detect_dir)   
        os.makedirs(self.detect_dir)
        print(f"Folder: {self.detect_dir} overwritten" )

        #Clases for detection boxes
        class_names = ('left normal', 'right normal')
        #D
        #print("class_names tuple:", class_names)


        class_colors = {
            'left normal': (255, 255, 0),    # Amarillo
            'right normal': (255, 0, 255)    # Magenta
        }
        self.gt_list_dir=[]

        for filename in glob.glob(osp.join(input_dir, "*.json")):# to find all json files in the input folder
            #D
            #print("Generating GT image from:", filename)

            #constructor labelme
            label_file = labelme.LabelFile(filename=filename)#create and objet of labelme of the filename

            #osp.splitext splits name and extention from the path
            base = osp.splitext(osp.basename(filename))[0]

            out_viz_file = osp.join(self.detect_dir, base + ".png")
            self.gt_list_dir.append(out_viz_file)
            #print("out_viz_file path or paths?:", out_viz_file)

            
            if osp.exists(out_viz_file):
                shutil.rmtree(out_viz_file)
                #print("Gt image already exists:", out_viz_file)
                
                os.makedirs(out_viz_file)
                print("Image has been overwritten!")
    
        
            img = labelme.utils.img_data_to_arr(label_file.imageData)#store image metadata in json files
            
            bboxes = []
            labels = []
            colors = []
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

                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)
                #D
                #print(f'Labels: {labels}')
                
                colors.append(class_colors[class_name_sp])

                captions = [class_names[label] for label in labels]
                #D
                #print('this is captions', captions)


                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    bboxes=bboxes,
                    captions=captions,
                    font_size=15,
                    colormap=colors,
                )
                imgviz.io.imsave(out_viz_file, viz)

        print('this is self.gt_list_dir:',self.gt_list_dir)



    def segmentation_labelme(self,input_dir): 
        shutil.rmtree(self.segmetation_dir) 
        os.makedirs(self.segmetation_dir)
        print(f"Folder: {self.segmetation_dir} overwritten" )

        #Clases for detection boxes
        class_names = {
            'background':0,
            'cancer': 1,
            'mix': 2,
            'warthin': 3
        }

    
        classes=[]
        
        for _,value in enumerate(class_names):
            class_name=value
            classes.append(class_name)
            

            #print(f'this is class names: {class_name}')
            #print(f'this is class id: {class_id}')

        #print(f'this is class idx: {class_idx}')

        classes=tuple(classes)
        #print(f'this is classes: {classes}')

        class_colors = np.array([
        (0, 0, 0),  # Negro
        (0, 255, 0),  # Verde para la clase 'cancer'
        (0, 255, 255),  # Azul para la clase 'mix'
        (255, 0, 0),  # Rojo para la clase 'warthin'
        ])

       
        self.seg_Gt_listdir=[]
        for filename in glob.glob(osp.join(input_dir, "*.json")):# to find all json files in the input folder
            label_file = labelme.LabelFile(filename=filename)
            

            base = osp.splitext(osp.basename(filename))[0]
            out_viz_file = osp.join(self.segmetation_dir, base + ".png")
            self.seg_Gt_listdir.append(out_viz_file)
            
            img = labelme.utils.img_data_to_arr(label_file.imageData)#store image metadata in json files
            
            #create a dic with the keys and verify if label_file.shapes keys are in keep_classes
            keep_classes = set(class_names.keys()) 
            #print('this is keep_classes: ',keep_classes) 
            shapes = [shape for shape in label_file.shapes if shape["label"] in keep_classes]
            #print('this is shapes: ',shapes) 
            
            #Returns class, instance
            cls, ins = labelme.utils.shapes_to_label(
                img_shape=img.shape,# dimensions of th img
                shapes=shapes,
                label_name_to_value=class_names,
                
            )

            ins[cls == -1] = 0
            #print(f'this is cls: {cls} and this is ins: {ins}')
            #To set the background pixels , if =-1 assing 0.
            
            #print(f'this is ins: {ins}')
            
            clsv = imgviz.label2rgb(cls,
                img,
                label_names=classes,
                font_size=15,
                loc="rb",
                colormap=class_colors,)
            
            imgviz.io.imsave(out_viz_file, clsv)

        #print('this is self.seg_Gt_listdir:',self.seg_Gt_listdir)
            
          
            
if __name__ == "__main__":
    obj=GtGenerator()
    input_dir=os.path.join('./dataset/','demo_test9/')
    #obj.detect_labelme(input_dir)
    #obj.segmentation_labelme(input_dir)

    
   

        