from PyQt5 import QtCore, QtGui, QtWidgets
from pre_detection import *
from pre_segmentation import *
from GT import *
from metrics import *
import os
import cv2 as cv
import shutil


class Ui_MainWindow(object):
    def __init__(self) -> None:
            #self.dir="C:/Users/User/OneDrive/Escritorio/NCKU/CLASSES/IMAGE PROCESSING/Hw_2/影像處理/labels.txt"
            self.detect=DetModel()
            self.segmentation=SegmModel()
            self.Gt_utils=GtGenerator()
            self.metrics=MetricsRaiza()
            self.actual_index =0
            self.actual_index_2 =0
            self.actual_index_seg =0
            self.msg=''
            self.ima_dir=os.path.join('./dataset/Sorted Data/','Original_imgs')
            self.json_dir=os.path.join('./dataset/Sorted Data/','Json_files')
            self.GT_dir=os.path.join('./dataset/Sorted Data/','GT')




    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(620, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -40, 621, 901))
        self.label.setStyleSheet("border-image:url('snow.jpg')")
        self.label.setText("")
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 0, 601, 71))
        self.textEdit.setObjectName("textEdit")
        self.Image_groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.Image_groupBox.setGeometry(QtCore.QRect(10, 90, 601, 241))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(28)
        font.setBold(False)
        font.setWeight(50)
        self.Image_groupBox.setFont(font)
        self.Image_groupBox.setObjectName("Image_groupBox")


        self.Load_folder_PButton = QtWidgets.QPushButton(self.Image_groupBox)
        self.Load_folder_PButton.setGeometry(QtCore.QRect(70, 30, 471, 41))
        self.Load_folder_PButton.setObjectName("Load_folder_PButton")


        self.Pre_PButton_4 = QtWidgets.QPushButton(self.Image_groupBox)
        self.Pre_PButton_4.setGeometry(QtCore.QRect(10, 90, 281, 41))
        self.Pre_PButton_4.setObjectName("Pre_PButton_4")


        self.Next_PButton_5 = QtWidgets.QPushButton(self.Image_groupBox)
        self.Next_PButton_5.setGeometry(QtCore.QRect(310, 90, 281, 41))
        self.Next_PButton_5.setObjectName("Next_PButton_5")


        self.Current_img_static = QtWidgets.QLabel(self.Image_groupBox)
        self.Current_img_static.setGeometry(QtCore.QRect(10, 150, 151, 61))
        self.Current_img_static.setObjectName("Current_img_static")


        self.Current_img_dynamic = QtWidgets.QLabel(self.Image_groupBox)
        self.Current_img_dynamic.setGeometry(QtCore.QRect(200, 150, 371, 61))
        self.Current_img_dynamic.setObjectName("Current_img_dynamic")


        self.detection_groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.detection_groupBox_2.setGeometry(QtCore.QRect(10, 330, 601, 331))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(28)
        font.setBold(False)
        font.setWeight(50)
        self.detection_groupBox_2.setFont(font)
        self.detection_groupBox_2.setObjectName("detection_groupBox_2")


        self.Detection_PButton_2 = QtWidgets.QPushButton(self.detection_groupBox_2)
        self.Detection_PButton_2.setGeometry(QtCore.QRect(50, 40, 511, 41))
        self.Detection_PButton_2.setObjectName("Detection_PButton_2")


        self.IoU_static_2 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.IoU_static_2.setGeometry(QtCore.QRect(20, 90, 61, 41))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.IoU_static_2.setFont(font)
        self.IoU_static_2.setObjectName("IoU_static_2")

        self.IoU_dynamic_2 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.IoU_dynamic_2.setGeometry(QtCore.QRect(90, 90, 491, 41))
        self.IoU_dynamic_2.setText("")
        self.IoU_dynamic_2.setObjectName("IoU_dynamic_2")


        self.Accuaracy_static_3 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Accuaracy_static_3.setGeometry(QtCore.QRect(10, 140, 141, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Accuaracy_static_3.setFont(font)
        self.Accuaracy_static_3.setObjectName("Accuaracy_static_3")

        
        self.Accuaracy_dynamic_4 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Accuaracy_dynamic_4.setGeometry(QtCore.QRect(160, 140, 421, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Accuaracy_dynamic_4.setFont(font)
        self.Accuaracy_dynamic_4.setText("")
        self.Accuaracy_dynamic_4.setObjectName("Accuaracy_dynamic_4")


        self.Precision_static_4 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Precision_static_4.setGeometry(QtCore.QRect(10, 200, 121, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Precision_static_4.setFont(font)
        self.Precision_static_4.setObjectName("Precision_static_4")


        self.Precision_dynamic_5 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Precision_dynamic_5.setGeometry(QtCore.QRect(140, 200, 441, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Precision_dynamic_5.setFont(font)
        self.Precision_dynamic_5.setText("")
        self.Precision_dynamic_5.setObjectName("Precision_dynamic_5")


        self.Precision_static_5 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Precision_static_5.setGeometry(QtCore.QRect(10, 260, 91, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Precision_static_5.setFont(font)
        self.Precision_static_5.setObjectName("Precision_static_5")



        self.Recall_dynamic_6 = QtWidgets.QLabel(self.detection_groupBox_2)
        self.Recall_dynamic_6.setGeometry(QtCore.QRect(110, 260, 471, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Recall_dynamic_6.setFont(font)
        self.Recall_dynamic_6.setText("")
        self.Recall_dynamic_6.setObjectName("Recall_dynamic_6")


        self.Segment_groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.Segment_groupBox_3.setGeometry(QtCore.QRect(10, 660, 601, 201))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(28)
        font.setBold(False)
        font.setWeight(50)
        self.Segment_groupBox_3.setFont(font)
        self.Segment_groupBox_3.setObjectName("Segment_groupBox_3")
        self.Segmentation_PButton_3 = QtWidgets.QPushButton(self.Segment_groupBox_3)
        self.Segmentation_PButton_3.setGeometry(QtCore.QRect(50, 40, 511, 51))
        self.Segmentation_PButton_3.setObjectName("Segmentation_PButton_3")


        self.Dice_coe_static_4 = QtWidgets.QLabel(self.Segment_groupBox_3)
        self.Dice_coe_static_4.setGeometry(QtCore.QRect(20, 110, 211, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Dice_coe_static_4.setFont(font)
        self.Dice_coe_static_4.setObjectName("Dice_coe_static_4")


        self.Dice_coe_dynamic_5 = QtWidgets.QLabel(self.Segment_groupBox_3)
        self.Dice_coe_dynamic_5.setGeometry(QtCore.QRect(250, 110, 341, 61))
        font = QtGui.QFont()
        font.setFamily("Freestyle Script")
        font.setPointSize(36)
        font.setBold(False)
        font.setWeight(50)
        self.Dice_coe_dynamic_5.setFont(font)
        self.Dice_coe_dynamic_5.setText("")
        self.Dice_coe_dynamic_5.setObjectName("Dice_coe_dynamic_5")

        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 620, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Image_groupBox.setTitle(_translate("MainWindow", "Image"))
        self.Load_folder_PButton.setText(_translate("MainWindow", "Load Folder"))
        self.Pre_PButton_4.setText(_translate("MainWindow", "Pre"))
        self.Next_PButton_5.setText(_translate("MainWindow", "Next"))
        self.Current_img_static.setText(_translate("MainWindow", "Current Image:"))
        self.Current_img_dynamic.setText(_translate("MainWindow", "File Name"))
        self.detection_groupBox_2.setTitle(_translate("MainWindow", "Detection"))
        self.Detection_PButton_2.setText(_translate("MainWindow", "Detection"))
        self.IoU_static_2.setText(_translate("MainWindow", "IoU:"))
        self.Accuaracy_static_3.setText(_translate("MainWindow", "Accuaracy:"))
        self.Precision_static_4.setText(_translate("MainWindow", "Precision:"))
        self.Precision_static_5.setText(_translate("MainWindow", "Recall:"))
        self.Segment_groupBox_3.setTitle(_translate("MainWindow", "Segment"))
        self.Segmentation_PButton_3.setText(_translate("MainWindow", "Segmentation"))
        self.Dice_coe_static_4.setText(_translate("MainWindow", "Dice Coeficcient:"))

        self.Load_folder_PButton.clicked.connect(self.load_img_folder)
        self.Pre_PButton_4.clicked.connect(self.prev_image)
        self.Next_PButton_5.clicked.connect(self.next_image)
        self.Detection_PButton_2.clicked.connect(self.det_pred_act)
        self.Segmentation_PButton_3.clicked.connect(self.seg_pred_act)

    def load_img_folder(self):

        self.dir=QtWidgets.QFileDialog.getExistingDirectory(None, "Select a folder","",QtWidgets.QFileDialog.ShowDirsOnly)
        #D
        #print(f'this is Dir: {self.dir}')
        self.Gt_utils.detect_labelme(self.dir)
        self.Gt_utils.segmentation_labelme(self.dir)
        
        try:

            if not self.dir:
                print('Folder corrupted or not selected')
                return None

            print('Folder loaded!')
            ima_listdir=os.listdir(self.dir)
            
            self.img_list= [file for file in ima_listdir if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) ]
            self.json_files = [f for f in ima_listdir if f.lower().endswith('.json')]
            
            shutil.rmtree(self.ima_dir)
            os.makedirs(self.ima_dir)        
            print(f'Images stored in directory: {self.ima_dir}')

            self.imgs_copied_dir = []
            for img in self.img_list:
                    source_path = os.path.join(self.dir, img)
                    destination_path = os.path.join(self.ima_dir, img)
                    
                    shutil.copyfile(source_path, destination_path)
                    self.imgs_copied_dir.append(os.path.join(self.ima_dir, img))

            shutil.rmtree(self.json_dir)
            os.makedirs(self.json_dir)
            print(f'Images stored in directory: {self.json_dir}')


            for json_file in self.json_files:
                    source_path = os.path.join(self.dir, json_file)
                    destination_path = os.path.join(self.json_dir, json_file)
                    shutil.copyfile(source_path, destination_path)


            if not os.path.exists(self.GT_dir):
                os.makedirs(self.GT_dir)
                print(f'Created directory: {self.GT_dir}')
            else:
                print(f'Images stored in existing directory: {self.GT_dir}')
            

            if self.imgs_copied_dir:
                self.show_current_image()
                
                    

            else:
                print('No images found in the folder or are corrupted!')

        
        except:
            print("Folder Corrupted or has chinese chars in the name!")

        
    def show_current_image(self):
        try:
            img = cv.imread(self.imgs_copied_dir[self.actual_index])
            self.ima_path=self.imgs_copied_dir[self.actual_index]
            #print('THIS IS THE PATH OF THE CURRENT IMAGE:',self.ima_path)
            self.FileName=os.path.basename(self.ima_path)
            self.Current_img_dynamic.setText(self.FileName)
            #print('THIS IS THE PATH OF THE CURRENT IMAGE FILE NAME:',self.FileName)
            

            if img is not None:
                img_resized = cv.resize(img, (600, 600))
                cv.imshow('Original', img_resized)

            else:
                print('Image could not be read')

        except Exception as e:
            print(f"Error: {e}")




    def Show_gt_current_img(self):
        try:
            img = cv.imread(self.Gt_utils.gt_list_dir[self.actual_index_2])
            #self.ima_path_2=self.Gt_utils.gt_list_dir[self.actual_index_2]
            
            
            if img is not None:
                img_resized = cv.resize(img, (600, 600))
                cv.imshow('GT', img_resized)
                key=cv.waitKey(0)

                if key==32:
                    cv.destroyAllWindows()
                    return

                

            else:
                print('Image could not be read')

        except Exception as e:
            print(f"Error: {e}")
        

    def showGT_seg_current_image(self):
        try:
            img = cv.imread(self.Gt_utils.seg_Gt_listdir[self.actual_index_seg])
            #ima_path=self.Gt_utils.seg_Gt_listdir[self.actual_index_seg]
            
            

            if img is not None:
                img_resized = cv.resize(img, (600, 600))
                cv.imshow('GT', img_resized)
                key=cv.waitKey(0)

                if key==32:
                    cv.destroyAllWindows()
                    return


            else:
                print('Image could not be read')

        except Exception as e:
            print(f"Error: {e}")
        


    def next_image(self):
        try:
            if self.imgs_copied_dir:
                self.actual_index = (self.actual_index + 1) % len(self.imgs_copied_dir)
                self.metrics.json_next()
                self.show_current_image()
                self.next_image_gt()
                self.next_Segimage_gt()
                key=cv.waitKey(0)

                if key==32:
                    cv.destroyAllWindows()
                    return
                   

        except Exception as e:
            print(f"Error: {e}")

    def prev_image(self):
        try:
            if self.imgs_copied_dir:
                self.actual_index = (self.actual_index - 1) % len(self.imgs_copied_dir)
                self.metrics.json_prev()
                self.show_current_image()
                self.prev_image_gt()
                self.prev_Segimage_gt()
                key=cv.waitKey(0)

                if key==32:
                    cv.destroyAllWindows()
                    return
                

        except Exception as e:
            print(f"Error: {e}")



    def next_image_gt(self):
        try:
            if self.Gt_utils.gt_list_dir:
                self.actual_index_2 = (self.actual_index_2 + 1) % len(self.Gt_utils.gt_list_dir)
            
            
        except Exception as e:
            print(f"Error: {e}")

    def prev_image_gt(self):
        try:
            if self.Gt_utils.gt_list_dir: 
                self.actual_index_2 = (self.actual_index_2 - 1) % len(self.Gt_utils.gt_list_dir)

            
        except Exception as e:
            print(f"Error: {e}")

    
    def next_Segimage_gt(self):
        try:
            if self.Gt_utils.seg_Gt_listdir:
                self.actual_index_seg = (self.actual_index_seg + 1) % len(self.Gt_utils.seg_Gt_listdir)
            
            
        except Exception as e:
            print(f"Error: {e}")

    def prev_Segimage_gt(self):
        try:
            if self.Gt_utils.seg_Gt_listdir:
                self.actual_index_seg = (self.actual_index_seg - 1) % len(self.Gt_utils.seg_Gt_listdir)

            
        except Exception as e:
            print(f"Error: {e}")



    def det_pred_act(self):
        self.detect.predict(self.ima_path)
        self.metrics.json_idx()
        bboxes=self.metrics.GT_xyxybox()
        self.msg=self.metrics.IOU(bboxes,self.detect.pred_xyxy)
        self.IoU_dynamic_2.setText(str(self.msg))
        Gt_Bmask,Pred_Bmask=self.metrics.bbox_binary_mask(self.metrics.bboxes,self.detect.pred_xyxy)
        Accuaracy,Precision,Recall=self.metrics.calculate_metrics(Pred_Bmask,Gt_Bmask)
        self.msg=Accuaracy
        self.Accuaracy_dynamic_4.setText(str(self.msg))
        self.msg=Precision
        self.Precision_dynamic_5.setText(str(self.msg))
        self.msg=Recall
        self.Recall_dynamic_6.setText(str(self.msg))
        self.Show_gt_current_img()
        
        #print('self.msg:',self.msg)
        #print('self.detect.pred_xyxy:',self.detect.pred_xyxy)
        #print('self.metrics.bboxes:',self.metrics.bboxes)
        


    def seg_pred_act(self):
        self.segmentation.predict(self.ima_path)
        self.metrics.json_idx()
        yolo_mask=self.metrics.round_yolo_coors(self.segmentation.pred_xyxy)
        GT_binary_masks=self.metrics.GT_masks()
        self.msg=self.metrics.Run_dice(yolo_mask,GT_binary_masks)
        self.Dice_coe_dynamic_5.setText(str(self.msg))
        self.showGT_seg_current_image()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) 
