from pickletools import pystring
from matplotlib.pyplot import gray
from func.convert import put_img_to_label
import ui6.Ui_Edge_mask_pyui as ui
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import numpy as np

import sys
sys.path.append('/app/ui6')
sys.path.append('/app/model')
import ui6.Ui_blur_mask as ui
from model.mask_viewmodel import maskTableModel
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import cv2


class BlurWindow(QMainWindow, ui.Ui_MainWindow):

    # blur_type, kernel_size
    return_blur = pyqtSignal(tuple)  # 回傳 blur type detail


    def __init__(self,main_signel):
        super().__init__() 
        self.setupUi(self)

        self.initUI()

        main_signel.connect(self.show_send_img)  # 一開始先顯示


    def initUI(self):
        ## 初始化元件以及依賴關係
        
        # 設定 kernel slider，及數值顯示
        self.slider_kernel.setRange(3,7)
        self.slider_kernel.setOrientation(Qt.Orientation.Horizontal)
        self.lb_kernel_size.setText(f'{self.slider_kernel.value()}')

        # 不同 blur filter 選擇
        self.cbb_filter.currentIndexChanged.connect(self.select_filter)
        
        
        self.slider_kernel.valueChanged.connect(self.show_kernel_number)

        self.cb_gray.clicked.connect(self.update_label)

        self.btn_reset.clicked.connect(self.preview_reset)

        self.groupBox_3.setVisible(False)


        self.btn_confirm.clicked.connect(self.confirm_blur_mask)
        

    def show_kernel_number(self,value):
        if value % 2 == 0:
            # 如果是偶數，將其調整為最接近的奇數
            # 這裡簡單地將其設定為 3 或 5
            if value == 4:
                self.slider_kernel.setValue(3)
            elif value == 6:
                self.slider_kernel.setValue(5)
        self.lb_kernel_size.setText(f'{self.slider_kernel.value()}')

        self.select_filter()  # 更新當前值


    def show_send_img(self,data):
        # 接收訊號資料，初始定義資料
        img_type, self.get_img = data   # str, np
        self.show_rgb = cv2.cvtColor(self.get_img,cv2.COLOR_BGR2RGB)
        self.now_preview = self.show_rgb
        put_img_to_label(self.show_rgb,self.lb_show)

    def preview_reset(self):
        if self.cb_gray.isChecked():
            gray_ = cv2.cvtColor(self.get_img,cv2.COLOR_BGR2GRAY)
            self.now_preview = gray_
            put_img_to_label(gray_,self.lb_show)

        else:
            self.now_preview = self.show_rgb
            put_img_to_label(self.show_rgb,self.lb_show)
    
    def update_label(self):
        ## 顯示效果疊加到圖片的 preview
        if self.cb_gray.isChecked():
            gray_ = cv2.cvtColor(self.get_img,cv2.COLOR_BGR2GRAY)
            self.now_preview = gray_
            put_img_to_label(gray_,self.lb_show)

        else:
            self.now_preview = self.show_rgb
            put_img_to_label(self.show_rgb,self.lb_show)

    def select_filter(self):
        get_kernel = self.slider_kernel.value()  # 獲得 nxn 大小 kernel

        get_filter = self.cbb_filter.currentIndex()

        result=np.array([False])
        if get_filter==1:
            self.groupBox_3.setVisible(False)
            result = cv2.GaussianBlur(self.now_preview,(get_kernel,get_kernel),0)

        elif get_filter==2:
            self.groupBox_3.setVisible(False)
            result = cv2.medianBlur(self.now_preview,get_kernel)
            
        elif get_filter==3:
            self.groupBox_3.setVisible(True)
            result = cv2.bilateralFilter(self.now_preview,get_kernel,0,0)

        if result.any():
            self.update_label_with_value(result)
    
    def update_label_with_value(self,result):
        put_img_to_label(result,self.lb_show)



    def confirm_blur_mask(self):

        mask_type = self.cbb_filter.currentIndex()
        if mask_type!=0:
            kernel_size = self.slider_kernel.value()

            self.return_blur.emit(('blur',mask_type,kernel_size))
            self.close()