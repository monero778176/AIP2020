
from pickletools import pystring
from matplotlib.pyplot import gray
from func.convert import put_img_to_label
import ui6.Ui_Edge_mask_pyui as ui
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import numpy as np
import cv2
from func.convert import *

import sys
sys.path.append('/app/ui6')
sys.path.append('/app/model')
import ui6.noise_ui as ui

class NoiseWindow(QMainWindow, ui.Ui_MainWindow):

    # mean, sigma
    gaussian_param = pyqtSignal(tuple)  # 回傳 blur type detail


    def __init__(self,main_signal):
        super().__init__() 
        self.setupUi(self)

        self.initUI()
        self.initParam()

        main_signal.connect(self.receive_preview)
        


    def initUI(self):
        self.slider_mean.setRange(-20,20)
        self.slider_mean.setValue(0)

        self.slider_sigma.setRange(0,25)
        self.slider_sigma.setValue(25)

        self.slider_mean.valueChanged.connect(self.update_slider_value)
        self.slider_sigma.valueChanged.connect(self.update_slider_value)

        self.lb_num_show.setText(f"mean:{self.slider_mean.value()}\nsigma:{self.slider_sigma.value()}")

        self.btn_confirm.clicked.connect(self.return_gaussian_value)


    def initParam(self):
        self.img_np = np.array([False])

    @pyqtSlot(tuple)
    def receive_preview(self,data):
        _,img_np = data
        self.img_np = img_np
        self.update_slider_value()

    

    def update_slider_value(self):
        self.lb_num_show.setText(f"mean:{self.slider_mean.value()}\nsigma:{self.slider_sigma.value()}")

        preview_img = self.img_np.copy()

        mean_ = self.slider_mean.value()
        sigma_ = self.slider_sigma.value()

        preview_img = add_gaussian_noise(preview_img,mean_,sigma_)

        put_img_to_label(preview_img,self.lb_preview)


    def return_gaussian_value(self):
        mean_ = self.slider_mean.value()
        sigma_ = self.slider_sigma.value()
        self.gaussian_param.emit(('gaussian_noise',mean_,sigma_))
        print(f'送出高斯結果:emit mean:{mean_}, sigma:{sigma_}')    
        self.close()

