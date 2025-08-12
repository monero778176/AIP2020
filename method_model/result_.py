
import ui6.Ui_Edge_mask_pyui as ui
from PyQt6.QtWidgets import *

import numpy as np

import sys
sys.path.append('/app/ui6')
sys.path.append('/app/model')
import ui6.result_pyui_ui as ui
from model.mask_viewmodel import maskTableModel
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from func.convert import put_img_to_label

class ResultWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self,main_signal):
        super().__init__()

        self.setupUi(self)
        self.initUI()
        main_signal.connect(self.receive_gray)


    def initUI(self):

        
        pass


    def receive_gray(self,data):
        print(f'接收到的圖片大小:{data.shape}')
        put_img_to_label(data,self.lb_result)




        