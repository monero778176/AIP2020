import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from method_model.result_ import ResultWindow
from model.mask_viewmodel import maskTableModel
import ui6.Ui_pyui as ui
from pathlib import Path
import os
import cv2
import io
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import ImageQt,Image
import numpy as np

from PyQt6 import QtWidgets
import plotly.graph_objects as go # Or plotly.express as px
import plotly.express as px

from sub_controller import subWindow  # 子視窗控制
from method_model.edge_ import EdgeWindow
from func.convert import *

os.environ["QT_QPA_PLATFORMTHEME"] = "xcb" 

class MyWindow(QMainWindow, ui.Ui_MainWindow):

    # 要顯示的 type(str), array
    send_img = pyqtSignal((np.ndarray))  # 傳遞到 resultWindow 的訊號

    def __init__(self):
        super().__init__()        

        self.setupUi(self)
        self.initUI()
        self.initParam()
        



    def initUI(self):
        self.origin_img = ''

        self.btn_select_img.clicked.connect(self.click_select_dir)
        self.actionselect_image.triggered.connect(self.click_select_dir)
        

        # 子視窗操作 (docker env not show action on manubar)
        # self.btn_sub_window.clicked.connect(self.open_sub_window)


        # edge detection window init
        self.actionedge_detection.triggered.connect(self.open_edge_window)
        self.btn_gray.clicked.connect(self.open_result_window)


        # sub window defined
        self.sub_window = subWindow()
        self.edge_window = EdgeWindow()
        # self.result_window = ResultWindow()  # show the image result by button

        # signal recive
        self.edge_window.mask_np.connect(self.reveive_select_mask)

    def initParam(self):
        self.gray_image = np.array([])



    def click_select_dir(self):
        print(f'應當觸發選擇視窗')
        filename, filetype = QFileDialog.getOpenFileName()
        print(f'選擇了:{filename}')
        exten = filename.split('.')[-1]

        if filename:
            image = cv2.imread(filename)
            print(f'image size:{image.size}')
            img_size = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # print(f'一般灰階:{gray_image.shape}')
            # print(f'一般灰階:{len(gray_image.shape)}')
            

            # height, width, channel = rgb_image.shape
            # bytesPerLine = 3 * width # For a 3-channel RGB image
            # q_image = QImage(rgb_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            # qpixmap = QPixmap.fromImage(q_image)

            # lbw,lbh = self.lb_input.size().width(),self.lb_input.size().height()  # 畫布大小取得

            # scale_pixmap = qpixmap.scaled(lbw,lbh,Qt.AspectRatioMode.KeepAspectRatio,
            #                     Qt.TransformationMode.SmoothTransformation)


            # self.lb_input.setPixmap(scale_pixmap)
            put_img_to_label(rgb_image,self.lb_input)
            self.lb_detail.setText(f'Width: \t {img_size[1]} \nHeight: \t {img_size[0]} \nType: \t {exten} \n  ')


            hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
            print(f'hisogram shape:{hist.shape}')
            
            # fig = plt.bar(range(1,257), hist.reshape(-1))
            show_matplotlib_plot(hist,'bar',self.lb_hist)  # 畫 histogram data 到 label 上
           
    def open_sub_window(self):
        
        self.sub_window.show()
        


    def open_edge_window(self):
        # init filter data
        self.edge_window.cb_select.setCurrentIndex(0)
        model = maskTableModel([])
        self.edge_window.tv_mask.setModel(model)

        # open window
        self.edge_window.show()


    def open_result_window(self):
        '''根據不同的 button 傳送'''
        if (self.gray_image.size)!=0:
            print(f'傳送灰階圖片，大小:{self.gray_image.shape}')

            self.result_window = ResultWindow(self.send_img)
            self.send_img.emit(self.gray_image)
            

            self.result_window.show()
    
    
        

    @pyqtSlot(list)
    def reveive_select_mask(self,mask):
        # ensure the main window receive mask data
        self.lb_mask_result.setText(f'接收的 array:{mask}')

# def convert_pil(fig):
#     return Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())


# def pil_convert_to_pixmap(pil_image):
#     pixmap = ImageQt.toqpixmap(pil_image)
#     return pixmap

  
# def show_plotly_plot():
    

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())