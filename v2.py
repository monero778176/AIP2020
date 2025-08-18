
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from method_model import edge_
from method_model.blur_ import BlurWindow
from method_model.noise_add import NoiseWindow
from method_model.result_ import ResultWindow
from model.mask_viewmodel import maskTableModel
import ui6.pyui_ui as ui
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

    # img_type(gray or rgb), np_img
    send_img = pyqtSignal(tuple)  # 傳遞到 resultWindow 的訊號
    # send_img2 = pyqtSignal(tuple)  # 傳遞到 resultWindow 的訊號

    # mask_type, mask_name, mask_np
    mask_request = pyqtSignal(str)  # edge 偵測、blur處理

    def __init__(self):
        super().__init__()        

        self.setupUi(self)
        self.initUI()
        self.initParam()
        



    def initUI(self):
        self.origin_img = ''

        # --- 設定圖片顯示標籤 ---
        # 尺寸策略設為 Ignored，使其完全由佈局(layout)控制大小，而不是由內容（圖片）決定。
        # 這可以防止標籤因加載大圖片而無限擴張，從而擠壓到其他UI元件。
        size_policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lb_input.setSizePolicy(size_policy)
        self.lb_output.setSizePolicy(size_policy)

        # 啟用內容縮放，讓圖片自動縮放以填滿標籤的當前大小，並保持長寬比。
        self.lb_input.setScaledContents(True)
        self.lb_output.setScaledContents(True)
        # -------------------------

        self.btn_select_img.clicked.connect(self.click_select_dir)
        self.actionSelect_Image.triggered.connect(self.click_select_dir)
 

        # edge detection window init
        
        # self.actionedge_detection.triggered.connect(self.open_edge_window)  # 開啟 edge mask 選擇介面

        self.btn_gray.clicked.connect(self.open_result_window)


        # sub window defined
        self.sub_window = subWindow()
        # self.result_window = ResultWindow()  # show the image result by button

        self.edge_window = EdgeWindow(self.mask_request)   # mask 主編輯頁面
        self.blur_window = BlurWindow(self.send_img)
        self.noise_window = NoiseWindow(self.send_img)

        ## manu bar 點擊事件
        self.actionEdge_detection.triggered.connect(lambda checked=False: self.open_edge_window('edge'))
        self.actionBlur_processing.triggered.connect(self.open_blur_window)
        self.actionGaussian_noise.triggered.connect(self.open_gaussian_noise)


        # signal recive
        self.noise_window.gaussian_param.connect(self.receive_select_mask)
        self.edge_window.mask_np.connect(self.receive_select_mask)  # 接收 edge mask 選擇結果訊號接收
        self.blur_window.return_blur.connect(self.receive_select_mask)

        self.btn_apply_effect.setVisible(False)
        self.btn_apply_effect.clicked.connect(self.apply_effect_state)  # 套用目前的 effect


        self.btn_check_effect.clicked.connect(self.effect_detail)
        


        ## 效果疊加
        self.list_effect.itemChanged.connect(self.on_list_item_changed)

    def initParam(self):
        self.gray_image = np.array([])

        self.rgb_image=np.array([False])  # 定義初始影像判定
        self.effect_img=np.array([False])  # 定義初始影像判定
        self.image=np.array([False])  # 定義初始影像判定

        self.effect_blur_list = []  
        self.effect_edge_list = []   # 

        self.effect_total = {}

        ## 四種 edge mask
        self.db_list = {
            1:[[-1,0,1],
                [-2,0,2],
                [-1,0,1]],
            2: [[1,2,1],
                [0,0,0],
                [-1,-2,-1]],

            # scharr
            3: [[-3,0,3],   
                [-10,0,10],
                [-3,0,3]],  # y-axis
            4: [[3,10,3],   
                [0,0,0],
                [-3,-10,-3]],  # y-axis
        }

        # Gaussian noise
        self.mean_ = 0
        self.sigma_ = 25


        self.filter_list = ['gaussian','median','bilateral']


    def click_select_dir(self):
        """
        瀏覽本機檔案並載入
        """
        filename, filetype = QFileDialog.getOpenFileName()
        print(f'選擇了:{filename}')
        exten = filename.split('.')[-1]

        if filename:
            image = cv2.imread(filename)
            self.image = image.copy()
            print(f'image size:{image.size}')
            img_size = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.rgb_image = rgb_image
            # self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # print(f'一般灰階:{gray_image.shape}')
            # print(f'一般灰階:{len(gray_image.shape)}')
            
            put_img_to_label(rgb_image,self.lb_input)
            self.lb_detail.setText(f'Width: \t {img_size[1]} \nHeight: \t {img_size[0]} \nType: \t {exten} \n  ')


            hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
            print(f'hisogram shape:{hist.shape}')
            
            # fig = plt.bar(range(1,257), hist.reshape(-1))
            show_matplotlib_plot(hist,'bar',self.lb_hist)  # 畫 histogram data 到 label 上

            self.lb_output.clear()
            self.effect_total = {}
           
    def open_sub_window(self):
        """
        測柿子視窗開啟功能
        """
        
        self.sub_window.show()
        


    def open_edge_window(self,text):
        """
        開啟 '邊緣遮測' 自定義選擇遮罩
        """
        self.edge_window.cb_select.setCurrentIndex(0)
        model = maskTableModel([])
        self.edge_window.tv_mask.setModel(model)

        # self.edge_window..connect(self.mask_request)
        print(f'edge window 傳送的是:{text}, type:{type(text)}')
        self.mask_request.emit(text)

        # open window
        self.edge_window.show()


    def open_blur_window(self):
        """
        開啟模糊處理視窗 (預覽功能)

        Args:
            str: 識別的名稱
            np: 陣列影像
        """
        if self.image.any():
            self.send_img.emit(('rgb',self.image))
            self.blur_window.show()


    def open_result_window(self):
        '''根據不同的 button 傳送'''
        if (self.gray_image.size)!=0:
            print(f'傳送灰階圖片，大小:{self.gray_image.shape}')

            self.result_window = ResultWindow(self.send_img)  #綁定訊號
            self.send_img.emit(('gray',self.gray_image))  # 發送資料給 sub window
            
            self.result_window.show()
    
    def open_gaussian_noise(self):
        """
        開啟 添加'高斯雜訊'預覽視窗
        """
        # self.noise_window = NoiseWindow(self.send_img)

        if self.rgb_image.any():
            self.send_img.emit(('output',self.rgb_image))
            self.noise_window.show()
        

    @pyqtSlot(tuple)
    def receive_select_mask(self,data):
        print(f'觸發接收功能')

        get_kernel = 0
        mask = []
        type_name = ''

        type_name,_, _ = data
        print(f'接收到的處理功能類型:{type_name}')


        if type_name=='edge':
            # 若有 edge ，需要先做 grayscale 效果會更好
            if item_text_exists(self.list_effect,'grayscale')==False:

                effect_name = 'grayscale'
                effect1 = QListWidgetItem(effect_name)
                font = effect1.font()
                font.setPointSize(10)
                effect1.setFont(font)

                effect1.setFlags(effect1.flags()| Qt.ItemFlag.ItemIsUserCheckable)
                effect1.setCheckState(Qt.CheckState.Checked)
                self.list_effect.addItem(effect1)

                save_value ={'category':'grayscale'}  # 建立對應的 effect dict
                save_to_dict_(self.effect_total,effect_name,save_value)

        if type_name=='edge':
            # 來自 edge window 添加的 mask 效果
            type_name,mask_type,mask = data   # 回傳 mask 名稱 以及對應的 mask
            
            # edge dection: 
            effect_name = f'{type_name} {mask_type}'
            effect1 = QListWidgetItem(effect_name)

            save_value = {'category':'edge','mask':mask_type,'mask_np': mask}
            self.effect_total = save_to_dict_(self.effect_total,effect_name,save_value)

            # 試驗回傳結果(最終拔除)
            self.lb_mask_result.setText(f'接收的 array:{mask}')

        if type_name=='gaussian_noise':
            # 雜訊細項
            print(f'處理高斯雜訊')
            _,self.mean_, self.sigma_ = data

            effect_name = f'{type_name} mean:{self.mean_} sigma:{self.sigma_}'
            effect1 = QListWidgetItem(effect_name)

            save_value = {'category':'noise','mean':self.mean_,'sigma':self.sigma_}
            self.effect_total = save_to_dict_(self.effect_total,effect_name,save_value)



        if type_name=='blur':  # blur
            # 類別, filter or mask name, kernel_size or mask_value
            _ ,mask_type, get_kernel = data

            if mask_type==1:
                #gaussian filter
                effect_name = f'{type_name} gaussian k-{get_kernel}'


            elif mask_type==2:
                # medium filter
                effect_name = f'{type_name} median k-{get_kernel}'

                
            elif mask_type==3:
                # bilateral filter
                effect_name = f'{type_name} bilateral k-{get_kernel}'

            effect1 = QListWidgetItem(f'{effect_name}')
            
            # 紀錄到效果字典
            save_value = {'category':'blur','filter':self.filter_list[mask_type],'kernel_size':get_kernel}
            self.effect_total =save_to_dict_(self.effect_total,effect_name,save_value)

        

            

        font = effect1.font()
        font.setPointSize(10)
        effect1.setFont(font)

        effect1.setFlags(effect1.flags()| Qt.ItemFlag.ItemIsUserCheckable)
        effect1.setCheckState(Qt.CheckState.Checked)
        self.list_effect.addItem(effect1)


        self.btn_apply_effect.setVisible(True)


    def apply_effect_state(self):
        """
        改寫 apply_effect_state 使用全域的 effect-list
        以 listwidget 的 itemname 的 checkState 作為
        """


        self.btn_apply_effect.setVisible(False)
        self.effect_img = self.rgb_image.copy()
        gray_state = False

        for i in range(self.list_effect.count()):
            item_ = self.list_effect.item(i)
            if item_.checkState()==Qt.CheckState.Checked:

                # 使用 ListWidget 作為 key，找對應的內容
                key_name = item_.text()
                get_value_dict = self.effect_total.get(key_name)

                if get_value_dict:
                    type_category = get_value_dict.get('category')   # 獲得處理的模式
                    if type_category=='grayscale' and gray_state==False:
                        self.effect_img = cv2.cvtColor(self.effect_img,cv2.COLOR_RGB2GRAY)
                        gray_state = True

                        

                    if type_category=='edge':
                        if gray_state==False:
                            self.effect_img = cv2.cvtColor(self.effect_img,cv2.COLOR_RGB2GRAY)
                            gray_state = True

                        self.effect_img = add_edge(self.effect_img,get_value_dict)
                        

                    if type_category=='blur':
                        self.effect_img = add_blur(self.effect_img,get_value_dict)


        put_img_to_label(self.effect_img,self.lb_output)



    def effect_detail(self):
        for item in self.effect_total:
            print(f'type name:{item}')
            item_value = self.effect_total.get(item)
            print(f'執行內容有:{item_value}')

    def on_list_item_changed(self, item):
        """
        當 list widget 項目狀態改變時觸發。
        主要功能:
        1. 顯示 "Apply Effect" 按鈕。
        2. 當 "edge" 相關效果被勾選時，自動勾選 "grayscale"。
        """
        self.btn_apply_effect.setVisible(True)

        # 暫時禁用信號避免遞迴觸發
        self.list_effect.blockSignals(True)

        try:
            item_text = item.text()
            effect_details = self.effect_total.get(item_text)

            # 如果是 edge 效果且被勾選，則自動勾選 grayscale
            if effect_details and effect_details.get('category') == 'edge':
                if item.checkState() == Qt.CheckState.Checked:
                    # 尋找 grayscale item 並勾選
                    for i in range(self.list_effect.count()):
                        list_item = self.list_effect.item(i)
                        if list_item.text() == 'grayscale':
                            list_item.setCheckState(Qt.CheckState.Checked)
                            break
        finally:
            # 重新啟用信號
            self.list_effect.blockSignals(False)

def item_text_exists(list_widget, target_text):
    """
    Checks if an item with the given text exists in the QListWidget.

    Args:
        list_widget (QListWidget): The QListWidget to search.
        target_text (str): The text to search for.

    Returns:
        bool: True if an item with the target_text exists, False otherwise.
    """
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.text() == target_text:
            return True
    return False     

def save_to_dict_(dict_,key_type,save_value):
    if dict_.get(key_type)==None:
        dict_.setdefault(key_type,save_value)
    else:
        dict_[key_type] = save_value 
        
    return dict_
    

if __name__=='__main__':

    import qdarktheme
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light")
    window = MyWindow()
    window.show()
    sys.exit(app.exec())