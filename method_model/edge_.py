
import ui6.Ui_Edge_mask_pyui as ui
from PyQt6.QtWidgets import *

import numpy as np

import sys
sys.path.append('/app/ui6')
sys.path.append('/app/model')
import ui6.Edge_mask_pyui_ui as ui
from model.mask_viewmodel import maskTableModel
from PyQt6.QtCore import pyqtSignal, pyqtSlot



class EdgeWindow(QMainWindow, ui.Ui_MainWindow):

    mask_np = pyqtSignal(tuple)  # 用來傳遞 mask  str:name, list: mask

    def __init__(self,mask_request):
        super().__init__()
        self.setupUi(self)

        self.initUI()
        self.initParam()

        mask_request.connect(self.load_template_with_type)  # 根據需求產生不同的


    def initUI(self):
        self.cb_select.currentIndexChanged.connect(self.load_mask_template)
        
        self.btn_confirm.clicked.connect(self.send_mask)
        
    def initParam(self):
        self.select_mask=[]
        self.mask_name = ''
        self.mask_type = ''
        # self.cb_select.setCurrentIndex(0)


    # @pyqtSlot(str)
    def load_template_with_type(self,text):
        print(f'這時候的內容數量:{self.cb_select.count()}')
        
        if self.cb_select.count()!=0:
            print(f'已存在選項，重新設定中...')
            self.cb_select.clear()
            # self.cb_select.setCurrentIndex(0)

        if text=='edge':
            self.cb_select.addItem('快速選擇模板')
            self.cb_select.addItems([
                'Sobal (x-axis)',
                'Sobal (y-axis)',
                'Scharr (x-axis)',
                'Scharr (y-axis)'
            ])

            self.cb_select.setCurrentIndex(0)
            self.mask_type = 'edge'  # 邊緣處理的效果
                
        # elif text=='blur':


    def load_mask_template(self):
        """
        加載預設的 mask template 
        (最終可能修改成 db)
        """

        # db 資料庫
        db_list = {
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
        # self.cb_select.setDisabled(0)

        get_index = self.cb_select.currentIndex()
        print(f'當前選擇的 index:{get_index}')
        if get_index!=0 or get_index!=-1:
            get_mask = db_list.get(get_index)  # 獲得 list
            self.select_mask = get_mask
            self.mask_name = self.cb_select.currentText()
            print(f'現在選擇的是:{self.select_mask}, type:{type(self.select_mask)}')

            # 自動的資料調入
            model = maskTableModel(get_mask)
            self.tv_mask.setModel(model)

            # 貼合 tableView 大小
            header = self.tv_mask.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
    def send_mask(self):
        print(f'確認鍵點下')
        if len(self.select_mask)!=0 and self.mask_name!='':
            self.mask_np.emit(('edge',self.mask_name,self.select_mask))

        self.close()

       

        


        


