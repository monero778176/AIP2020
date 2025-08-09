import sys
from PyQt6.QtWidgets import *
import ui6.pyui_ui as ui
from pathlib import Path

class MyWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()        

        self.setupUi(self)


    def initUI(self):
        self.lb_input = ''

        self.btn_select_img.clicked.connect(self.click_select_dir)

        self.dir = str(Path.cwd())



    def click_select_dir(self):
        print(f'應當觸發選擇視窗')
        filePath, filterType = QFileDialog.getOpenFileNames()
        print(f'get file path: {filePath}')
        

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())