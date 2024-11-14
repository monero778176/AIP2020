import math
import ntpath
import random
import numpy
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import io

import ui.pyui as ui
# import ui.mask as mask_ui
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.PIL_im = ''
        self.Gaussian_im = ''
        self.dic_path = ''
        self.kernel = ''
        self.edge_kernel = ''
        self.setupUi(self)
        self.setWindowTitle("AIP worker")
        self.cv_img = ''
        self.initUI()

    def initUI(self):

        # action 事件
        self.actionclear_result.triggered.connect(self.clearLabel)
        self.actionChoose_File.triggered.connect(self.getImage)
        self.actionExit.triggered.connect(qApp.quit)

        self.actionhistogram_equalization.setEnabled(False)
        self.actionhistogram_equalization.triggered.connect(self.gray_equalization)
        self.actionGaussian_Noise.setEnabled(False)
        self.actionGaussian_Noise.triggered.connect(self.Guassianbtnclick)

        self.btn_mask.clicked.connect(self.smooth)
        self.init_Slider_TableWidget()

        # self.btnChoose.clicked.connect(self.getImage)
        self.btnSave.setEnabled(False)
        self.btn2Bmp.setEnabled(False)
        self.btn2Haar.setEnabled(False)
        # self.btn2Gray.setEnabled(False)
        self.btn2Png.setEnabled(False)

        # 載入預設mask
        self.pushButton_preset.clicked.connect(self.preset_smooth_edge_mask)

        # 只能從RadioButton上直接觸發事件
        self.radioButton.setChecked(True)  # 先設定預設點選的radiobutton
        self.radioButton.clicked.connect(self.CheckRadioCheck)
        self.radioButton_2.clicked.connect(self.CheckRadioCheck)
        self.radioButton_3.clicked.connect(self.CheckRadioCheck)

        self.btnSave.clicked.connect(self.saveImage)
        self.btn2Bmp.clicked.connect(self.saveBmp)
        self.btn2Png.clicked.connect(self.savePng)

        # self.btn2Gray.clicked.connect(self.RGB2Gray)
        self.btn2Haar.clicked.connect(self.read2Haar)

    def kernel_set(self):
        smooth_size = self.slider_mask.value()
        edge_size = self.slider_mask_2.value()

        if smooth_size == 3:
            self.kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        elif smooth_size == 5:
            self.kernel = np.array(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        if edge_size == 3:
            self.edge_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        elif edge_size == 5:
            self.edge_kernel = np.array(
                [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1]])

    def preset_smooth_edge_mask(self):
        self.kernel_set()

        t1_column = self.tableWidget.columnCount()
        t1_row = self.tableWidget.rowCount()
        for i in range(t1_column):
            for j in range(t1_row):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.kernel[i, j])))

        t2_column = self.tableWidget_2.columnCount()
        t2_row = self.tableWidget_2.rowCount()
        for i in range(t2_column):
            for j in range(t2_row):
                self.tableWidget_2.setItem(i, j, QTableWidgetItem(str(self.edge_kernel[i, j])))

    def sliderValue(self):
        self.tableWidget.setRowCount(self.slider_mask.value())
        self.tableWidget.setColumnCount(self.slider_mask.value())
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.resizeColumnsToContents()
        self.kernel_set()
        self.preset_smooth_edge_mask()

    def sliderValue2(self):
        self.tableWidget_2.setRowCount(self.slider_mask_2.value())
        self.tableWidget_2.setColumnCount(self.slider_mask_2.value())
        self.tableWidget_2.resizeRowsToContents()
        self.tableWidget_2.resizeColumnsToContents()
        self.kernel_set()
        self.preset_smooth_edge_mask()

    def init_Slider_TableWidget(self):
        self.slider_mask.setMinimum(3)
        self.slider_mask.setMaximum(5)
        self.slider_mask.setValue(3)
        self.slider_mask.setSingleStep(2)
        self.slider_mask.setTickInterval(2)

        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(3)
        self.tableWidget.resizeRowsToContents()
        self.tableWidget.resizeColumnsToContents()
        self.slider_mask.valueChanged.connect(self.sliderValue)

        self.slider_mask_2.setMinimum(3)
        self.slider_mask_2.setMaximum(5)
        self.slider_mask_2.setValue(3)
        self.slider_mask_2.setSingleStep(2)
        self.slider_mask_2.setTickInterval(2)

        self.tableWidget_2.setColumnCount(3)
        self.tableWidget_2.setRowCount(3)
        self.tableWidget_2.resizeRowsToContents()
        self.tableWidget_2.resizeColumnsToContents()
        self.slider_mask_2.valueChanged.connect(self.sliderValue2)

    def CheckRadioCheck(self):
        if self.dic_path != '':
            if self.radioButton_2.isChecked():
                self.btn2Png.setEnabled(True)
                self.btnSave.setEnabled(False)
                self.btn2Bmp.setEnabled(False)
            elif self.radioButton_3.isChecked():
                self.btn2Png.setEnabled(True)
                self.btnSave.setEnabled(False)
                self.btn2Bmp.setEnabled(False)
            elif self.radioButton.isChecked():
                self.btn2Png.setEnabled(False)
                self.btnSave.setEnabled(True)
                self.btn2Bmp.setEnabled(True)

    def getImage(self):
        print(os.path.dirname(os.path.abspath(__file__)))
        self.dic_path = os.path.dirname(os.path.abspath(__file__))

        fname = QFileDialog.getOpenFileName(self, '選擇檔愛', self.dic_path, "Image files (*.jpg *.ppm *.bmp)")

        if fname != ('', ''):
            self.basename = ntpath.basename(fname[0])
            self.filename = fname[0]
            extension_type = self.filename.split('.')[1]
            self.first = fname[0].split('.')[0]
            # print(fname)
            self.label_ext.setText(extension_type)
            self.btnSave.setEnabled(True)
            self.btn2Bmp.setEnabled(True)
            # self.btn2Gray.setEnabled(True)
            self.btn2Haar.setEnabled(True)
            self.actionhistogram_equalization.setEnabled(True)
            self.actionGaussian_Noise.setEnabled(True)

            self.actionSaveFile.setEnabled(True)
            self.label_input.clear()
            self.label_output.clear()
            self.label_noise_dis.clear()
            print(self.filename)
            if extension_type == 'jpg' or extension_type == 'bmp':
                self.cv_img = Image.open(self.filename)

                cv_img_input = cv2.cvtColor(numpy.asarray(self.cv_img), cv2.COLOR_BGR2GRAY)
                cv_img_input = Image.fromarray(cv_img_input)
                pix = self.pil2pixmap(cv_img_input)
            if extension_type == 'ppm':
                # PIL to opencv
                self.cv_img = Image.open(self.filename)  # PIL read
                self.cv_img_output = cv2.cvtColor(numpy.asarray(self.cv_img),
                                                  cv2.COLOR_BGR2RGB)  # numpy to opencv  (PIL to opencv,getopencv)
                cv_img_input = cv2.cvtColor(numpy.asarray(self.cv_img), cv2.COLOR_BGR2GRAY)
                cv_img_input = Image.fromarray(cv_img_input)
                pix = self.pil2pixmap(cv_img_input)

            # PIL to opencv
            self.cv_img_output = cv2.cvtColor(np.array(self.cv_img), cv2.COLOR_BGR2RGB)

            self.label_input.setScaledContents(True)
            self.label_input.setPixmap(pix)
            self.label_height.setText(str(pix.height()))
            self.label_width.setText(str(pix.width()))
            self.RGB2Gray()

    def saveImage(self):
        name = QFileDialog.getSaveFileName(self, '儲存檔案', "", "Image files (*.jpg *.png *.bmp)")

        if name != ('', ''):
            extension_type = name[0].split('.')[1]

            self.first = name[0].split('.')[0]  # 供其他histogram儲存檔案路徑
            if self.radioButton.isChecked():  # 如果是要儲存output輸出的話
                self.btn2Png.setEnabled(False)
                if len(self.cv_img_output.shape) == 2:
                    saveImage = Image.fromarray(self.cv_img_output)
                    pix = self.pil2pixmap(saveImage)

                    # 通道為3個的影像使用此方法儲存
                else:
                    saveImage = Image.fromarray(cv2.cvtColor(self.cv_img_output, cv2.COLOR_BGR2RGB))
                    print(saveImage.mode)
                    pix = self.pil2pixmap(saveImage)

                saveImage.save(name[0])
                self.label_output.setScaledContents(True)
                self.label_output.setPixmap(pix)

    def saveBmp(self):
        savefile = self.filename.split('.')[0]

        if savefile:
            if self.cv_img_output is not None:
                print(self.cv_img_output.shape)
                print(len(self.cv_img_output.shape))
                # 灰階圖片使用此方法儲存
                if len(self.cv_img_output.shape) == 2:
                    saveImage = Image.fromarray(self.cv_img_output)
                    pix = self.pil2pixmap(saveImage)  # 這邊需要將cv2轉pixel

                # 通道為3個的影像使用此方法儲存
                else:
                    saveImage = Image.fromarray(cv2.cvtColor(self.cv_img_output, cv2.COLOR_BGR2RGB))
                    print(saveImage.mode)
                    pix = self.pil2pixmap(saveImage)
                # 上面將opencv轉為PIL，再用PIL存檔
                saveImage.save(savefile + '.bmp')
                self.label_output.setScaledContents(True)
                self.label_output.setPixmap(pix)

    def savePng(self):
        print('有觸發PNG按鈕')
        if self.radioButton_2.isChecked():
            if self.PIL_im == '':
                self.label_message.setText('得先產生直方圖')

            else:
                self.PIL_im.save(self.first + 'H.png')
                self.label_message.setText('儲存成功')
        if self.radioButton_3.isChecked():
            if self.Gaussian_im == '':
                self.label_message.setText('得先計算高斯雜訊圖')

            else:
                self.Gaussian_im.save(self.first + 'GN.png')
                self.label_message.setText('儲存成功')

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.lpljuhdata, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(convert_to_Qt_format)

    def pil2pixmap(self, im):

        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
        elif im.mode == "L":
            im = im.convert("RGBA")
        # Bild in RGBA konvertieren, falls nicht bereits passiert
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def RGB2Gray(self):
        self.label_histogram.clear()
        if len(self.cv_img_output) != 0:
            self.label_message.setText('')
            output_shape = self.cv_img_output.shape
            if len(output_shape) != 2:
                print('走非灰階')
                gray = cv2.cvtColor(self.cv_img_output, cv2.COLOR_BGR2GRAY)
                self.cv_img_output = gray
                inner_output_shape = self.cv_img_output.shape
                self.drawBar(gray)
                if output_shape[1] % 2 != 0:
                    self.cv_img_output = self.addColumn(inner_output_shape, self.cv_img_output)

            elif len(output_shape) == 2:
                print('走灰階')
                print(self.cv_img_output)
                print(self.cv_img_output.shape)
                self.drawBar(self.cv_img_output)
                inner_output_shape = self.cv_img_output.shape
                if output_shape[1] % 2 != 0:
                    self.cv_img_output = self.addColumn(inner_output_shape, self.cv_img_output)

    def addColumn(self, output_shape, im):

        print('加值前:', im.shape)

        new_img = np.zeros([output_shape[0], output_shape[1] + 1], np.uint8)
        for i in range(output_shape[0]):
            add_value = im[i][-1]
            new_img[i] = np.add(new_img[i], np.append(im[i], [add_value], axis=0))
        print('加值後:', im.shape)
        print(new_img.shape)
        return new_img

    def drawBar(self, gray_input_image):
        # hist = cv2.calcHist([gray_input_image], [0], None, [256], [0, 256])
        # 字典表
        hist2 = gray_input_image.ravel()  # 扁平化

        # plt畫圖
        plt.title(self.basename + "'s histogram")
        plt.xlabel('pixels range')
        plt.ylabel('number of pixel')
        # plt.xticks(hist2[0],range(0,256))
        plt.hist(hist2.ravel(), 256, [0, 256])

        # 使用內存將影像寫到output
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.PIL_im = Image.open(buf)
        pix = self.pil2pixmap(self.PIL_im)
        plt.close()
        buf.close()
        # 畫出output影像
        pix_output = self.pil2pixmap(Image.fromarray(gray_input_image))  # 這邊需要將cv2轉pixel
        self.label_output.setScaledContents(True)
        self.label_output.setPixmap(pix_output)
        # 畫出直方圖
        self.label_histogram.setScaledContents(True)
        self.label_histogram.setPixmap(pix)

    def Guassianbtnclick(self):
        inputSD = self.input_SD.text()
        self.label_change_hist.setText("Gaussian Noise")
        if inputSD != '':
            if inputSD.isdigit():
                standard = int(inputSD)
                if 0 < standard < 10:
                    if self.cv_img_output is not None:
                        self.calculateGuassian(standard, self.cv_img_output)

                else:
                    self.label_message.setText('請輸入1~10的標準差')

            else:
                self.label_message.setText('請輸入數字')
        else:
            self.label_message.setText('請輸入點什麼')

    # 計算z1,z2
    def calculateGuassian(self, standard, im):
        hist = {}

        shape = self.cv_img_output.shape
        new_img = np.zeros((shape[0], shape[1]), np.int8)
        if len(shape) == 2:  # 代表灰階
            for i in range(shape[0]):
                for j in range(0, len(self.cv_img_output[i]), 2):
                    r = random.uniform(0, 1)
                    phi = random.uniform(0, 1)
                    nlog = 2 * np.sqrt(-2 * np.log(r))
                    double_pi = math.pi * 2 * phi
                    z1 = np.cos(double_pi) * nlog
                    z2 = np.sin(double_pi) * nlog
                    z1 = round(z1, 1)
                    z2 = round(z2, 1)
                    new_img[i][j] = self.GuassianBound(self.cv_img_output[i][j] + z1)
                    new_img[i][j + 1] = self.GuassianBound(self.cv_img_output[i][j + 1] + z2)

                    if hist.get(z1) is None:
                        hist[z1] = 1
                    elif hist.get(z1) is not None:
                        hist[z1] += 1
                    elif hist.get(z2) is None:
                        hist[z2] = 1
                    else:
                        hist[z2] += 1
            plt.title("Gaussian's Noise distribution")
            plt.bar(hist.keys(), hist.values())
            # 用內存寫到label上
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.Gaussian_im = Image.open(buf)
            pix = self.pil2pixmap(self.Gaussian_im)
            plt.close()
            buf.close()

            pix_output = self.pil2pixmap(Image.fromarray(new_img))  # 這邊需要將cv2轉pixel
            self.label_output.setScaledContents(True)
            self.label_output.setPixmap(pix_output)
            self.label_noise_dis.setScaledContents(True)
            self.label_noise_dis.setPixmap(pix)

    def read2Haar(self):  # 讀入的圖片要做Haar轉換

        if len(self.cv_img_output) != 0 and self.input_depth.text() != '':
            str_depth = self.input_depth.text()
            # try:
            self.depth = int(str_depth)
            self.Haar(self.depth)
            # except:
            #     self.label_message.setText('請輸入層數為數字')

    def Haar(self, depth):
        img = cv2.resize(self.cv_img_output, (512, 512))
        height, width = img.shape
        print(height, width)
        # 產生新畫布
        tmp = np.zeros((height, width), np.uint8)
        wave = np.zeros((height, width), np.uint8)
        img2 = img
        # depth = int(input('輸入層數：'))

        depth_count = 1
        while depth_count <= depth:

            new_width = int(width / depth_count)
            new_height = int(height / depth_count)

            print(new_width)
            print(new_height)

            for i in range(new_height):
                for j in range(0, int(new_width / 2)):
                    tmp[i][j] = ((int(img2[i][j * 2]) + int(img2[i][j * 2 + 1])) / 2)
                    tmp[i][j + (int(new_width / 2))] = int((int(img2[i][j * 2]) - int(img2[i][j * 2 + 1])) / 2)

            for i in range(0, int(new_height / 2)):
                for j in range(new_width):
                    wave[i][j] = int((int(tmp[i * 2][j]) + int(tmp[i * 2 + 1][j])) / 2)
                    wave[i + (int(new_height / 2))][j] = int((int(tmp[i * 2][j]) - int(tmp[i * 2 + 1][j])) / 2)

            depth_count += 1

            img2 = wave

        cv_img_output = img2

        PIL_Haar = Image.fromarray(cv_img_output)
        filename = self.filename.split('.')[0].split('/')[-1]
        if not os.path.exists('./haar/'):
            os.makedirs('./haar/')
        PIL_Haar.save('./haar/' + filename + str(depth) + 'transform.png')
        self.pix_output_harr = self.pil2pixmap(PIL_Haar)  # 這邊需要將cv2轉pixel
        self.label_output.setScaledContents(True)
        self.label_output.setPixmap(self.pix_output_harr)

    def gray_equalization(self):
        self.label_change_hist.setText("histogram equalization")
        np.set_printoptions(suppress=True)
        # 讀取圖檔
        # img = cv2.imread(r'C:\Users\syaun\Pictures\file_example_JPG_100kB.jpg')
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.cv_img_output
        print(gray.shape)
        MN = gray.shape[0] * gray.shape[1]
        print("MN %d" % MN)
        print(gray.ravel())
        output = gray.ravel()

        hist = [0 for _ in range(256)]
        hist = np.array(hist)

        for i in range(len(output)):
            hist[output[i]] = hist[output[i]] + 1
        # print(type(hist))
        #
        # print(hist)
        # print(type(hist))

        cumulative_hist = [0 for _ in range(256)]
        cumulative_hist = np.array(cumulative_hist)
        # cumulative
        min = 0
        for i in range(256):
            if i == 0:
                cumulative_hist[i] = hist[i]
            else:
                cumulative_hist[i] = cumulative_hist[i - 1] + hist[i]

            if min == 0 and hist[i] != 0:
                min = hist[i]
            elif min != 0 and hist[i] < min and hist[i] != 0:
                min = hist[i]

        new_list = [0 for _ in range(256)]
        for i in range(256):
            tg = round((cumulative_hist[i] - min) / (MN - min) * 255)
            new_list[i] = tg
        gray2 = gray
        for row in range(gray.shape[0]):
            for col in range(gray.shape[1]):
                gray2[row][col] = new_list[gray[row][col]]
        plt.hist(gray2.ravel(), 256, (0, 256))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.equalization = Image.open(buf)
        pix = self.pil2pixmap(self.equalization)
        plt.close()
        buf.close()
        self.cv_img_output = gray2
        pix_output = self.pil2pixmap(Image.fromarray(gray2))  # 這邊需要將cv2轉pixel
        self.label_output.setScaledContents(True)
        self.label_output.setPixmap(pix_output)
        self.label_noise_dis.setScaledContents(True)
        self.label_noise_dis.setPixmap(pix)

    def GuassianBound(self, value):
        if value < 0:
            value = 0
        elif value > 255:
            value = 255
        return int(value)

    def clearLabel(self):
        self.label_histogram.clear()
        self.label_output.clear()
        self.label_noise_dis.clear()

    def smooth(self):
        # 客製化kernel
        state1 = True
        state2 = True
        mask_tb = self.tableWidget
        table_column = mask_tb.columnCount()
        table_row = mask_tb.rowCount()
        mask_tb2 = self.tableWidget_2
        table_column2 = mask_tb2.columnCount()
        table_row2 = mask_tb2.rowCount()

        want_pow = table_column ** 2
        want_pow2 = table_row2 ** 2
        print(table_column)
        print(table_row)
        list1 = []
        list2 = []
        for i in range(table_column):
            for j in range(table_row):
                if self.tableWidget.item(i, j) == None:
                    state2=False
                else:
                    try:
                        list1.append(int(self.tableWidget.item(i, j).text()))
                    except:
                        print('無法轉換')

        for i in range(table_column2):
            for j in range(table_row2):
                if self.tableWidget_2.item(i, j) == None:
                    state2=False
                else:
                    try:
                        list2.append(int(self.tableWidget_2.item(i, j).text()))
                    except:
                        print('無法轉換')

        if len(list1) == want_pow:
            self.kernel = np.array(list1).reshape(table_column, table_row)
        if len(list2) == want_pow2:
            self.edge_kernel = np.array(list2).reshape(table_column2, table_row2)
        # 讀取圖片
        if self.cv_img != '' and self.kernel != '' and self.edge_kernel != '' and state1 and state2:
            kernel = self.kernel
            edge_kernel = self.edge_kernel
            kernel_sum = self.kernel.sum()
            kernel_size = kernel.shape[0]
            need_padding = kernel.shape[0] // 2
            edge_kernel_size = edge_kernel.shape[0]
            edge_need_padding = edge_kernel.shape[0] // 2

            img = cv2.cvtColor(numpy.asarray(self.cv_img), cv2.COLOR_BGR2GRAY)
            print('轉完後的shape', img.shape)
            # img = cv2.imread('./Lenna.jpg', 0)
            clone_img = np.zeros((img.shape[0], img.shape[1]))
            clone_padding = np.pad(img, [need_padding, need_padding], 'constant').copy()
            print(clone_padding.shape)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    convol = (clone_padding[i:i + kernel_size, j:j + kernel_size] * kernel).sum()
                    convol = convol // kernel_sum
                    clone_img[i, j] = convol
            print('第一張圖')
            for_edge_detection_img = clone_img.copy()  # 經過smooth的img
            edge_imgH, edge_imgW = img.shape[:2]

            clone_img = Image.fromarray(clone_img)
            pix_out = self.pil2pixmap(clone_img)

            self.label_output.setScaledContents(True)
            self.label_output.setPixmap(pix_out)
            #
            # # 拉布拉斯filter
            laplacian = np.array((
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]), dtype="int")

            # Sobel的X和Y
            hor_sobel = self.edge_kernel
            print('horizon')
            print(hor_sobel)
            # print('sobel',hor_sobel)
            # hor_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "int")
            ver_sobel = np.rot90(hor_sobel)
            # ver_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], "int")

            # Sobel結果儲存位置
            SobelX = np.zeros((edge_imgH, edge_imgW), int)
            SobelY = np.zeros((edge_imgH, edge_imgW), int)
            Lablace = np.zeros((edge_imgH, edge_imgW), int)

            after_padding = np.pad(for_edge_detection_img, [edge_need_padding, edge_need_padding], mode='constant')
            print('after padding', after_padding.shape)

            for i in range(for_edge_detection_img.shape[0]):
                for j in range(for_edge_detection_img.shape[1]):
                    roi = after_padding[i:i + edge_kernel_size, j:j + edge_kernel_size]
                    # print(roi.shape)
                    SobelX[i][j] = (roi * hor_sobel).sum()
                    SobelY[i][j] = (roi * ver_sobel).sum()
                    # Lablace[i, j] = (roi * laplacian).sum()

            # 強度
            G = np.abs(SobelX) + np.abs(SobelY)
            angle = np.arctan2(SobelY, SobelX)
            self.edge_detection = Image.fromarray(G)
            pix = self.pil2pixmap(self.edge_detection)

            self.label_change_hist.setText('edge detection output')
            self.label_noise_dis.setScaledContents(True)
            self.label_noise_dis.setPixmap(pix)
        else:
            self.label_message.setText('未選圖或未填選smooth、edge的filter')


# class MyThread(threading.Thread):
#     def __init__(self, queue, num):
#         threading.Thread.__init__(self)
#         self.num = num
#         self.queue = queue
#
#     def run(self):
#         # while self.queue.qsize() >0 :
#         r = random.uniform(0, 1)
#         nlog =


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
