# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(976, 579)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(300, 10, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(640, 370, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(640, 410, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(640, 450, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_width = QtWidgets.QLabel(self.centralwidget)
        self.label_width.setGeometry(QtCore.QRect(750, 410, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_width.setFont(font)
        self.label_width.setText("")
        self.label_width.setObjectName("label_width")
        self.label_ext = QtWidgets.QLabel(self.centralwidget)
        self.label_ext.setGeometry(QtCore.QRect(750, 450, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_ext.setFont(font)
        self.label_ext.setText("")
        self.label_ext.setObjectName("label_ext")
        self.label_height = QtWidgets.QLabel(self.centralwidget)
        self.label_height.setGeometry(QtCore.QRect(750, 370, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_height.setFont(font)
        self.label_height.setText("")
        self.label_height.setObjectName("label_height")
        self.btnChoose = QtWidgets.QPushButton(self.centralwidget)
        self.btnChoose.setGeometry(QtCore.QRect(620, 20, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(14)
        self.btnChoose.setFont(font)
        self.btnChoose.setObjectName("btnChoose")
        self.btnSave = QtWidgets.QPushButton(self.centralwidget)
        self.btnSave.setGeometry(QtCore.QRect(790, 210, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(14)
        self.btnSave.setFont(font)
        self.btnSave.setObjectName("btnSave")
        self.label_input = QtWidgets.QLabel(self.centralwidget)
        self.label_input.setGeometry(QtCore.QRect(20, 50, 221, 181))
        self.label_input.setText("")
        self.label_input.setObjectName("label_input")
        self.label_output = QtWidgets.QLabel(self.centralwidget)
        self.label_output.setGeometry(QtCore.QRect(300, 50, 221, 181))
        self.label_output.setText("")
        self.label_output.setObjectName("label_output")
        self.btn2Bmp = QtWidgets.QPushButton(self.centralwidget)
        self.btn2Bmp.setGeometry(QtCore.QRect(790, 160, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(14)
        self.btn2Bmp.setFont(font)
        self.btn2Bmp.setObjectName("btn2Bmp")
        self.btn2Gray = QtWidgets.QPushButton(self.centralwidget)
        self.btn2Gray.setGeometry(QtCore.QRect(740, 20, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(12)
        self.btn2Gray.setFont(font)
        self.btn2Gray.setObjectName("btn2Gray")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 260, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_histogram = QtWidgets.QLabel(self.centralwidget)
        self.label_histogram.setGeometry(QtCore.QRect(20, 300, 231, 191))
        self.label_histogram.setText("")
        self.label_histogram.setObjectName("label_histogram")
        self.btnClear = QtWidgets.QPushButton(self.centralwidget)
        self.btnClear.setGeometry(QtCore.QRect(740, 90, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(14)
        self.btnClear.setFont(font)
        self.btnClear.setObjectName("btnClear")
        self.btnGaussianNoise = QtWidgets.QPushButton(self.centralwidget)
        self.btnGaussianNoise.setGeometry(QtCore.QRect(860, 20, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(12)
        self.btnGaussianNoise.setFont(font)
        self.btnGaussianNoise.setObjectName("btnGaussianNoise")
        self.label_SD = QtWidgets.QLabel(self.centralwidget)
        self.label_SD.setGeometry(QtCore.QRect(640, 330, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_SD.setFont(font)
        self.label_SD.setObjectName("label_SD")
        self.input_SD = QtWidgets.QLineEdit(self.centralwidget)
        self.input_SD.setGeometry(QtCore.QRect(710, 330, 113, 31))
        self.input_SD.setObjectName("input_SD")
        self.label_message = QtWidgets.QLabel(self.centralwidget)
        self.label_message.setGeometry(QtCore.QRect(650, 280, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_message.setFont(font)
        self.label_message.setText("")
        self.label_message.setObjectName("label_message")
        self.label_noise_dis = QtWidgets.QLabel(self.centralwidget)
        self.label_noise_dis.setGeometry(QtCore.QRect(290, 300, 231, 191))
        self.label_noise_dis.setText("")
        self.label_noise_dis.setObjectName("label_noise_dis")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(300, 260, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(16)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(620, 170, 151, 131))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 30, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 90, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.btn2Png = QtWidgets.QPushButton(self.centralwidget)
        self.btn2Png.setGeometry(QtCore.QRect(790, 260, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setPointSize(14)
        self.btn2Png.setFont(font)
        self.btn2Png.setObjectName("btn2Png")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 976, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/outline_logout_black_24dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExit.setIcon(icon)
        self.actionExit.setObjectName("actionExit")
        self.actionOpenFile = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/outline_file_open_black_24dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenFile.setIcon(icon1)
        self.actionOpenFile.setObjectName("actionOpenFile")
        self.actionSaveFile = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/outline_save_black_24dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionSaveFile.setIcon(icon2)
        self.actionSaveFile.setObjectName("actionSaveFile")
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.toolBar.addAction(self.actionOpenFile)
        self.toolBar.addAction(self.actionSaveFile)
        self.toolBar.addAction(self.actionExit)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Input"))
        self.label_2.setText(_translate("MainWindow", "Output"))
        self.label_3.setText(_translate("MainWindow", "Height"))
        self.label_4.setText(_translate("MainWindow", "Width"))
        self.label_5.setText(_translate("MainWindow", "Extension"))
        self.btnChoose.setText(_translate("MainWindow", "Choose File"))
        self.btnSave.setText(_translate("MainWindow", "Save File"))
        self.btn2Bmp.setText(_translate("MainWindow", "To bmp"))
        self.btn2Gray.setText(_translate("MainWindow", "Histograme\n"
"(Grayscale)"))
        self.label_6.setText(_translate("MainWindow", "Histogram"))
        self.btnClear.setText(_translate("MainWindow", "Clear"))
        self.btnGaussianNoise.setText(_translate("MainWindow", "Gaussian Noise"))
        self.label_SD.setText(_translate("MainWindow", "SD"))
        self.label_7.setText(_translate("MainWindow", "Noise Distribution"))
        self.groupBox.setTitle(_translate("MainWindow", "Type of output to Save"))
        self.radioButton.setText(_translate("MainWindow", "output image"))
        self.radioButton_2.setText(_translate("MainWindow", "Histogram"))
        self.radioButton_3.setText(_translate("MainWindow", "Gaussian Noise"))
        self.btn2Png.setText(_translate("MainWindow", "To png"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setToolTip(_translate("MainWindow", "Exit this application"))
        self.actionOpenFile.setText(_translate("MainWindow", "OpenFile"))
        self.actionOpenFile.setToolTip(_translate("MainWindow", "Open to choose file"))
        self.actionOpenFile.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSaveFile.setText(_translate("MainWindow", "SaveFile"))
        self.actionSaveFile.setShortcut(_translate("MainWindow", "Ctrl+S"))
import ui.resouce.toolbar1_rc
