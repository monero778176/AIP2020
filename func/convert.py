


from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def put_img_to_label(rgb_image,put_obj):
    '''
    將 cv image 放置到 label
    - rgb_image: 轉換 rgb np image
    - put_obj: ui label元件
    '''

    if len(rgb_image.shape)==2: # grayscale
        rgb_image = np.stack((rgb_image,)*3, axis=-1)
        
    height, width, channel = rgb_image.shape
    bytesPerLine = 3 * width # For a 3-channel RGB image
    q_image = QImage(rgb_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    qpixmap = QPixmap.fromImage(q_image)

    lbw,lbh = put_obj.size().width(),put_obj.size().height()  # 畫布大小取得

    scale_pixmap = qpixmap.scaled(lbw,lbh,Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)

    put_obj.setPixmap(scale_pixmap)


def show_matplotlib_plot(np_data,plot_type,ui_obj):
    """
    生成 Matplotlib 圖形並將其顯示在 QLabel 上
    """
    # 1. 創建 Matplotlib 圖形 (Figure)
    fig = Figure(figsize=(6, 4),dpi=100)
    ax = fig.add_subplot(111)

    # 生成一些資料
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x)

    # 繪製圖形
    # ax.plot(x, y)

    if plot_type=='bar':
        np_data = np_data.reshape(-1)  # 直方圖
        ax.bar(range(1,257),np_data)

    '''option: plot 的其他資訊'''
    ax.set_title("Gray Histogram")
    # ax.set_xlabel("X軸")
    # ax.set_ylabel("Y軸")
    # ax.grid(True)

    # 2. 將圖形渲染到緩衝區
    buffer = io.BytesIO()
    # 使用 FigureCanvasAgg 將圖形繪製成 PNG 格式並寫入緩衝區
    FigureCanvasAgg(fig).print_png(buffer)
    
    # 3. 從緩衝區創建 QPixmap
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), 'png')
    
    # 4. 調整 QLabel 和 QPixmap 的大小以確保完整顯示
    # 這裡可以根據 QLabel 的大小來縮放圖片
    scaled_pixmap = pixmap.scaled(ui_obj.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    # 5. 將 QPixmap 設定到 QLabel
    ui_obj.setPixmap(scaled_pixmap)

    