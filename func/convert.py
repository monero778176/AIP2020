


import json
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import cv2


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
    
    # 直接設置原始 QPixmap，縮放交由 QLabel 的 setScaledContents 屬性處理
    put_obj.setPixmap(qpixmap)


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



def add_gaussian_noise(image, mean=0, std=25):
    """
    向图像添加高斯噪声。

    Args:
        image: 要添加噪声的图像 (numpy数组)。
        mean: 高斯分布的均值。
        std:  高斯分布的标准差。

    Returns:
        添加了高斯噪声的图像。
    """
    # 生成与图像大小相同的随机噪声
    noise = np.random.normal(mean, std, image.shape)
    # 将噪声添加到图像中，并确保像素值在有效范围内
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_blur(img,json_data):
    get_type = json_data.get('filter')
    kernel_size = int(json_data.get('kernel_size'))
    if get_type=='gassian':
        img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    elif get_type=='median':
        img = cv2.medianBlur(img,(kernel_size,kernel_size),0)

    elif get_type=='bilateral':
        img = cv2.bilateralFilter(img,kernel_size,0,0)

    return img


def add_edge(img,json_data):
    get_type_mask = json_data.get('mask')  # 獲取名稱
    mask = json_data.get('mask_np')

    mask_name = get_type_mask.split(' ')[0]
    xy_axis = get_type_mask.split(' ')[-1].replace('(','').replace(')','').split('-')[0]
    print(f'獲得mask 名稱:{mask_name} 對應的 xy 軸:{xy_axis}')


    if mask_name=='Sobal' and xy_axis=='x':
        print(f'進入:{mask_name} {mask_name}-axis')
        img =  cv2.filter2D(img,0,kernel=np.array(mask))
    elif mask_name=='Sobal' and xy_axis=='y':
        print(f'進入:{mask_name} {mask_name}-axis')
        
        img =  cv2.filter2D(img,0,kernel=np.array(mask))
    elif mask_name=='Scharr' and xy_axis=='x':
        print(f'進入:{mask_name} {mask_name}-axis')

        img =  cv2.filter2D(img,0,kernel=np.array(mask))
    elif mask_name=='Scharr' and xy_axis=='y':
        print(f'進入:{mask_name} {mask_name}-axis')

        img =  cv2.filter2D(img,0,kernel=np.array(mask))

    return img


    