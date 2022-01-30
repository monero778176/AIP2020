# AIP2020
本專案使用python-PyQt5做GUI，功能如下
<br>
## 基本操作
+ 讀取圖片
+ 另存新檔[可接受格式有:.bmp、.jpg、.png]
+ 圖片detail(width、height、extension)
+ 高斯雜訊分布圖
+ 灰階pixel分布圖

## 進階功能
+ Harr小波轉換
+ 加入高斯雜訊
+ 高斯模糊
+ 邊緣mask

關於mask的轉換並不是使用opencv套件做convolution，而是使用傳統的直接陣列中的mask去做convolution，因此速度上會較慢。