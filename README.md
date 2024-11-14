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
+ Haar小波轉換
+ 加入高斯雜訊
+ 高斯模糊
+ 邊緣mask

### Haar轉換範例圖
![2次轉換](haar/gray2transform.png)


### 程式執行
```
python -m venv aip_env
"aip_env/Script/activate"
pip install -r requirements.txt

python main.py
```



### icon來源
my_icon.ico: from [flaticon](https://www.flaticon.com/)