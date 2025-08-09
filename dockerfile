FROM python:3.11-slim

# 安裝必要套件 (X11)
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxrandr2 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libfreetype6 \
    libx11-xcb1 \
    libxcb-render0 \
    libxcb-shm0 \
    && rm -rf /var/lib/apt/lists/*

RUN  apt-get update && apt install -y libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxcb-icccm4

RUN apt update && apt install libegl1 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# 設定工作目錄
WORKDIR /app

# 安裝 Python 套件
COPY requirements2.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements2.txt

# 複製程式
COPY . .

# 預設執行
# CMD ["python", "app.py"]
CMD ["tail", "-f", "/dev/null"]
