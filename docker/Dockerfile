# 使用官方 Python 3.9 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装依赖库
RUN apt-get update && apt-get install -y \
    git \
    portaudio19-dev \
    python3-pyaudio \
    sox \
    pulseaudio \
    libsox-fmt-all \
    ffmpeg \
    wget \
    swig \
    libpcre3 \
    libpcre3-dev \
    libatlas-base-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 克隆项目仓库
RUN git clone https://github.com/lzxaini/wukong-robot.git .

# 安装 PyAudio
RUN pip install pyaudio

# 安装 Python 依赖
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# 安装 wukong-contrib
RUN mkdir -p $HOME/.wukong \
    && cd $HOME/.wukong \
    && git clone https://github.com/lzxaini/wukong-contrib.git contrib \
    && pip install -r contrib/requirements.txt

# 下载并编译 snowboy
RUN wget https://wzpan-1253537070.cos.ap-guangzhou.myqcloud.com/misc/snowboy.tar.bz2 \
    && tar -xvjf snowboy.tar.bz2 \
    && cd snowboy/swig/Python3 \
    && make \
    && cp _snowboydetect.so /app/snowboy/

# 暴露端口
EXPOSE 5001

# 设置 ENTRYPOINT
ENTRYPOINT ["python", "wukong.py"]
