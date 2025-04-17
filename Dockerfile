# Base image với CUDA 12.6 và Python (từ PyTorch)
FROM pytorch/pytorch:2.3.0-cuda12.6-cudnn8-runtime

# Set thư mục làm việc
WORKDIR /app

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy toàn bộ code project vào trong container
COPY . /app

# Cài đặt thư viện Python từ file requirements.txt
RUN pip install -r requirements.txt

# (Tuỳ chọn) mở port cho tensorboard, jupyter...
EXPOSE 6006

# Lệnh mặc định khi container chạy
CMD ["bash"]
