# 1. Dùng Python 3.9 bản nhẹ nhất
FROM python:3.9-slim

# 2. Tạo thư mục làm việc
WORKDIR /app

# 3. Copy file thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ code VÀ FILE CSV vào trong image
COPY . .

# 5. Chạy file main
CMD ["python", "main.py"]