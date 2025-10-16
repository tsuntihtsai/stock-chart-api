# 使用輕量級的 Python 映像檔
FROM python:3.10-slim

# 設定環境變數，防止 Python 輸出緩衝，提高效率
ENV PYTHONUNBUFFERED 1

# 設定工作目錄
WORKDIR /app

# 複製依賴文件並安裝
COPY requirements.txt .
# 使用 --no-cache-dir 減少映像檔大小
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式程式碼
COPY . .

# 定義服務運行的端口
# Zeabur 會在部署時將外部流量映射到這個端口
ENV PORT 8080 
EXPOSE 8080

# 使用 waitress 作為生產環境的 WSGI 伺服器啟動應用程式
# 格式為: waitress-serve [HOST]:[PORT] [MODULE]:[VARIABLE]
CMD ["waitress-serve", "--listen=0.0.0.0:8080", "app:app"]
