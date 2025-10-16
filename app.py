from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# API 端點：Agent 或使用者會呼叫這個 URL
@app.route('/api/kline', methods=['GET'])
def get_kline_chart():
    # 獲取股票代碼參數，例如 ?symbol=2330.TW
    symbol = request.args.get('symbol')
    
    if not symbol:
        # 如果沒有提供股票代碼，返回錯誤
        return jsonify({'error': 'Missing required parameter: symbol'}), 400

    try:
        # --- 數據獲取 ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90) # 抓取近 90 天數據
        
        # 使用 yfinance 獲取 OHLCV 數據
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
        
        if data.empty:
            return jsonify({'error': f"無法獲取 {symbol} 的數據。請檢查代碼或時間範圍。"}), 404

        # --- 均線計算 ---
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()

        # --- 繪圖邏輯 ---
        add_plots = [
            mpf.make_addplot(data['MA5'], color='blue', label='MA5'),
            mpf.make_addplot(data['MA20'], color='red', label='MA20')
        ]
        
        # 使用 BytesIO 作為內存緩衝區來儲存圖片
        buffer = io.BytesIO()
        
        mpf.plot(
            data, 
            type='candle', 
            volume=True, 
            addplot=add_plots, 
            style='yahoo',
            title=f'{symbol} K-Line Chart (MA5 & MA20)',
            # 將圖片儲存到緩衝區
            savefig=dict(fname=buffer, format='png', dpi=100)
        )
        buffer.seek(0) # 將指標移回開頭

        # --- 返回圖片 ---
        # 直接以 PNG 格式返回圖片的二進制數據
        return send_file(
            buffer, 
            mimetype='image/png', 
            as_attachment=False # 設置為 False 讓瀏覽器直接顯示
        )

    except Exception as e:
        app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
        # 返回通用的 500 錯誤
        return jsonify({'error': f"服務器內部錯誤。"}), 500

# 根路由，用於健康檢查
@app.route('/')
def home():
    return 'Stock K-Line Chart API is running. Call /api/kline?symbol=STOCK_CODE to get chart.'

if __name__ == '__main__':
    # 僅用於本地開發測試
    app.run(host='0.0.0.0', port=5000)
