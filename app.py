from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
import yfinance as yf
from datetime import datetime, timedelta
import ta
import time  # 【新增】用於處理 Rate Limit

app = Flask(__name__)

# --- 輔助函式：清理和準備數據 ---
def prepare_data(data, symbol):
    # 確保索引名稱存在
    if data.index.name is None:
        data.index.name = 'Date'
        
    # 移除 yfinance 可能帶來的 'Adj Close' 欄位
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    
    # 強制轉換 OHLCV 欄位為 float 類型，處理 yfinance 可能帶來的資料型態問題
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlc_cols:
        # 使用 errors='coerce' 將無法轉換的值設為 NaN
        data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

    # 移除任何主要 OHLCV 欄位為 NaN 的行
    data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    return data

# --- 輔助函式：計算所有技術指標 ---
def calculate_indicators(data):
    
    # 1. 均線 (MA5, MA20)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # 2. KD 指標 (Stochastics Oscillator, 14, 3, 3)
    stoch = ta.momentum.StochasticOscillator(
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        window=14, 
        smooth_window=3
    )
    data['K'] = stoch.stoch().dropna()
    data['D'] = stoch.stoch_signal().dropna()
    
    # 3. MACD 指標 (12, 26, 9)
    macd = ta.trend.MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd().dropna()
    data['Signal'] = macd.macd_signal().dropna()
    data['Hist'] = macd.macd_diff().dropna()

    # 4. DMI 指標 (ADX, DMI+, DMI-, 14)
    adx = ta.trend.ADX(data['High'], data['Low'], data['Close'], window=14)
    data['ADX'] = adx.adx().dropna()
    data['DMI+'] = adx.adx_pos().dropna()
    data['DMI-'] = adx.adx_neg().dropna()
    
    return data

# --- API 端點 ---
@app.route('/api/kline', methods=['GET'])
def get_kline_chart():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Missing required parameter: symbol'}), 400

    try:
        # 【解決 Rate Limit 關鍵點】: 在請求數據前加入延遲
        time.sleep(3) # 暫停 3 秒，防止連續請求被鎖
        
        # --- 數據獲取 ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90) # 抓取 6 個月數據

        # multi_level_index=False 解決欄位結構問題
        data = yf.download(
            symbol, 
            start=start_date, 
            end=end_date, 
            interval='1d', 
            progress=False,
            multi_level_index=False
        )

        if data.empty:
            return jsonify({'error': f"無法獲取 {symbol} 的數據。請檢查代碼或時間範圍。"}), 404

        # --- 數據準備與指標計算 ---
        data = prepare_data(data, symbol)
        
        # 再次檢查數據是否在清理後為空
        if data.empty:
             return jsonify({'error': f"{symbol} 數據在清理後為空。"}), 404
             
        data = calculate_indicators(data)


        # --- 繪圖設定 ---
        add_plots = []

        # MA5, MA20 (Panel 0: 主圖)
        add_plots.append(mpf.make_addplot(data['MA5'], color='blue', label='MA5', panel=0))
        add_plots.append(mpf.make_addplot(data['MA20'], color='red', label='MA20', panel=0))

        # KD 指標 (Panel 1)
        add_plots.append(mpf.make_addplot(data['K'], panel=1, color='purple', linestyle='-', label='K', ylabel='Stochastics'))
        add_plots.append(mpf.make_addplot(data['D'], panel=1, color='orange', linestyle='-', label='D'))
        add_plots.append(mpf.make_addplot([80]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot([20]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))

        # MACD 指標 (Panel 2)
        add_plots.append(mpf.make_addplot(data['MACD'], panel=2, color='green', label='MACD', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(data['Signal'], panel=2, color='red', label='Signal'))
        # 柱狀圖 (Hist) 
        colors = ['red' if v >= 0 else 'green' for v in data['Hist']]
        add_plots.append(
            mpf.make_addplot(data['Hist'], type='bar', panel=2, color=colors, alpha=0.6, secondary_y=False)
        )
        add_plots.append(mpf.make_addplot([0]*len(data), panel=2, color='black', linestyle=':', alpha=0.5)) # 零軸線

        # DMI/ADX 指標 (Panel 3)
        add_plots.append(mpf.make_addplot(data['ADX'], panel=3, color='black', linestyle='-', label='ADX', ylabel='DMI/ADX'))
        add_plots.append(mpf.make_addplot(data['DMI+'], panel=3, color='lime', linestyle='-', label='+DI'))
        add_plots.append(mpf.make_addplot(data['DMI-'], panel=3, color='red', linestyle='-', label='-DI'))
        add_plots.append(mpf.make_addplot([20]*len(data), panel=3, color='gray', linestyle=':', alpha=0.5)) # ADX 強弱線


        # --- 繪製 K 線圖並儲存到緩衝區 ---
        buffer = io.BytesIO()

        mpf.plot(
            data, 
            type='candle', 
            volume=True, 
            addplot=add_plots, 
            style='yahoo',
            title=f'{symbol} K-Line Chart (MAs, KD, MACD, DMI)',
            figratio=(16, 12), # 調整圖形比例以容納所有面板
            savefig=dict(fname=buffer, format='png', dpi=100)
        )
        buffer.seek(0)
        
        # --- 返回圖片 ---
        return send_file(
            buffer, 
            mimetype='image/png', 
            as_attachment=False
        )

    except Exception as e:
        # 使用 app.logger.error 記錄錯誤，並將錯誤信息返回給客戶端
        app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
        return jsonify({'error': "服務器內部錯誤或數據抓取失敗。", 'details': str(e)}), 500

# 根路由，用於健康檢查
@app.route('/')
def home():
    return 'Stock K-Line Chart API is running. Call /api/kline?symbol=STOCK_CODE to get chart.'

if __name__ == '__main__':
    # 僅用於本地開發測試
    app.run(host='0.0.0.0', port=5000)
