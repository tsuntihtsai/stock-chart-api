from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
import yfinance as yf
from datetime import datetime, timedelta
import ta
import time
import requests_cache  # 【新增】用於快取請求，減少對 Yahoo 的直接存取

app = Flask(__name__)

# --- 【關鍵優化 1】設定 Requests 快取與偽裝瀏覽器標頭 ---
# 建立一個有效期限為 10 分鐘的快取，這能極大地防止因為前端重複重整而觸發的 Rate Limit
session = requests_cache.CachedSession(
    'yfinance_cache',
    expire_after=600,  # 快取 10 分鐘 (600秒)
    allowable_methods=['GET']
)
# 偽裝成一般的桌面瀏覽器
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# --- 輔助函式：清理和準備數據 ---
def prepare_data(data, symbol):
    if data.index.name is None:
        data.index.name = 'Date'
        
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])
    
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlc_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

    data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return data

# --- 輔助函式：計算所有技術指標 ---
def calculate_indicators(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    stoch = ta.momentum.StochasticOscillator(
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        window=14, 
        smooth_window=3
    )
    data['K'] = stoch.stoch().dropna()
    data['D'] = stoch.stoch_signal().dropna()
    
    macd = ta.trend.MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd().dropna()
    data['Signal'] = macd.macd_signal().dropna()
    data['Hist'] = macd.macd_diff().dropna()

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

    # 針對台灣股市代碼自動補尾綴（例如 2330 變成 2330.TW）
    if symbol.isdigit() and len(symbol) == 4:
        symbol = f"{symbol}.TW"

    # --- 【關鍵優化 2】具備重試機制的數據獲取 ---
    data = pd.DataFrame()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用我們設定好 User-Agent 與快取的 session
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval='1d', 
                progress=False,
                multi_level_index=False,
                session=session  # 【新增】套用快取機制
            )
            if not data.empty:
                break
        except Exception as yf_err:
            app.logger.warning(f"第 {attempt + 1} 次嘗試獲取 {symbol} 失敗: {yf_err}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # 漸進式延遲：2秒、4秒...
            else:
                return jsonify({'error': '受到 Yahoo 頻率限制，請稍後再試。', 'details': str(yf_err)}), 429

    if data.empty:
        return jsonify({'error': f"無法獲取 {symbol} 的數據。請檢查代碼或時間範圍。"}), 404

    try:
        # --- 數據準備與指標計算 ---
        data = prepare_data(data, symbol)
        if data.empty:
             return jsonify({'error': f"{symbol} 數據在清理後為空。"}), 404
             
        data = calculate_indicators(data)

        # --- 繪圖設定 ---
        add_plots = []
        add_plots.append(mpf.make_addplot(data['MA5'], color='blue', label='MA5', panel=0))
        add_plots.append(mpf.make_addplot(data['MA20'], color='red', label='MA20', panel=0))

        add_plots.append(mpf.make_addplot(data['K'], panel=1, color='purple', linestyle='-', label='K', ylabel='Stochastics'))
        add_plots.append(mpf.make_addplot(data['D'], panel=1, color='orange', linestyle='-', label='D'))
        add_plots.append(mpf.make_addplot([80]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot([20]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))

        add_plots.append(mpf.make_addplot(data['MACD'], panel=2, color='green', label='MACD', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(data['Signal'], panel=2, color='red', label='Signal'))
        colors = ['red' if v >= 0 else 'green' for v in data['Hist']]
        add_plots.append(
            mpf.make_addplot(data['Hist'], type='bar', panel=2, color=colors, alpha=0.6, secondary_y=False)
        )
        add_plots.append(mpf.make_addplot([0]*len(data), panel=2, color='black', linestyle=':', alpha=0.5))

        add_plots.append(mpf.make_addplot(data['ADX'], panel=3, color='black', linestyle='-', label='ADX', ylabel='DMI/ADX'))
        add_plots.append(mpf.make_addplot(data['DMI+'], panel=3, color='lime', linestyle='-', label='+DI'))
        add_plots.append(mpf.make_addplot(data['DMI-'], panel=3, color='red', linestyle='-', label='-DI'))
        add_plots.append(mpf.make_addplot([20]*len(data), panel=3, color='gray', linestyle=':', alpha=0.5))

        # --- 繪製 K 線圖並儲存到緩衝區 ---
        buffer = io.BytesIO()
        mpf.plot(
            data, 
            type='candle', 
            volume=True, 
            addplot=add_plots, 
            style='yahoo',
            title=f'{symbol} K-Line Chart (MAs, KD, MACD, DMI)',
            figratio=(16, 12),
            savefig=dict(fname=buffer, format='png', dpi=100)
        )
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png', as_attachment=False)

    except Exception as e:
        app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
        return jsonify({'error': "服務器內部錯誤或數據處理失敗。", 'details': str(e)}), 500

@app.route('/')
def home():
    return 'Stock K-Line Chart API is running. Call /api/kline?symbol=STOCK_CODE to get chart.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
