from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
import yfinance as yf
from datetime import datetime, timedelta
import ta
import time
from functools import lru_cache  # 【改用】內建快取，不與 curl_cffi 衝突

app = Flask(__name__)

# --- 【關鍵優化】使用記憶體快取替代 requests-cache ---
# 透過將資料抓取獨立成一個帶有快取的函式，相同參數在 10 分鐘內只會真正執行一次
# maxsize=128 代表最多快取 128 支不同的股票
@lru_cache(maxsize=128)
def fetch_stock_data(symbol, start_date_str, end_date_str):
    """
    因為 lru_cache 必須使用可雜湊(hashable)的參數，
    所以我們把 datetime 物件轉成字串（YYYY-MM-DD）作為參數。
    """
    # 讓 yfinance 內建的 curl_cffi 自行發揮，不帶自訂 session
    data = yf.download(
        symbol, 
        start=start_date_str, 
        end=end_date_str, 
        interval='1d', 
        progress=False,
        multi_level_index=False
    )
    return data

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

    # 自動處理台股格式
    if symbol.isdigit() and len(symbol) == 4:
        symbol = f"{symbol}.TW"

    # 為了讓快取精準，我們把時間固定到「天」（不含時分秒）
    # 這樣今天之內發送的請求，字串參數都會完全相同，成功觸發快取
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    data = pd.DataFrame()
    max_retries = 3
    
    # 具備漸進式退避的重試機制
    for attempt in range(max_retries):
        try:
            # 呼叫帶有快取的函式
            data = fetch_stock_data(symbol, start_date_str, end_date_str)
            if not data.empty:
                break
        except Exception as yf_err:
            app.logger.warning(f"第 {attempt + 1} 次嘗試獲取 {symbol} 失敗: {yf_err}")
            
            # 如果失敗了，有可能是快取了壞資料，清除快取重試
            fetch_stock_data.cache_clear()
            
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))  # 失敗時拉長等待時間 (3s, 6s)
            else:
                return jsonify({'error': '受到 Yahoo 頻率限制或網路阻擋，請稍後再試。', 'details': str(yf_err)}), 429

    if data.empty:
        return jsonify({'error': f"無法獲取 {symbol} 的數據。請檢查代碼或時間範圍。"}), 404

    try:
        # 複製一份資料避免改動到快取記憶體中的原始 Dataframe
        df = data.copy()
        
        # --- 數據準備與指標計算 ---
        df = prepare_data(df, symbol)
        if df.empty:
             return jsonify({'error': f"{symbol} 數據在清理後為空。"}), 404
             
        df = calculate_indicators(df)

        # --- 繪圖設定 ---
        add_plots = []
        add_plots.append(mpf.make_addplot(df['MA5'], color='blue', label='MA5', panel=0))
        add_plots.append(mpf.make_addplot(df['MA20'], color='red', label='MA20', panel=0))

        add_plots.append(mpf.make_addplot(df['K'], panel=1, color='purple', linestyle='-', label='K', ylabel='Stochastics'))
        add_plots.append(mpf.make_addplot(df['D'], panel=1, color='orange', linestyle='-', label='D'))
        add_plots.append(mpf.make_addplot([80]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))

        add_plots.append(mpf.make_addplot(df['MACD'], panel=2, color='green', label='MACD', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(df['Signal'], panel=2, color='red', label='Signal'))
        colors = ['red' if v >= 0 else 'green' for v in df['Hist']]
        add_plots.append(
            mpf.make_addplot(df['Hist'], type='bar', panel=2, color=colors, alpha=0.6, secondary_y=False)
        )
        add_plots.append(mpf.make_addplot([0]*len(df), panel=2, color='black', linestyle=':', alpha=0.5))

        add_plots.append(mpf.make_addplot(df['ADX'], panel=3, color='black', linestyle='-', label='ADX', ylabel='DMI/ADX'))
        add_plots.append(mpf.make_addplot(df['DMI+'], panel=3, color='lime', linestyle='-', label='+DI'))
        add_plots.append(df.make_addplot(df['DMI-'], panel=3, color='red', linestyle='-', label='-DI'))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=3, color='gray', linestyle=':', alpha=0.5))

        # --- 繪製 K 線圖 ---
        buffer = io.BytesIO()
        mpf.plot(
            df, 
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
