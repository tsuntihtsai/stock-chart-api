from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta
import ta
import time
from FinMind.data import DataLoader  # 【改用】FinMind 財經資料庫

app = Flask(__name__)
api = DataLoader()

# 如果你有註冊 FinMind 帳號（免費），可以在這裡填入 Token 提高存取額度
# api.login_by_token(token="YOUR_FINMIND_TOKEN")

def prepare_data_fm(df):
    """將 FinMind 的資料格式轉換為 mplfinance 相容格式"""
    # FinMind 欄位對應轉換
    df = df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'max': 'High',
        'min': 'Low',
        'close': 'Close',
        'trading_volume': 'Volume'
    })
    
    # 設定時間索引
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 確保資料型態為 float
    ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlc_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def calculate_indicators(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    stoch = ta.momentum.StochasticOscillator(
        high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3
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

@app.route('/api/kline', methods=['GET'])
def get_kline_chart():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Missing required parameter: symbol'}), 400

    # 提取純數字（例如將 2330.TW 轉為 2330）
    stock_id = ''.join(filter(str.isdigit, symbol))
    if not stock_id:
        return jsonify({'error': '請輸入正確的台股代碼（例如 2330）'}), 400

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    try:
        # 從 FinMind 獲取台股日 K 線資料
        df_raw = api.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df_raw.empty:
            return jsonify({'error': f"FinMind 無法獲取股票代碼 {stock_id} 的數據。"}), 404

        # 資料轉換與指標計算
        df = prepare_data_fm(df_raw)
        df = calculate_indicators(df)

        # --- 繪圖設定 (與你原本的完全相同) ---
        add_plots = []
        add_plots.append(mpf.make_addplot(df['MA5'], color='blue', label='MA5', panel=0))
        add_plots.append(mpf.make_addplot(df['MA20'], color='red', label='MA20', panel=0))
        add_plots.append(mpf.make_addplot(df['K'], panel=1, color='purple', label='K', ylabel='Stochastics'))
        add_plots.append(mpf.make_addplot(df['D'], panel=1, color='orange', label='D'))
        add_plots.append(mpf.make_addplot([80]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot(df['MACD'], panel=2, color='green', label='MACD', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(df['Signal'], panel=2, color='red', label='Signal'))
        colors = ['red' if v >= 0 else 'green' for v in df['Hist']]
        add_plots.append(mpf.make_addplot(df['Hist'], type='bar', panel=2, color=colors, alpha=0.6, secondary_y=False))
        add_plots.append(mpf.make_addplot([0]*len(df), panel=2, color='black', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot(df['ADX'], panel=3, color='black', label='ADX', ylabel='DMI/ADX'))
        add_plots.append(mpf.make_addplot(df['DMI+'], panel=3, color='lime', label='+DI'))
        add_plots.append(mpf.make_addplot(df['DMI-'], panel=3, color='red', label='-DI'))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=3, color='gray', linestyle=':', alpha=0.5))

        buffer = io.BytesIO()
        mpf.plot(
            df, type='candle', volume=True, addplot=add_plots, style='yahoo',
            title=f'{stock_id} K-Line Chart (FinMind Data)',
            figratio=(16, 12), savefig=dict(fname=buffer, format='png', dpi=100)
        )
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=False)

    except Exception as e:
        app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
        return jsonify({'error': "服務器內部錯誤或數據處理失敗。", 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
