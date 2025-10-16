from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
import yfinance as yf
from datetime import datetime, timedelta
# 【新增】引入 ta 函式庫
import ta
import time # <-- 新增



app = Flask(__name__)

# ... (home 和 get_kline_chart 路由定義保持不變) ...

@app.route('/api/kline', methods=['GET'])
def get_kline_chart():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Missing required parameter: symbol'}), 400
    
    try:
            # 【新增延遲】: 讓服務暫停 2-5 秒，避免連續發送請求被鎖
            # 這對 Agent 連續呼叫 Tool 時特別有用
            time.sleep(3) 
            
            # --- 數據獲取 ---
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90) 
            
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
    
            if 'Adj Close' in data.columns:
                data = data.drop(columns=['Adj Close'])
            
            # 確保 OHLCV 欄位是 float 類型
            ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in ohlc_cols:
                data[col] = data[col].astype(float)
            data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            if data.index.name is None:
                data.index.name = 'Date'
            
            # 確保數據框不為空
            if data.empty:
                 return jsonify({'error': f"{symbol} 數據在清理後為空。"}), 404
                 
            # --- 指標計算開始 ---
    
            # 1. 計算均線 (MA5, MA20) - 保持不變
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
    
            # 2. 計算 KD 指標 (Stochastics Oscillator)
            # 使用 ta.momentum.StochasticOscillator
            stoch = ta.momentum.StochasticOscillator(
                high=data['High'], 
                low=data['Low'], 
                close=data['Close'], 
                window=14, 
                smooth_window=3
            )
            data['K'] = stoch.stoch().dropna()      # %K 線
            data['D'] = stoch.stoch_signal().dropna() # %D 線
            
            # 3. 計算 MACD 指標
            # 使用 ta.trend.MACD
            macd = ta.trend.MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
            data['MACD'] = macd.macd().dropna()
            data['Signal'] = macd.macd_signal().dropna()
            data['Hist'] = macd.macd_diff().dropna() # 柱狀圖
    
            # 4. 計算 DMI 指標 (ADX, ADM, ADZ)
            # 使用 ta.trend.ADX
            adx = ta.trend.ADX(data['High'], data['Low'], data['Close'], window=14)
            data['ADX'] = adx.adx().dropna()
            data['DMI+'] = adx.adx_pos().dropna() # +DI
            data['DMI-'] = adx.adx_neg().dropna() # -DI
    
            # --- 繪圖設定 ---
            
            # 均線 Addplots (Panel 0: 與 K 線主圖同一區域)
            add_plots = [
                mpf.make_addplot(data['MA5'], color='blue', label='MA5', panel=0),
                mpf.make_addplot(data['MA20'], color='red', label='MA20', panel=0)
            ]
    
            # KD 指標 (Panel 1)
            add_plots.append(mpf.make_addplot(data['K'], panel=1, color='purple', linestyle='-', label='K'))
            add_plots.append(mpf.make_addplot(data['D'], panel=1, color='orange', linestyle='-', label='D'))
            # 增加 20 和 80 的超買超賣水平線
            add_plots.append(mpf.make_addplot([80]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))
            add_plots.append(mpf.make_addplot([20]*len(data), panel=1, color='gray', linestyle=':', alpha=0.5))
    
    
            # MACD 指標 (Panel 2)
            add_plots.append(mpf.make_addplot(data['MACD'], panel=2, color='green', label='MACD'))
            add_plots.append(mpf.make_addplot(data['Signal'], panel=2, color='red', label='Signal'))
            # 柱狀圖 (Hist) 
            # mplfinance 專門的 bar 類型繪圖
            colors = ['red' if v >= 0 else 'green' for v in data['Hist']]
            add_plots.append(
                mpf.make_addplot(
                    data['Hist'], 
                    type='bar', 
                    panel=2, 
                    color=colors, 
                    alpha=0.6, 
                    secondary_y=False
                )
            )
            add_plots.append(mpf.make_addplot([0]*len(data), panel=2, color='black', linestyle=':', alpha=0.5)) # 零軸線
    
    
            # DMI/ADX 指標 (Panel 3)
            add_plots.append(mpf.make_addplot(data['ADX'], panel=3, color='black', linestyle='-', label='ADX'))
            add_plots.append(mpf.make_addplot(data['DMI+'], panel=3, color='lime', linestyle='-', label='+DI'))
            add_plots.append(mpf.make_addplot(data['DMI-'], panel=3, color='red', linestyle='-', label='-DI'))
            add_plots.append(mpf.make_addplot([20]*len(data), panel=3, color='gray', linestyle=':', alpha=0.5)) # ADX 強弱線
    
    
            # --- 繪製 K 線圖 ---
            
            buffer = io.BytesIO()
    
            # 繪圖指令，調整 figratio 以增加垂直空間給新增的指標面板
            mpf.plot(
                data, 
                type='candle', 
                volume=True, # Volume 會自動分配到 Panel 0 下方
                addplot=add_plots, 
                style='yahoo',
                title=f'{symbol} K-Line Chart with MAs, KD, MACD, DMI',
                figratio=(16, 12), # 調整圖形比例，增加高度 (例如 16寬 x 12高)
                savefig=dict(fname=buffer, format='png', dpi=100)
            )
            buffer.seek(0)
            
            # --- 輸出結果到 Flask ---
            return send_file(
                buffer, 
                mimetype='image/png', 
                as_attachment=False
            )
    
        except Exception as e:
            app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
            # 在這裡，我們可以返回更詳細的錯誤資訊，例如，如果 data.empty 是在清理之後發生的，
            # 則說明指標計算導致了問題。
            return jsonify({'error': f"服務器內部錯誤: {e}", 'trace': str(e)}), 500
