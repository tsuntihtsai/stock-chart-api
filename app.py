from flask import Flask, request, send_file, jsonify
import pandas as pd
import mplfinance as mpf
import io
from datetime import datetime, timedelta
import ta
import time
from FinMind.data import DataLoader

app = Flask(__name__)
api = DataLoader()

# --- 修正後的資料準備函式 ---
def prepare_data_fm(df):
    """將 FinMind 的資料格式轉換為 mplfinance 相容格式，並自動相容不同版本的欄位名"""
    
    # 1. 先將所有欄位名稱轉為小寫，避免大小寫不一致的問題
    df.columns = df.columns.str.lower()
    
    # 2. 定義 FinMind 可能出現的欄位對應（對應到大寫的 OHLCV）
    mapping = {
        'open': 'Open',
        'max': 'High',
        'high': 'High',  # 有些版本可能叫 high
        'min': 'Low',
        'low': 'Low',    # 有些版本可能叫 low
        'close': 'Close',
        'trading_volume': 'Volume',
        'volume': 'Volume' # 如果原本就叫 volume
    }
    
    # 篩選出存在於目前 df 中的欄位進行更名
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # 3. 處理日期索引
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
        df.set_index('Date', inplace=True)
    elif df.index.name and df.index.name.lower() == 'date':
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
    # 4. 確保必要的五個主欄位都存在，若不存在則拋出明確錯誤
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            # 如果真的缺少 Volume，補一個全為 0 的欄位避免繪圖崩潰
            if col == 'Volume':
                df['Volume'] = 0.0
            else:
                raise KeyError(f"資料中缺少必要的欄位: {col}")
        
        # 強制轉換型態為 float
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    # 移除 NaN 行
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    # 只回傳 mplfinance 需要的欄位
    return df[required_cols]

def calculate_indicators(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    stoch = ta.momentum.StochasticOscillator(
        high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3
    )
    data['K'] = stoch.stoch()
    data['D'] = stoch.stoch_signal()
    
    macd = ta.trend.MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd()
    data['Signal'] = macd.macd_signal()
    data['Hist'] = macd.macd_diff()

    adx = ta.trend.ADX(data['High'], data['Low'], data['Close'], window=14)
    data['ADX'] = adx.adx()
    data['DMI+'] = adx.adx_pos()
    data['DMI-'] = adx.adx_neg()

    # ==================== 🔴 這裡加入強力排毒 ====================
    # 有些套件會把前面幾天的 K、D、MACD 自動補 0，我們強制把前 20 天的無效值全清空
    import numpy as np
    cols_to_clean = ['MA5', 'MA20', 'K', 'D', 'MACD', 'Signal', 'Hist', 'ADX', 'DMI+', 'DMI-']
    for col in cols_to_clean:
        if col in data.columns:
            # 前 20 筆資料強制變為 NaN，這樣畫圖就不會往下衝到 0
            data.iloc[:20, data.columns.get_loc(col)] = np.nan
            
    # 最後再切片剔除前 20 筆
    data = data.iloc[20:]
    # ============================================================
    return data


@app.route('/api/kline', methods=['GET'])
def get_kline_chart():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Missing required parameter: symbol'}), 400

    # 提取純數字
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

        # --- 繪圖設定 ---

        # --- 1. 定義台股專屬顏色與風格（紅漲綠跌、黑色背景線） ---
        mc = mpf.make_marketcolors(
            up='red',          # 上漲為紅
            down='green',      # 下跌為綠
            edge='inherit',    # 蠟燭邊框跟隨主色
            wick='inherit',    # 燭芯跟隨主色
            volume='inherit',  # 量柱跟隨主色
            inherit=True
        )
        
        # 建立客製化面板風格（接近玩股網的淡灰色網格）
        custom_style = mpf.make_mpf_style(
            base_mpf_style='yahoo', 
            marketcolors=mc,
            gridcolor='#e0e0e0',  # 淺灰色網格線
            gridstyle='-'         # 實線網格，看起來更俐落
        )
        # ==================== 🔴 核心修復：加入這行程式碼 ====================
        # 因為 20MA 需要 20 天資料，我們直接剔除前 20 筆無效的 NaN 資料
        df = df.iloc[20:]
        # ===================================================================

        # --- 2. 重新調整均線與指標圖層（加入細節優化） ---
        add_plots = []
        # 主圖均線：調整線條粗細 (width) 讓它更絲滑，並補上玩股網有的 10MA 與 60MA
        add_plots.append(mpf.make_addplot(df['MA5'], color='blue', label='MA5', panel=0))
        # 如果你想跟附圖一樣有 10MA 或 60MA，可以自己在 calculate_indicators 算好後加在這裡：
        # add_plots.append(mpf.make_addplot(df['MA10'], color='#orange', width=1.2, panel=0))
        add_plots.append(mpf.make_addplot(df['MA20'], color='#ff69b4',  label='MA20', panel=0)) 
        
        # 副圖指標 (保持你原本的 panel 配置，但可以微調線條以符合新風格)
        add_plots.append(mpf.make_addplot(df['K'], panel=1, color='purple', width=1.0, label='K'))
        add_plots.append(mpf.make_addplot(df['D'], panel=1, color='orange', width=1.0, label='D'))
        add_plots.append(mpf.make_addplot([80]*len(df), panel=1, color='gray', linestyle=':', alpha=0.4))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=1, color='gray', linestyle=':', alpha=0.4))
        
        #add_plots = []
        #add_plots.append(mpf.make_addplot(df['MA5'], color='blue', label='MA5', panel=0))
        #add_plots.append(mpf.make_addplot(df['MA20'], color='red', label='MA20', panel=0))
        #add_plots.append(mpf.make_addplot(df['K'], panel=1, color='purple', label='K', ylabel='Stochastics'))
        #add_plots.append(mpf.make_addplot(df['D'], panel=1, color='orange', label='D'))
        #add_plots.append(mpf.make_addplot([80]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))
        #add_plots.append(mpf.make_addplot([20]*len(df), panel=1, color='gray', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot(df['MACD'], panel=2, color='green', label='MACD', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(df['Signal'], panel=2, color='red', label='Signal'))
        colors = ['red' if v >= 0 else 'green' for v in df['Hist']]
        add_plots.append(mpf.make_addplot(df['Hist'], type='bar', panel=2, color=colors, alpha=0.6, secondary_y=False))
        add_plots.append(mpf.make_addplot([0]*len(df), panel=2, color='black', linestyle=':', alpha=0.5))
        add_plots.append(mpf.make_addplot(df['ADX'], panel=3, color='black', label='ADX', ylabel='DMI/ADX'))
        add_plots.append(mpf.make_addplot(df['DMI+'], panel=3, color='lime', label='+DI'))
        add_plots.append(mpf.make_addplot(df['DMI-'], panel=3, color='red', label='-DI'))
        add_plots.append(mpf.make_addplot([20]*len(df), panel=3, color='gray', linestyle=':', alpha=0.5))
        # --- 3. 調整 mpf.plot 的參數 ---
        buffer = io.BytesIO()
        mpf.plot(
            df, 
            type='candle', 
            volume=True, 
            addplot=add_plots, 
            style=custom_style, # 🔴 修改：套用剛剛自訂的台股風格
            title=f'{stock_id} K-Line Chart',
            figratio=(16, 10),  # 調整成較扁的黃金比例，視覺上更像網頁看盤軟體
            tight_layout=True,  # 自動縮減不必要的邊距，填滿畫面
            savefig=dict(fname=buffer, format='png', dpi=120) # 提升 DPI 讓線條更清晰不模糊
        )
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=False)
        
    except Exception as e:
        app.logger.error(f"處理 {symbol} 時發生錯誤: {e}")
        return jsonify({'error': "服務器內部錯誤或數據處理失敗。", 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
