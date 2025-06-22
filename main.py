import os
import time
import pandas as pd
import requests
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
FINANDY_SECRET = os.getenv("FINANDY_SECRET")
FINANDY_HOOK_URL = "https://hook.finandy.com/AAT36Jzdkdb5q0vzrlUK"

MAX_OPEN_TRADES = 4
LEVERAGE = 10
START_AMOUNT = 10  # Solo la primera compra

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_futures_symbols():
    tickers = client.futures_ticker()
    filtered = [
        t['symbol'] for t in tickers
        if "USDT" in t['symbol'] and float(t['quoteVolume']) > 100_000_000
    ]
    return filtered

def get_klines(symbol, interval='5m', limit=100):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines or len(klines) < 60:
            print(f"[SKIP] {symbol} no tiene suficientes datos.")
            return None
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print(f"[KLINES ERROR] {symbol} => {e}")
        return None

def technical_signal(df):
    if df is None or len(df) < 60:
        return None

    try:
        ema20 = EMAIndicator(df['close'], window=20).ema_indicator()
        ema50 = EMAIndicator(df['close'], window=50).ema_indicator()
        macd_diff = MACD(df['close']).macd_diff()
        rsi = RSIIndicator(df['close']).rsi()

        latest = -1
        # Verificar que no haya NaN
        if any(pd.isna(x.iloc[latest]) for x in [ema20, ema50, macd_diff, rsi]):
            return None

        long_condition = (
            ema20.iloc[latest] > ema50.iloc[latest] and
            macd_diff.iloc[latest] > 0 and
            45 <= rsi.iloc[latest] <= 60
        )

        short_condition = (
            ema20.iloc[latest] < ema50.iloc[latest] and
            macd_diff.iloc[latest] < 0 and
            40 <= rsi.iloc[latest] <= 55
        )

        if long_condition:
            return "buy"
        elif short_condition:
            return "sell"
        else:
            return None
    except Exception as e:
        print(f"[TECH ERROR] Error en indicadores => {e}")
        return None

def get_open_positions():
    positions = client.futures_account()['positions']
    return [p['symbol'] for p in positions if float(p['positionAmt']) != 0]

def get_available_usdt():
    balances = client.futures_account_balance()
    for b in balances:
        if b['asset'] == 'USDT':
            return float(b['availableBalance'])
    return 0

def send_signal_to_finandy(symbol, side):
    payload = {
        "secret": FINANDY_SECRET,
        "symbol": symbol,
        "side": side,
        "leverage": LEVERAGE,
        "marginType": "cross",
        "amount": START_AMOUNT
    }
    response = requests.post(FINANDY_HOOK_URL, json=payload)
    print(f"[SIGNAL] {symbol} | {side.upper()} | ${START_AMOUNT} | Status: {response.status_code} | {response.text}")

def run_bot():
    open_positions = get_open_positions()
    available_usdt = get_available_usdt()

    if len(open_positions) >= MAX_OPEN_TRADES:
        print("🔒 Límite de operaciones abiertas alcanzado.")
        return

    if available_usdt < START_AMOUNT:
        print(f"💰 Capital insuficiente. Requiere mínimo ${START_AMOUNT}, disponible: ${available_usdt}")
        return

    symbols = get_futures_symbols()
    print(f"📊 Analizando {len(symbols)} pares con volumen > 100M...")

    for symbol in symbols:
        if symbol in open_positions:
            continue

        try:
            df = get_klines(symbol)
            signal = technical_signal(df)
            if signal:
                send_signal_to_finandy(symbol, signal)
                open_positions.append(symbol)
                if len(open_positions) >= MAX_OPEN_TRADES:
                    break
        except Exception as e:
            print(f"[ERROR] {symbol} => {e}")
        time.sleep(1.2)

if __name__ == "__main__":
    while True:
        print("🔍 Escaneando mercado...")
        run_bot()
        time.sleep(10)
