import os
import time
import pandas as pd
import requests
import logging
import json
import numpy as np
import pickle
import hashlib
from datetime import datetime, timedelta
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from dotenv import load_dotenv

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
FINANDY_SECRET = os.getenv("FINANDY_SECRET")
FINANDY_HOOK_URL = "https://hook.finandy.com/AAT36Jzdkdb5q0vzrlUK"

# Parámetros
MAX_OPEN_TRADES = 5
MIN_VOLUME = 150_000_000
MAX_SPREAD_PERCENT = 0.15
API_CALL_DELAY = 1.0
SCAN_INTERVAL = 30
LEARNING_DATA_FILE = 'trading_learning_data.pkl'

bot_state = {
    'last_signals': {},
    'failed_symbols': set(),
    'last_reset': datetime.now().date(),
    'symbol_cooldown': {},
    'signal_stats': {
        'total_analyzed': 0,
        'rejected_conditions': 0,
        'rejected_probability': 0,
        'rejected_cooldown': 0,
        'rejected_position': 0,
        'sent_signals': 0
    }
}

def create_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df if df['close'].notna().all() else None

class TradingBotFinal:
    def __init__(self):
        load_dotenv()
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        logger.info("✅ Bot inicializado correctamente")

    def reset_daily_counters(self):
        today = datetime.now().date()
        if bot_state['last_reset'] != today:
            bot_state['failed_symbols'].clear()
            bot_state['last_reset'] = today
            bot_state['signal_stats'] = {k: 0 for k in bot_state['signal_stats']}
            logger.info("🔄 Contadores diarios reseteados")

    def get_futures_symbols(self):
        try:
            tickers = self.client.futures_ticker()
            symbols = [t['symbol'] for t in tickers 
                      if t['symbol'].endswith('USDT') 
                      and float(t.get('quoteVolume', 0)) >= MIN_VOLUME]
            return symbols[:40]
        except Exception as e:
            logger.error(f"❌ Error obteniendo símbolos: {e}")
            return []

    def get_klines_multi_timeframe(self, symbol):
        try:
            klines_5m = self.client.futures_klines(symbol=symbol, interval='5m', limit=60)
            klines_15m = self.client.futures_klines(symbol=symbol, interval='15m', limit=30)
            
            if len(klines_5m) < 50:
                return None, None
                
            return create_dataframe(klines_5m), create_dataframe(klines_15m)
        except:
            return None, None

    def get_open_positions(self):
        """Obtener posiciones abiertas en futuros"""
        try:
            account = self.client.futures_account()
            positions = []
            for pos in account['positions']:
                if float(pos['positionAmt']) != 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'size': float(pos['positionAmt']),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'pnl': float(pos.get('unrealizedPnl', 0))
                    })
            return positions
        except Exception as e:
            logger.error(f"❌ Error obteniendo posiciones: {e}")
            return []

    def get_available_usdt(self):
        try:
            balances = self.client.futures_account_balance()
            usdt_balance = next((float(b['balance']) for b in balances if b['asset'] == 'USDT'), 0)
            return usdt_balance
        except:
            return 0

    def is_symbol_in_cooldown(self, symbol):
        if symbol in bot_state['symbol_cooldown']:
            last_trade = bot_state['symbol_cooldown'][symbol]
            remaining = (datetime.now() - last_trade).total_seconds() / 60
            return remaining < 30, max(0, 30 - remaining)
        return False, 0

    def check_existing_position(self, symbol):
        """Verificar si hay posición abierta en el símbolo"""
        try:
            positions = self.get_open_positions()
            return symbol in [p['symbol'] for p in positions]
        except:
            return False

    def check_brutal_movement(self, symbol):
        """Detectar movimientos extremos"""
        try:
            klines = self.client.futures_klines(symbol=symbol, interval='5m', limit=6)
            if len(klines) < 6:
                return False, None, 0
            
            df = create_dataframe(klines)
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100
            
            if abs(change_pct) > 5:
                opposite_side = 'sell' if change_pct > 0 else 'buy'
                return True, opposite_side, change_pct
            
            return False, None, 0
        except:
            return False, None, 0

    def calculate_signal_score(self, df_5m, df_15m):
        """Evaluar señal con score"""
        try:
            close_5m = df_5m['close']
            vol_5m = df_5m['volume']

            # Indicadores
            ema9 = EMAIndicator(close_5m, window=9).ema_indicator().iloc[-1]
            ema21 = EMAIndicator(close_5m, window=21).ema_indicator().iloc[-1]
            ema50 = EMAIndicator(close_5m, window=50).ema_indicator().iloc[-1]
            
            rsi = RSIIndicator(close_5m, window=14).rsi().iloc[-1]
            macd_line = MACD(close_5m).macd().iloc[-1]
            macd_signal = MACD(close_5m).macd_signal().iloc[-1]

            # Score LONG
            long_score = 0
            long_checks = [
                ema9 > ema21,
                ema21 > ema50,
                45 <= rsi <= 70,
                macd_line > macd_signal,
                vol_5m.iloc[-1] > vol_5m.rolling(5).mean().iloc[-2]
            ]
            long_score = sum(long_checks) / len(long_checks)

            # Score SHORT
            short_score = 0
            short_checks = [
                ema9 < ema21,
                ema21 < ema50,
                30 <= rsi <= 55,
                macd_line < macd_signal,
                vol_5m.iloc[-1] > vol_5m.rolling(5).mean().iloc[-2]
            ]
            short_score = sum(short_checks) / len(short_checks)

            if long_score >= 0.6:
                return 'buy', long_score
            elif short_score >= 0.7:
                return 'sell', short_score
            
            return None, 0
        except:
            return None, 0

    def send_signal_to_finandy(self, symbol, side, reason=""):
        """Enviar señal"""
        payload = {"secret": FINANDY_SECRET, "symbol": symbol, "side": side}
        
        try:
            response = requests.post(FINANDY_HOOK_URL, json=payload, timeout=10)
            success = response.status_code == 200
            if success:
                bot_state['symbol_cooldown'][symbol] = datetime.now()
            return success
        except:
            return False

    def btc_market_filter(self):
        """Verificar estado del mercado BTC"""
        try:
            klines = self.client.futures_klines(symbol="BTCUSDT", interval='15m', limit=4)
            if len(klines) >= 4:
                df = create_dataframe(klines)
                change = abs((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0])
                return change <= 0.025
        except:
            pass
        return True

    def run_bot_cycle(self):
        """Ejecutar ciclo completo"""
        try:
            self.reset_daily_counters()
            
            open_positions = self.get_open_positions()
            logger.info(f"📊 RESUMEN: {len(open_positions)}/{MAX_OPEN_TRADES} trades abiertos")
            
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info("🛑 Límite alcanzado")
                return

            if not self.btc_market_filter():
                logger.warning("⚠️ Mercado BTC volátil")
                return

            total_usdt = self.get_available_usdt()
            if total_usdt < 50:
                logger.warning(f"💰 Capital: ${total_usdt:.2f}")
                return

            symbols = self.get_futures_symbols()
            if not symbols:
                logger.error("❌ Sin símbolos disponibles")
                return

            signals_sent = 0
            analyzed = 0
            
            for symbol in symbols:
                if signals_sent >= 3:
                    break
                
                analyzed += 1
                
                # Verificaciones rápidas
                in_cooldown, remaining = self.is_symbol_in_cooldown(symbol)
                if in_cooldown:
                    logger.debug(f"⏰ {symbol} cooldown ({remaining:.1f}min)")
                    continue
                
                if self.check_existing_position(symbol):
                    logger.debug(f"📈 {symbol} ya tiene posición")
                    continue
                
                try:
                    df_5m, df_15m = self.get_klines_multi_timeframe(symbol)
                    if df_5m is None:
                        continue
                    
                    # Movimiento brusco
                    brutal_move, opposite, change_pct = self.check_brutal_movement(symbol)
                    if brutal_move:
                        logger.info(f"🎯 {symbol}: {change_pct:.1f}% → {opposite}")
                        if self.send_signal_to_finandy(symbol, opposite, "Contrarian"):
                            signals_sent += 1
                            continue
                    
                    # Señal técnica
                    signal, score = self.calculate_signal_score(df_5m, df_15m)
                    if signal and score >= 0.6:
                        logger.info(f"✅ {symbol}: {signal.upper()} (score: {score:.1%})")
                        if self.send_signal_to_finandy(symbol, signal, "Technical"):
                            signals_sent += 1
                            
                except Exception as e:
                    logger.debug(f"⚠️ Error {symbol}: {e}")
                
                time.sleep(API_CALL_DELAY)

            logger.info(f"📈 CICLO: {signals_sent} señal(es) | {analyzed} analizados")
            
        except Exception as e:
            logger.error(f"💥 ERROR: {e}")

def main():
    logger.info("🚀 Iniciando Trading Bot Final...")
    
    bot = TradingBotFinal()
    
    while True:
        try:
            start_time = time.time()
            logger.info("\n" + "="*60)
            
            bot.run_bot_cycle()
            
            elapsed = time.time() - start_time
            sleep_time = max(0, SCAN_INTERVAL - elapsed)
            logger.info(f"⏱️ Esperando {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido")
            break
        except Exception as e:
            logger.error(f"💥 ERROR: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()