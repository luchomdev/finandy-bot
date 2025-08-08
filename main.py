import os
import time
import pandas as pd
import requests
import logging
import json
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
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Cargar variables de entorno
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
FINANDY_SECRET = os.getenv("FINANDY_SECRET")
FINANDY_HOOK_URL = "https://hook.finandy.com/AAT36Jzdkdb5q0vzrlUK"


# Parámetros del bot
MAX_OPEN_TRADES = 5
MIN_VOLUME = 150_000_000
MAX_SPREAD_PERCENT = 0.1
API_CALL_DELAY = 2.0
SCAN_INTERVAL = 30


bot_state = {
    'last_signals': {},
    'failed_symbols': set(),
    # Se elimina daily_trades para no limitar operaciones diarias
    'last_reset': datetime.now().date()
}


def create_dataframe(klines):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df['close'].isna().sum() > 0:
        return None
    return df

# NUEVO: funciones separadas para patrones de vela
def es_martillo_alcista(row):
    cuerpo = abs(row['close'] - row['open'])
    mecha_inferior = min(row['close'], row['open']) - row['low']
    mecha_superior = row['high'] - max(row['close'], row['open'])
    return mecha_inferior > cuerpo * 2 and mecha_superior < cuerpo

def es_martillo_bajista(row):
    cuerpo = abs(row['close'] - row['open'])
    mecha_superior = row['high'] - max(row['close'], row['open'])
    mecha_inferior = min(row['close'], row['open']) - row['low']
    return mecha_superior > cuerpo * 2 and mecha_inferior < cuerpo

# def es_martillo(row):
    cuerpo = abs(row['close'] - row['open'])
    mecha_superior = row['high'] - max(row['close'], row['open'])
    mecha_inferior = min(row['close'], row['open']) - row['low']
    return (mecha_inferior > cuerpo * 2) and (mecha_superior < cuerpo)


class TradingBot:
    def __init__(self):
        try:
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                raise ValueError("ERROR: Variables de entorno BINANCE_API_KEY o BINANCE_API_SECRET no configuradas")
            if not FINANDY_SECRET:
                raise ValueError("ERROR: Variable de entorno FINANDY_SECRET no configurada")
            logger.info("INIT: Variables de entorno cargadas correctamente")
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
            logger.info("INIT: Cliente Binance inicializado")
            self._test_connection()
        except Exception as e:
            logger.error(f"ERROR: Error inicializando bot: {e}")
            raise

    def _test_connection(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"TEST: Probando conexión a Binance (intento {attempt + 1}/{max_retries})...")
                server_time = self.client.get_server_time()
                logger.info(f"TEST: Hora del servidor Binance: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
                account = self.client.futures_account()
                logger.info("SUCCESS: Conexión a Binance Futures establecida correctamente")
                return True
            except requests.exceptions.ConnectionError as e:
                logger.error(f"ERROR: Error de conexión (intento {attempt + 1}): {e}")
                if "getaddrinfo failed" in str(e):
                    logger.error("ERROR: Problema de DNS. Soluciones sugeridas.")
            except Exception as e:
                logger.error(f"ERROR: Error de autenticación o API (intento {attempt + 1}): {e}")
                if "Invalid API-key" in str(e):
                    logger.error("ERROR: Verificar BINANCE_API_KEY en archivo .env")
                elif "Signature for this request" in str(e):
                    logger.error("ERROR: Verificar BINANCE_API_SECRET en archivo .env")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.info(f"WAIT: Esperando {wait_time} segundos antes del siguiente intento...")
                time.sleep(wait_time)
        raise Exception("ERROR: No se pudo establecer conexión con Binance después de 3 intentos")

    def get_futures_symbols(self):
        try:
            tickers = self.client.futures_ticker()
            exchange_info = self.client.futures_exchange_info()
            symbol_info = {s['symbol']: s for s in exchange_info['symbols']}
            total_usdt = self.get_available_usdt()
            filtered_symbols = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                volume = float(ticker['quoteVolume'])
                if volume < MIN_VOLUME:
                    continue
                if symbol in symbol_info:
                    symbol_data = symbol_info[symbol]
                    if symbol_data['status'] != 'TRADING':
                        continue
                bid_price = float(ticker.get('bidPrice', 0))
                ask_price = float(ticker.get('askPrice', 0))
                if bid_price > 0 and ask_price > 0:
                    spread_percent = ((ask_price - bid_price) / bid_price) * 100
                    if spread_percent > MAX_SPREAD_PERCENT:
                        continue
                if not self._check_minimum_order_value(symbol, symbol_data, total_usdt):
                    continue
                filtered_symbols.append(symbol)
            logger.info(f"SYMBOLS: {len(filtered_symbols)} símbolos válidos encontrados")
            return filtered_symbols[:50]
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo símbolos: {e}")
            return []

    def get_klines_multi_timeframe(self, symbol):
        try:
            klines_5m = self.client.futures_klines(symbol=symbol, interval='5m', limit=100)
            klines_15m = self.client.futures_klines(symbol=symbol, interval='15m', limit=50)
            if not klines_5m or not klines_15m or len(klines_5m) < 60:
                return None, None
            df_5m = create_dataframe(klines_5m)
            df_15m = create_dataframe(klines_15m)
            if df_5m is not None:
                last_timestamp = pd.to_datetime(df_5m['timestamp'].iloc[-1], unit='ms')
                if datetime.now() - last_timestamp.tz_localize(None) > timedelta(minutes=10):
                    logger.warning(f"WARNING: Datos obsoletos para {symbol}")
                    return None, None
            return df_5m, df_15m
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo klines para {symbol}: {e}")
            return None, None

    def get_trend_higher_tf(self, symbol, interval='1h'):
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=100)
            df = create_dataframe(klines)
            if df is None:
                return False
            close = df['close']
            ema21 = EMAIndicator(close, window=21).ema_indicator()
            ema50 = EMAIndicator(close, window=50).ema_indicator()
            # Tendencia alcista si EMA21 > EMA50
            return ema21.iloc[-1] > ema50.iloc[-1]
        except Exception as e:
            logger.error(f"ERROR: get_trend_higher_tf para {symbol} intervalo {interval}: {e}")
            return False

    def technical_signal_enhanced(self, df_5m, df_15m, symbol):
        if df_5m is None or df_15m is None or len(df_5m) < 60:
            return None

        try:
            close_5m = df_5m['close']
            high_5m = df_5m['high']
            low_5m = df_5m['low']
            vol_5m = df_5m['volume']

            ema9_5m = EMAIndicator(close_5m, window=9).ema_indicator()
            ema21_5m = EMAIndicator(close_5m, window=21).ema_indicator()
            ema50_5m = EMAIndicator(close_5m, window=50).ema_indicator()

            macd_5m = MACD(close_5m)
            macd_line = macd_5m.macd()
            macd_signal = macd_5m.macd_signal()
            macd_diff = macd_5m.macd_diff()

            rsi_5m = RSIIndicator(close_5m, window=14).rsi()
            bb_5m = BollingerBands(close_5m, window=20, window_dev=2)
            bb_upper = bb_5m.bollinger_hband()
            bb_lower = bb_5m.bollinger_lband()
            bb_middle = bb_5m.bollinger_mavg()
            bb_width = (bb_upper - bb_lower) / bb_middle

            atr = AverageTrueRange(high_5m, low_5m, close_5m, window=14).average_true_range()

            close_15m = df_15m['close']
            ema21_15m = EMAIndicator(close_15m, window=21).ema_indicator()
            ema50_15m = EMAIndicator(close_15m, window=50).ema_indicator()
            rsi_15m = RSIIndicator(close_15m, window=14).rsi()

            latest = -1
            
            # Validación NAs
            indicators_5m = [ema9_5m, ema21_5m, ema50_5m, macd_diff, rsi_5m, bb_width, atr]
            indicators_15m = [ema21_15m, ema50_15m, rsi_15m]
            
            # NUEVO: preparación para validaciones adicionales
            prev_close = df_5m['close'].iloc[-2]
            current_close = df_5m['close'].iloc[-1]
            vol_actual = vol_5m.iloc[-1]
            vol_3media = vol_5m.rolling(window=3).mean().iloc[-2]
            patron_martillo_long = es_martillo_alcista(df_5m.iloc[latest])
            patron_martillo_short = es_martillo_bajista(df_5m.iloc[latest])
            if any(pd.isna(ind.iloc[latest]) for ind in indicators_5m + indicators_15m):
                return None

            current_price = close_5m.iloc[latest]
            atr_value = atr.iloc[latest] / current_price
            max_volatility = 0.025
            min_macd_strength = atr_value * 0.5
            if bb_width.iloc[latest] > max_volatility and atr_value > 0.02:
                return None

            ema_diff_5m = abs(ema9_5m.iloc[latest] - ema21_5m.iloc[latest]) / current_price
            if ema_diff_5m < 0.001:
                return None

            trend_15m_bullish = ema21_15m.iloc[latest] > ema50_15m.iloc[latest]
            trend_15m_bearish = ema21_15m.iloc[latest] < ema50_15m.iloc[latest]

            # CONFIRMACIÓN VOLUMEN
            avg_vol_5m = vol_5m.rolling(window=5).mean()
            volumen_confirmado = vol_5m.iloc[latest] > avg_vol_5m.iloc[-2]


            long_conditions = [
                ema9_5m.iloc[latest] > ema21_5m.iloc[latest],
                ema21_5m.iloc[latest] > ema50_5m.iloc[latest],
                trend_15m_bullish,
                45 < rsi_5m.iloc[latest] < 70,
                30 < rsi_15m.iloc[latest] < 75,
                macd_diff.iloc[latest] > min_macd_strength,
                macd_line.iloc[latest] > macd_signal.iloc[latest],
                current_price > bb_middle.iloc[latest],
                current_price < bb_upper.iloc[latest] * 0.98,
                volumen_confirmado,
                patron_martillo_long, 
                self.get_trend_higher_tf(symbol, interval='1h')
            ]

            short_conditions = [
                ema9_5m.iloc[latest] < ema21_5m.iloc[latest],
                ema21_5m.iloc[latest] < ema50_5m.iloc[latest],
                trend_15m_bearish,
                30 < rsi_5m.iloc[latest] < 55,
                25 < rsi_15m.iloc[latest] < 70,
                macd_diff.iloc[latest] < -min_macd_strength,
                macd_line.iloc[latest] < macd_signal.iloc[latest],
                current_price < bb_middle.iloc[latest],
                current_price > bb_lower.iloc[latest] * 1.02,
                volumen_confirmado,
                patron_martillo_short,
                not self.get_trend_higher_tf(symbol, interval='1h')
            ]

            signal = None
            if sum(long_conditions) >= 7:
                if self._can_send_signal(symbol, 'buy'):
                    logger.info(f"SIGNAL: SEÑAL LONG fuerte para {symbol} ({sum(long_conditions)}/12 condiciones)")
                    signal = "sell"
            elif sum(short_conditions) >= 7:
                if self._can_send_signal(symbol, 'sell'):
                    logger.info(f"SIGNAL: SEÑAL SHORT fuerte para {symbol} ({sum(short_conditions)}/12 condiciones)")
                    signal = "buy"

            # Filtro estructura de precio
            next_support, next_resistance = self.price_structure_filter(df_5m, df_15m, current_price)
            if signal == "buy" and next_resistance and ((next_resistance - current_price) / current_price) < 0.007:
                logger.info(f"Filtro estructura: resistencia muy cerca ({next_resistance}) -> señal descartada")
                return None
            if signal == "sell" and next_support and ((current_price - next_support) / current_price) < 0.007:
                logger.info(f"Filtro estructura: soporte muy cerca ({next_support}) -> señal descartada")
                return None

            # NUEVO: Confirmación volumen ruptura
            if vol_actual < vol_3media * 1.1:
                logger.info("VOL: Volumen no confirma ruptura. Señal descartada.")
                return None


            return signal

        except Exception as e:
         logger.error(f"ERROR: Error en análisis técnico para {symbol}: {e}")
        return None


    def btc_market_filter(self):
        df_btc_15m, _ = self.get_klines_multi_timeframe("BTCUSDT")
        if df_btc_15m is None:
            return True
        atr_btc = AverageTrueRange(df_btc_15m['high'], df_btc_15m['low'], df_btc_15m['close'], window=14).average_true_range().iloc[-1]
        change_15m = (df_btc_15m['close'].iloc[-1] - df_btc_15m['close'].iloc[-4]) / df_btc_15m['close'].iloc[-4]
        if abs(change_15m) > 0.015 or (atr_btc / df_btc_15m['close'].iloc[-1]) > 0.02:
            logger.warning("BTC muy volátil → Pausando señales")
            return False
        return True

    def detect_recent_pivots(self, df, threshold=0.01):
        pivots = []
        last_pivot = df['close'].iloc[0]
        last_direction = None
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            change = (price - last_pivot) / last_pivot
            if abs(change) >= threshold:
                direction = "up" if change > 0 else "down"
                if last_direction != direction:
                    pivots.append((df.index[i], price))
                    last_pivot = price
                    last_direction = direction
        return pivots[-5:]

    def price_structure_filter(self, df_5m, df_15m, current_price):
        pivots_5m = self.detect_recent_pivots(df_5m, threshold=0.008)
        pivots_15m = self.detect_recent_pivots(df_15m, threshold=0.01)
        levels = sorted(set([p[1] for p in pivots_5m + pivots_15m]))
        next_resistance = min([lvl for lvl in levels if lvl > current_price], default=None)
        next_support = max([lvl for lvl in levels if lvl < current_price], default=None)
        return next_support, next_resistance

    def _check_minimum_order_value(self, symbol, symbol_info, total_usdt):
        try:
            first_order_value = total_usdt * 0.128
            filters = symbol_info.get('filters', [])
            min_notional = 0
            min_qty = 0
            for filter_item in filters:
                if filter_item['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']:
                    min_notional = float(filter_item.get('notional', filter_item.get('minNotional', 0)))
                elif filter_item['filterType'] in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                    min_qty = float(filter_item.get('minQty', 0))
            if min_notional > 0 and first_order_value < min_notional:
                logger.warning(f"SKIP: {symbol} rechazado - Primera orden ${first_order_value:.2f} < MIN_NOTIONAL ${min_notional:.2f}")
                return False
            if min_qty > 0:
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    calculated_qty = first_order_value / current_price
                    if calculated_qty < min_qty:
                        logger.debug(f"FILTER: {symbol} rechazado - Cantidad calculada {calculated_qty:.6f} < MIN_QTY {min_qty:.6f}")
                        return False
                except Exception as e:
                    logger.debug(f"FILTER: Error obteniendo precio para {symbol}: {e}")
                    return False
            logger.debug(f"FILTER: {symbol} aprobado - Primera orden ${first_order_value:.2f} cumple requisitos")
            return True
        except Exception as e:
            logger.error(f"ERROR: Error verificando valor mínimo para {symbol}: {e}")
            return False

    def _can_send_signal(self, symbol, side):
        key = f"{symbol}_{side}"
        last_signal_time = bot_state['last_signals'].get(key, 0)
        current_time = time.time()
        if current_time - last_signal_time < 900:
            return False
        bot_state['last_signals'][key] = current_time
        return True

    def get_open_positions(self):
        try:
            positions = self.client.futures_account()['positions']
            open_positions = []
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    pnl = 0.0
                    if 'unrealizedPnl' in pos:
                        pnl = float(pos['unrealizedPnl'])
                    elif 'unRealizedProfit' in pos:
                        pnl = float(pos['unRealizedProfit'])
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'size': float(pos['positionAmt']),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'pnl': pnl
                    })
            return open_positions
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo posiciones: {e}")
            return []

    def get_available_usdt(self):
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    total_balance = float(balance['balance'])
                    logger.info(f"BALANCE: USDT total: ${total_balance:.4f}")
                    return total_balance
            logger.warning("WARNING: No se encontró balance de USDT")
            return 0
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo balance: {e}")
            return 0

    def send_signal_to_finandy(self, symbol, side):
        payload = {
            "secret": FINANDY_SECRET,
            "symbol": symbol,
            "side": side
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    FINANDY_HOOK_URL,
                    json=payload,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code == 200:
                    logger.info(f"SUCCESS: SEÑAL ENVIADA: {symbol} | {side.upper()}")
                    return True
                else:
                    logger.warning(f"WARNING: Respuesta Finandy: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"ERROR: Error enviando señal (intento {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        logger.error(f"ERROR: Falló envío de señal para {symbol} después de {max_retries} intentos")
        bot_state['failed_symbols'].add(symbol)
        return False

    def reset_daily_counters(self):
        today = datetime.now().date()
        if bot_state['last_reset'] != today:
            bot_state['failed_symbols'].clear()
            bot_state['last_reset'] = today
            logger.info("RESET: Contadores diarios reseteados")

    def run_bot_cycle(self):
        try:
            self.reset_daily_counters()

            open_positions = self.get_open_positions()
            logger.info(f"POSITIONS: Posiciones abiertas: {len(open_positions)}")
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info("LIMIT: Límite de operaciones abiertas alcanzado")
                return

            if not self.btc_market_filter():
                logger.warning("PAUSE: Mercado BTC volátil, se pausará el ciclo")
                return

            total_usdt = self.get_available_usdt()
            if total_usdt < 50:
                logger.warning(f"WARNING: Capital insuficiente para operar. Balance total: ${total_usdt:.2f}")
                return

            symbols = self.get_futures_symbols()
            if not symbols:
                logger.error("ERROR: No se pudieron obtener símbolos")
                return

            open_symbols = {pos['symbol'] for pos in open_positions}
            signals_sent = 0

            logger.info(f"SCAN: Analizando {len(symbols)} símbolos...")
            for symbol in symbols:
                # Limitar máximo de señales por ciclo a 2 (opcional, mantiene control de cantidad simultánea)
                if signals_sent >= 2:
                    break
                if symbol in open_symbols or symbol in bot_state['failed_symbols']:
                    continue
                try:
                    df_5m, df_15m = self.get_klines_multi_timeframe(symbol)
                    signal = self.technical_signal_enhanced(df_5m, df_15m, symbol)
                    if signal:
                        if self.send_signal_to_finandy(symbol, signal):
                            signals_sent += 1
                            open_symbols.add(symbol)
                except Exception as e:
                    logger.error(f"ERROR: Error procesando {symbol}: {e}")
                    bot_state['failed_symbols'].add(symbol)
                time.sleep(API_CALL_DELAY)

            if signals_sent == 0:
                logger.info("SCAN: No se encontraron señales válidas en este ciclo")
            else:
                logger.info(f"SUCCESS: Ciclo completado: {signals_sent} señales enviadas")
        except Exception as e:
            logger.error(f"ERROR: Error en ciclo del bot: {e}")


def main():
    logger.info("START: Iniciando Trading Bot Mejorado...")
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        logger.info("NETWORK: Conectividad a internet verificada")
    except Exception as e:
        logger.error(f"ERROR: Sin conexión a internet: {e}")
        logger.error("FIX: Verificar conexión de red antes de continuar")
        return
    try:
        bot = TradingBot()
        logger.info("SUCCESS: Bot inicializado correctamente")
        while True:
            start_time = time.time()
            logger.info("=" * 50)
            logger.info("CYCLE: Iniciando nuevo ciclo de escaneo...")
            try:
                bot.run_bot_cycle()
            except requests.exceptions.ConnectionError as e:
                logger.error(f"ERROR: Error de conexión durante ciclo: {e}")
                logger.info("WAIT: Esperando 60 segundos para reconectar...")
                time.sleep(60)
                continue
            except Exception as e:
                logger.error(f"ERROR: Error en ciclo: {e}")
                time.sleep(30)
                continue
            execution_time = time.time() - start_time
            logger.info(f"TIMING: Ciclo completado en {execution_time:.2f} segundos")
            logger.info(f"WAIT: Esperando {SCAN_INTERVAL} segundos hasta próximo escaneo...")
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        logger.info("STOP: Bot detenido por usuario")
    except Exception as e:
        logger.error(f"ERROR: Error crítico del bot: {e}")
        logger.info("RESTART: Reiniciando en 60 segundos...")
        time.sleep(60)
        main()


if __name__ == "__main__":
    main()