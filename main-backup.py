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

# Configurar logging robusto
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
MIN_VOLUME = 150_000_000  # Volumen mínimo aumentado
MAX_SPREAD_PERCENT = 0.1  # Máximo spread permitido

# Control de rate limits
API_CALL_DELAY = 2.0  # Delay más conservador
SCAN_INTERVAL = 30    # Escaneo cada 30 segundos

# Estado del bot
bot_state = {
    'last_signals': {},
    'failed_symbols': set(),
    'daily_trades': 0,
    'last_reset': datetime.now().date()
}

class TradingBot:
    def __init__(self):
        try:
            # Verificar variables de entorno primero
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                raise ValueError("ERROR: Variables de entorno BINANCE_API_KEY o BINANCE_API_SECRET no configuradas")
            
            if not FINANDY_SECRET:
                raise ValueError("ERROR: Variable de entorno FINANDY_SECRET no configurada")
            
            logger.info("INIT: Variables de entorno cargadas correctamente")
            
            # Inicializar cliente Binance
            self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
            logger.info("INIT: Cliente Binance inicializado")
            
            # Verificar conectividad con reintentos
            self._test_connection()
            
        except Exception as e:
            logger.error(f"ERROR: Error inicializando bot: {e}")
            raise

    def _test_connection(self):
        """Prueba la conexión a Binance con reintentos"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"TEST: Probando conexión a Binance (intento {attempt + 1}/{max_retries})...")
                
                # Probar conectividad básica
                server_time = self.client.get_server_time()
                logger.info(f"TEST: Hora del servidor Binance: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
                
                # Probar acceso a futures
                account = self.client.futures_account()
                logger.info("SUCCESS: Conexión a Binance Futures establecida correctamente")
                return True
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"ERROR: Error de conexión (intento {attempt + 1}): {e}")
                if "getaddrinfo failed" in str(e):
                    logger.error("ERROR: Problema de DNS. Posibles soluciones:")
                    logger.error("   1. Verificar conexión a internet")
                    logger.error("   2. Cambiar DNS a 8.8.8.8 o 1.1.1.1") 
                    logger.error("   3. Desactivar VPN/Proxy si está activo")
                    logger.error("   4. Verificar firewall/antivirus")
                    
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
        """Obtiene símbolos de futuros con filtros mejorados"""
        try:
            tickers = self.client.futures_ticker()
            exchange_info = self.client.futures_exchange_info()
            
            # Crear mapa de información de símbolos
            symbol_info = {s['symbol']: s for s in exchange_info['symbols']}
            
            # Obtener capital total para calcular valor mínimo de primera orden
            total_usdt = self.get_available_usdt()
            
            filtered_symbols = []
            for ticker in tickers:
                symbol = ticker['symbol']
                
                # Filtros básicos
                if not symbol.endswith('USDT'):
                    continue
                    
                volume = float(ticker['quoteVolume'])
                if volume < MIN_VOLUME:
                    continue
                
                # Verificar que el símbolo esté activo
                if symbol in symbol_info:
                    symbol_data = symbol_info[symbol]
                    if symbol_data['status'] != 'TRADING':
                        continue
                
                # Filtrar por spread
                bid_price = float(ticker.get('bidPrice', 0))
                ask_price = float(ticker.get('askPrice', 0))
                if bid_price > 0 and ask_price > 0:
                    spread_percent = ((ask_price - bid_price) / bid_price) * 100
                    if spread_percent > MAX_SPREAD_PERCENT:
                        continue
                
                # Verificar valor mínimo de primera orden del grid
                if not self._check_minimum_order_value(symbol, symbol_data, total_usdt):
                    continue
                
                filtered_symbols.append(symbol)
            
            logger.info(f"SYMBOLS: {len(filtered_symbols)} símbolos válidos encontrados")
            return filtered_symbols[:50]  # Limitar a top 50 por volumen
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo símbolos: {e}")
            return []

    def get_klines_multi_timeframe(self, symbol):
        """Obtiene datos de múltiples timeframes para confirmación"""
        try:
            # Timeframe principal (5m)
            klines_5m = self.client.futures_klines(symbol=symbol, interval='5m', limit=100)
            # Timeframe de confirmación (15m)
            klines_15m = self.client.futures_klines(symbol=symbol, interval='15m', limit=50)
            
            if not klines_5m or not klines_15m or len(klines_5m) < 60:
                return None, None
            
            def create_dataframe(klines):
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar datos válidos
                if df['close'].isna().sum() > 0:
                    return None
                    
                return df
            
            df_5m = create_dataframe(klines_5m)
            df_15m = create_dataframe(klines_15m)
            
            # Verificar que los datos sean recientes (última vela < 10 minutos)
            if df_5m is not None:
                last_timestamp = pd.to_datetime(df_5m['timestamp'].iloc[-1], unit='ms')
                if datetime.now() - last_timestamp.tz_localize(None) > timedelta(minutes=10):
                    logger.warning(f"WARNING: Datos obsoletos para {symbol}")
                    return None, None
            
            return df_5m, df_15m
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo klines para {symbol}: {e}")
            return None, None

    def technical_signal_enhanced(self, df_5m, df_15m, symbol):
        """Análisis técnico mejorado con múltiples confirmaciones"""
        if df_5m is None or df_15m is None or len(df_5m) < 60:
            return None

        try:
            # Indicadores timeframe 5m
            close_5m = df_5m['close']
            high_5m = df_5m['high']
            low_5m = df_5m['low']
            
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
            
            # ATR para volatilidad
            atr = AverageTrueRange(high_5m, low_5m, close_5m, window=14).average_true_range()
            
            # Indicadores timeframe 15m para tendencia
            close_15m = df_15m['close']
            ema21_15m = EMAIndicator(close_15m, window=21).ema_indicator()
            ema50_15m = EMAIndicator(close_15m, window=50).ema_indicator()
            rsi_15m = RSIIndicator(close_15m, window=14).rsi()
            
            latest = -1
            
            # Verificar NaN values
            indicators_5m = [ema9_5m, ema21_5m, ema50_5m, macd_diff, rsi_5m, bb_width, atr]
            indicators_15m = [ema21_15m, ema50_15m, rsi_15m]
            
            if any(pd.isna(ind.iloc[latest]) for ind in indicators_5m + indicators_15m):
                return None

            # Parámetros mejorados
            current_price = close_5m.iloc[latest]
            atr_value = atr.iloc[latest] / current_price  # ATR normalizado
            max_volatility = 0.025  # Límite de volatilidad más estricto
            min_macd_strength = atr_value * 0.5  # MACD mínimo basado en ATR
            
            # Filtros de mercado
            # 1. Volatilidad controlada
            if bb_width.iloc[latest] > max_volatility and atr_value > 0.02:
                return None
            
            # 2. Evitar rangos laterales (EMAs muy cercanas)
            ema_diff_5m = abs(ema9_5m.iloc[latest] - ema21_5m.iloc[latest]) / current_price
            if ema_diff_5m < 0.001:  # EMAs muy cercanas = rango lateral
                return None
            
            # 3. Confirmación de tendencia en 15m
            trend_15m_bullish = ema21_15m.iloc[latest] > ema50_15m.iloc[latest]
            trend_15m_bearish = ema21_15m.iloc[latest] < ema50_15m.iloc[latest]
            
            # Condiciones LONG mejoradas
            long_conditions = [
                ema9_5m.iloc[latest] > ema21_5m.iloc[latest],  # Tendencia corto plazo alcista
                ema21_5m.iloc[latest] > ema50_5m.iloc[latest],  # Tendencia medio plazo alcista
                trend_15m_bullish,  # Confirmación 15m
                45 < rsi_5m.iloc[latest] < 70,  # RSI en zona favorable
                30 < rsi_15m.iloc[latest] < 75,  # RSI 15m no sobrecomprado
                macd_diff.iloc[latest] > min_macd_strength,  # MACD positivo y fuerte
                macd_line.iloc[latest] > macd_signal.iloc[latest],  # MACD por encima de señal
                current_price > bb_middle.iloc[latest],  # Precio por encima de BB media
                current_price < bb_upper.iloc[latest] * 0.98  # No muy cerca del BB superior
            ]
            
            # Condiciones SHORT mejoradas
            short_conditions = [
                ema9_5m.iloc[latest] < ema21_5m.iloc[latest],  # Tendencia corto plazo bajista
                ema21_5m.iloc[latest] < ema50_5m.iloc[latest],  # Tendencia medio plazo bajista
                trend_15m_bearish,  # Confirmación 15m
                30 < rsi_5m.iloc[latest] < 55,  # RSI en zona favorable
                25 < rsi_15m.iloc[latest] < 70,  # RSI 15m no sobrevendido
                macd_diff.iloc[latest] < -min_macd_strength,  # MACD negativo y fuerte
                macd_line.iloc[latest] < macd_signal.iloc[latest],  # MACD por debajo de señal
                current_price < bb_middle.iloc[latest],  # Precio por debajo de BB media
                current_price > bb_lower.iloc[latest] * 1.02  # No muy cerca del BB inferior
            ]
            
            # Señales de alta probabilidad
            if sum(long_conditions) >= 7:  # Al menos 7 de 9 condiciones
                # Verificar que no hayamos enviado señal reciente
                if self._can_send_signal(symbol, 'buy'):
                    logger.info(f"SIGNAL: SEÑAL LONG fuerte para {symbol} ({sum(long_conditions)}/9 condiciones)")
                    return "buy"
            elif sum(short_conditions) >= 7:  # Al menos 7 de 9 condiciones
                if self._can_send_signal(symbol, 'sell'):
                    logger.info(f"SIGNAL: SEÑAL SHORT fuerte para {symbol} ({sum(short_conditions)}/9 condiciones)")
                    return "sell"
                    
            return None

        except Exception as e:
            logger.error(f"ERROR: Error en análisis técnico para {symbol}: {e}")
            return None

    def _check_minimum_order_value(self, symbol, symbol_info, total_usdt):
        """Verifica si el valor de la primera orden del grid cumple con el mínimo permitido"""
        try:
            # Calcular el valor de la primera orden del grid
            # Grid de 5 órdenes: 1era orden ≈ 13% del capital (9.5/74 ≈ 12.8%)
            first_order_value = total_usdt * 0.128  # 12.8% del capital para la primera orden
            
            # Obtener filtros del símbolo
            filters = symbol_info.get('filters', [])
            min_notional = 0
            min_qty = 0
            
            for filter_item in filters:
                if filter_item['filterType'] in ['MIN_NOTIONAL', 'NOTIONAL']:
                    min_notional = float(filter_item.get('notional', filter_item.get('minNotional', 0)))
                elif filter_item['filterType'] in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                    min_qty = float(filter_item.get('minQty', 0))

            
            # Verificar valor mínimo notional (valor en USDT)
            if min_notional > 0 and first_order_value < min_notional:
                logger.warning(f"SKIP: {symbol} rechazado - Primera orden ${first_order_value:.2f} < MIN_NOTIONAL ${min_notional:.2f}")
                return False

            
            # Verificar cantidad mínima si es necesario
            if min_qty > 0:
                # Obtener precio actual para calcular cantidad
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calcular cantidad que se podría comprar con la primera orden
                    calculated_qty = first_order_value / current_price
                    
                    if calculated_qty < min_qty:
                        logger.debug(f"FILTER: {symbol} rechazado - Cantidad calculada {calculated_qty:.6f} < MIN_QTY {min_qty:.6f}")
                        return False
                        
                except Exception as e:
                    logger.debug(f"FILTER: Error obteniendo precio para {symbol}: {e}")
                    return False
            
            # Si llegamos aquí, el símbolo cumple con los requisitos mínimos
            logger.debug(f"FILTER: {symbol} aprobado - Primera orden ${first_order_value:.2f} cumple requisitos")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Error verificando valor mínimo para {symbol}: {e}")
            return False

    def _can_send_signal(self, symbol, side):
        """Verifica si se puede enviar una señal (evita spam)"""
        key = f"{symbol}_{side}"
        last_signal_time = bot_state['last_signals'].get(key, 0)
        current_time = time.time()
        
        # Evitar señales duplicadas en menos de 15 minutos
        if current_time - last_signal_time < 900:  # 15 minutos
            return False
            
        bot_state['last_signals'][key] = current_time
        return True

    def get_open_positions(self):
        """Obtiene posiciones abiertas con manejo de errores"""
        try:
            positions = self.client.futures_account()['positions']
            open_positions = []
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    # Manejo seguro de campos que pueden no estar presentes
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
        """Obtiene balance USDT total con validación"""
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    # Usar 'balance' en lugar de 'availableBalance' para obtener el balance total
                    total_balance = float(balance['balance'])
                    available_balance = float(balance['availableBalance'])
                    
                    logger.info(f"BALANCE: USDT total: ${total_balance:.4f}")
                    logger.info(f"BALANCE: USDT disponible: ${available_balance:.4f}")
                    
                    # Retornar el balance total (no el disponible)
                    return total_balance
            
            logger.warning("WARNING: No se encontró balance de USDT")
            return 0
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo balance: {e}")
            return 0

    def send_signal_to_finandy(self, symbol, side):
        """Envía señal a Finandy con validación y retry"""
        payload = {
            "secret": FINANDY_SECRET,
            "symbol": symbol,
            "side": side
        }
        
        # Retry logic
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
                    bot_state['daily_trades'] += 1
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
        """Resetea contadores diarios"""
        today = datetime.now().date()
        if bot_state['last_reset'] != today:
            bot_state['daily_trades'] = 0
            bot_state['failed_symbols'].clear()
            bot_state['last_reset'] = today
            logger.info("RESET: Contadores diarios reseteados")

    def run_bot_cycle(self):
        """Ejecuta un ciclo completo del bot"""
        try:
            self.reset_daily_counters()
            
            # Verificar límites diarios
            if bot_state['daily_trades'] >= 30:  # Límite diario
                logger.info("LIMIT: Límite diario de trades alcanzado")
                return
            
            open_positions = self.get_open_positions()
            logger.info(f"POSITIONS: Posiciones abiertas: {len(open_positions)}")
            
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info("LIMIT: Límite de operaciones abiertas alcanzado (4/4)")
                return

            # Usar balance total (no disponible) para verificar capital suficiente
            total_usdt = self.get_available_usdt()
            # Verificar que tengamos balance suficiente para operar
            if total_usdt < 50:  # Balance mínimo de seguridad
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
                if signals_sent >= 2:  # Máximo 2 señales por ciclo
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
    """Función principal del bot"""
    logger.info("START: Iniciando Trading Bot Mejorado...")
    
    # Verificar conectividad a internet primero
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
            
            # Calcular tiempo de ejecución
            execution_time = time.time() - start_time
            logger.info(f"TIMING: Ciclo completado en {execution_time:.2f} segundos")
            
            # Esperar antes del próximo ciclo
            logger.info(f"WAIT: Esperando {SCAN_INTERVAL} segundos hasta próximo escaneo...")
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("STOP: Bot detenido por usuario")
    except Exception as e:
        logger.error(f"ERROR: Error crítico del bot: {e}")
        logger.info("RESTART: Reiniciando en 60 segundos...")
        time.sleep(60)
        # Reiniciar recursivamente
        main()
 
if __name__ == "__main__":
    main()