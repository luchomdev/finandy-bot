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

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.start_balance = None
        self.current_balance = None
        
    def add_trade(self, symbol, side, entry_price, exit_price, quantity, pnl):
        """Registra un trade completado"""
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl
        })
        
    def calculate_stats(self):
        """Calcula métricas de performance"""
        if not self.trades:
            return None
            
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        profit_factor = (len(wins) * avg_win) / (len(losses) * abs(avg_loss)) if losses else float('inf')
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'max_drawdown': self.calculate_max_drawdown()
        }
        
    def calculate_max_drawdown(self):
        """Calcula el máximo drawdown"""
        if not self.trades:
            return 0
            
        running_balance = self.start_balance
        peak = running_balance
        max_drawdown = 0
        
        for trade in self.trades:
            running_balance += trade['pnl']
            if running_balance > peak:
                peak = running_balance
            drawdown = (peak - running_balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown

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
            
            # Inicializar performance tracker
            self.performance_tracker = PerformanceTracker()
            usdt_balance = self.get_available_usdt()
            self.performance_tracker.start_balance = usdt_balance
            self.performance_tracker.current_balance = usdt_balance
            
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
        """Obtiene símbolos de futuros con filtros mejorados para scalping"""
        try:
            # Obtener todos los símbolos y ordenarlos por liquidez y spread
            tickers = self.client.futures_ticker()
            exchange_info = self.client.futures_exchange_info()
            
            symbol_info = {s['symbol']: s for s in exchange_info['symbols']}
            symbol_data = []
            
            for ticker in tickers:
                symbol = ticker['symbol']
                
                if not symbol.endswith('USDT'):
                    continue
                    
                # Calcular métricas clave para scalping
                volume = float(ticker['quoteVolume'])
                bid_price = float(ticker.get('bidPrice', 0))
                ask_price = float(ticker.get('askPrice', 0))
                
                if bid_price <= 0 or ask_price <= 0:
                    continue
                    
                spread_percent = ((ask_price - bid_price) / bid_price) * 100
                price = (bid_price + ask_price) / 2
                
                # Añadir filtros adicionales para scalping
                if (volume > MIN_VOLUME and 
                    spread_percent < MAX_SPREAD_PERCENT and 
                    symbol in symbol_info and
                    symbol_info[symbol]['status'] == 'TRADING'):
                    
                    # Calcular volatilidad reciente (últimas 24h)
                    klines = self.client.futures_klines(symbol=symbol, interval='1h', limit=24)
                    if len(klines) >= 24:
                        closes = [float(k[4]) for k in klines]
                        high = max(closes)
                        low = min(closes)
                        volatility = (high - low) / low * 100
                        
                        # Puntuación para scalping (mayor es mejor)
                        score = (volume / 1_000_000) * (1 / spread_percent) * (volatility / 10)
                        
                        symbol_data.append({
                            'symbol': symbol,
                            'volume': volume,
                            'spread': spread_percent,
                            'volatility': volatility,
                            'price': price,
                            'score': score
                        })
            
            # Ordenar por mejor puntuación para scalping
            symbol_data.sort(key=lambda x: x['score'], reverse=True)
            return [x['symbol'] for x in symbol_data[:30]]  # Top 30 para scalping
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo símbolos: {e}")
            return []

    def get_klines_for_scalping(self, symbol):
        """Obtiene datos optimizados para scalping"""
        try:
            # Timeframe principal para scalping (1m)
            klines_1m = self.client.futures_klines(symbol=symbol, interval='1m', limit=100)
            # Timeframe de confirmación (3m)
            klines_3m = self.client.futures_klines(symbol=symbol, interval='3m', limit=50)
            
            if not klines_1m or not klines_3m or len(klines_1m) < 60:
                return None, None
            
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
            
            df_1m = create_dataframe(klines_1m)
            df_3m = create_dataframe(klines_3m)
            
            # Verificar que los datos sean recientes (última vela < 2 minutos)
            if df_1m is not None:
                last_timestamp = pd.to_datetime(df_1m['timestamp'].iloc[-1], unit='ms')
                if datetime.now() - last_timestamp.tz_localize(None) > timedelta(minutes=2):
                    logger.warning(f"WARNING: Datos obsoletos para {symbol}")
                    return None, None
            
            return df_1m, df_3m
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo klines para {symbol}: {e}")
            return None, None

    def technical_signal_enhanced(self, df_1m, df_3m, symbol):
        """Análisis técnico optimizado para scalping"""
        if df_1m is None or df_3m is None or len(df_1m) < 60:
            return None

        try:
            # Indicadores principales (1m)
            close = df_1m['close']
            high = df_1m['high']
            low = df_1m['low']
            
            # EMA rápidas para scalping
            ema5 = EMAIndicator(close, window=5).ema_indicator()
            ema10 = EMAIndicator(close, window=10).ema_indicator()
            ema20 = EMAIndicator(close, window=20).ema_indicator()
            
            # MACD ajustado para scalping
            macd = MACD(close, window_slow=12, window_fast=26, window_sign=9)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            macd_diff = macd.macd_diff()
            
            # RSI ajustado
            rsi = RSIIndicator(close, window=10).rsi()
            
            # Bollinger Bands ajustados
            bb = BollingerBands(close, window=10, window_dev=1.5)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_middle = bb.bollinger_mavg()
            
            # VWAP para scalping
            typical_price = (high + low + close) / 3
            vwap = (typical_price * df_1m['volume']).cumsum() / df_1m['volume'].cumsum()
            
            # Indicadores de confirmación (3m)
            close_3m = df_3m['close']
            ema10_3m = EMAIndicator(close_3m, window=10).ema_indicator()
            ema20_3m = EMAIndicator(close_3m, window=20).ema_indicator()
            
            latest = -1
            current_price = close.iloc[latest]
            
            # Condiciones LONG para scalping
            long_conditions = [
                ema5.iloc[latest] > ema10.iloc[latest] > ema20.iloc[latest],  # EMA alineadas
                macd_line.iloc[latest] > macd_signal.iloc[latest],  # MACD positivo
                macd_diff.iloc[latest] > 0,  # MACD en crecimiento
                rsi.iloc[latest] > 50 and rsi.iloc[latest] < 70,  # RSI favorable
                current_price > vwap.iloc[latest],  # Precio sobre VWAP
                current_price > bb_middle.iloc[latest],  # Precio sobre BB medio
                ema10_3m.iloc[latest] > ema20_3m.iloc[latest]  # Tendencia 3m
            ]
            
            # Condiciones SHORT para scalping
            short_conditions = [
                ema5.iloc[latest] < ema10.iloc[latest] < ema20.iloc[latest],  # EMA alineadas
                macd_line.iloc[latest] < macd_signal.iloc[latest],  # MACD negativo
                macd_diff.iloc[latest] < 0,  # MACD en decrecimiento
                rsi.iloc[latest] < 50 and rsi.iloc[latest] > 30,  # RSI favorable
                current_price < vwap.iloc[latest],  # Precio bajo VWAP
                current_price < bb_middle.iloc[latest],  # Precio bajo BB medio
                ema10_3m.iloc[latest] < ema20_3m.iloc[latest]  # Tendencia 3m
            ]
            
            # Señales más agresivas para scalping
            if sum(long_conditions) >= 6:  # 6 de 7 condiciones
                if self._can_send_signal(symbol, 'buy'):
                    logger.info(f"SIGNAL: SEÑAL LONG fuerte para {symbol} ({sum(long_conditions)}/7 condiciones)")
                    return "buy"
            elif sum(short_conditions) >= 6:  # 6 de 7 condiciones
                if self._can_send_signal(symbol, 'sell'):
                    logger.info(f"SIGNAL: SEÑAL SHORT fuerte para {symbol} ({sum(short_conditions)}/7 condiciones)")
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
                if filter_item['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(filter_item.get('minNotional', 0))
                elif filter_item['filterType'] == 'LOT_SIZE':
                    min_qty = float(filter_item.get('minQty', 0))
            
            # Verificar valor mínimo notional (valor en USDT)
            if min_notional > 0 and first_order_value < min_notional:
                logger.debug(f"FILTER: {symbol} rechazado - Primera orden ${first_order_value:.2f} < MIN_NOTIONAL ${min_notional:.2f}")
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
            if bot_state['daily_trades'] >= 20:  # Límite diario
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
                    df_1m, df_3m = self.get_klines_for_scalping(symbol)
                    signal = self.technical_signal_enhanced(df_1m, df_3m, symbol)
                    
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