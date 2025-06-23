import os
import time
import pandas as pd
import requests
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from dotenv import load_dotenv
from collections import defaultdict, deque

# Configurar logging robusto
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_bot.log'),
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

# Parámetros optimizados para scalping
MAX_OPEN_TRADES = 6  # Incrementado para más oportunidades
MIN_VOLUME = 100_000_000  # Mayor volumen para mejor liquidez
MAX_SPREAD_PERCENT = 0.08  # Spread más estricto
MIN_PRICE_CHANGE_1H = 0.5  # Mínimo cambio de precio en 1h
MAX_PRICE_CHANGE_1H = 8.0  # Máximo cambio para evitar volatilidad extrema

# Control de rate limits optimizado
API_CALL_DELAY = 1.5  # Más rápido para scalping
SCAN_INTERVAL = 15    # Escaneo más frecuente
SIGNAL_COOLDOWN = 600  # 10 minutos entre señales del mismo símbolo

# Configuración de risk management
DAILY_LOSS_LIMIT = 0.05  # Máximo 5% de pérdida diaria
MAX_DRAWDOWN = 0.15  # Máximo 15% de drawdown
WIN_RATE_THRESHOLD = 0.45  # Mínimo 45% de win rate

# Estado del bot mejorado
bot_state = {
    'last_signals': {},
    'failed_symbols': set(),
    'daily_trades': 0,
    'daily_pnl': 0.0,
    'win_count': 0,
    'loss_count': 0,
    'last_reset': datetime.now().date(),
    'symbol_performance': defaultdict(list),
    'market_regime': 'neutral',  # trending, ranging, volatile
    'high_performing_symbols': set(),
    'blacklisted_symbols': set(),
    'recent_trades': deque(maxlen=100)
}

class ScalpingBot:
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
            self._initialize_symbol_filters()
            
        except Exception as e:
            logger.error(f"ERROR: Error inicializando bot: {e}")
            raise

    def _initialize_symbol_filters(self):
        """Inicializa filtros de símbolos basados en rendimiento histórico"""
        try:
            # Obtener top performers por volumen y volatilidad controlada
            tickers = self.client.futures_24hr_ticker()
            
            # Filtrar y rankear símbolos
            ranked_symbols = []
            for ticker in tickers:
                if not ticker['symbol'].endswith('USDT'):
                    continue
                    
                volume = float(ticker['quoteVolume'])
                price_change = abs(float(ticker['priceChangePercent']))
                
                if (volume > MIN_VOLUME and 
                    MIN_PRICE_CHANGE_1H <= price_change <= MAX_PRICE_CHANGE_1H):
                    
                    # Score basado en volumen y volatilidad óptima
                    score = volume * (1 + min(price_change / 5.0, 1.0))
                    ranked_symbols.append((ticker['symbol'], score, price_change))
            
            # Tomar top 30 símbolos
            ranked_symbols.sort(key=lambda x: x[1], reverse=True)
            self.preferred_symbols = [s[0] for s in ranked_symbols[:30]]
            
            logger.info(f"INIT: {len(self.preferred_symbols)} símbolos preferenciales cargados")
            
        except Exception as e:
            logger.error(f"ERROR: Error inicializando filtros: {e}")
            self.preferred_symbols = []

    def _test_connection(self):
        """Prueba la conexión a Binance con reintentos"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"TEST: Probando conexión a Binance (intento {attempt + 1}/{max_retries})...")
                
                server_time = self.client.get_server_time()
                logger.info(f"TEST: Hora del servidor Binance: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
                
                account = self.client.futures_account()
                logger.info("SUCCESS: Conexión a Binance Futures establecida correctamente")
                return True
                
            except Exception as e:
                logger.error(f"ERROR: Error de conexión (intento {attempt + 1}): {e}")
                
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.info(f"WAIT: Esperando {wait_time} segundos antes del siguiente intento...")
                time.sleep(wait_time)
        
        raise Exception("ERROR: No se pudo establecer conexión con Binance después de 3 intentos")
    
    def capital_requerido_por_moneda(self, capital_total):
        primera = capital_total * 0.1
        ordenes = [primera * (1.2 ** i) for i in range(5)]
        return sum(ordenes)

    def detect_market_regime(self):
        """Detecta el régimen de mercado actual"""
        try:
            # Usar BTC como proxy del mercado
            klines = self.client.futures_klines(symbol='BTCUSDT', interval='1h', limit=24)
            if not klines:
                return 'neutral'
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            close_prices = pd.to_numeric(df['close'])
            volumes = pd.to_numeric(df['volume'])
            
            # Calcular métricas
            price_volatility = close_prices.pct_change().std() * 100
            trend_strength = abs(close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100
            volume_trend = volumes.tail(6).mean() / volumes.head(6).mean()
            
            # Determinar régimen
            if trend_strength > 3 and volume_trend > 1.2:
                regime = 'trending'
            elif price_volatility > 4 and volume_trend > 1.5:
                regime = 'volatile'
            else:
                regime = 'ranging'
            
            if bot_state['market_regime'] != regime:
                logger.info(f"MARKET: Régimen cambiado de {bot_state['market_regime']} a {regime}")
                bot_state['market_regime'] = regime
            
            return regime
            
        except Exception as e:
            logger.error(f"ERROR: Error detectando régimen de mercado: {e}")
            return 'neutral'

    def get_scalping_symbols(self):
        """Obtiene símbolos optimizados para scalping"""
        try:
            # Detectar régimen de mercado
            market_regime = self.detect_market_regime()
            
            # Filtrar símbolos basado en rendimiento reciente
            valid_symbols = []
            
            for symbol in self.preferred_symbols:
                if symbol in bot_state['blacklisted_symbols']:
                    continue
                    
                # Verificar rendimiento reciente
                recent_performance = bot_state['symbol_performance'].get(symbol, [])
                if len(recent_performance) >= 5:
                    win_rate = sum(1 for p in recent_performance[-5:] if p > 0) / 5
                    if win_rate < 0.3:  # Menos del 30% de éxito
                        continue
                
                # Verificar liquidez en tiempo real
                if self._check_symbol_liquidity(symbol):
                    valid_symbols.append(symbol)
                    
                if len(valid_symbols) >= 20:  # Limitar búsqueda
                    break
            
            logger.info(f"SYMBOLS: {len(valid_symbols)} símbolos válidos para scalping")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo símbolos: {e}")
            return []

    def _check_symbol_liquidity(self, symbol):
        """Verifica liquidez del símbolo"""
        try:
            depth = self.client.futures_order_book(symbol=symbol, limit=5)
            
            # Verificar spread
            best_bid = float(depth['bids'][0][0])
            best_ask = float(depth['asks'][0][0])
            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            
            # Verificar profundidad del order book
            bid_depth = sum(float(bid[1]) for bid in depth['bids'])
            ask_depth = sum(float(ask[1]) for ask in depth['asks'])
            
            return (spread_percent <= MAX_SPREAD_PERCENT and 
                    bid_depth > 10 and ask_depth > 10)
            
        except Exception as e:
            logger.debug(f"DEBUG: Error verificando liquidez {symbol}: {e}")
            return False

    def get_klines_scalping(self, symbol):
        """Obtiene datos optimizados para scalping"""
        try:
            # Múltiples timeframes para confirmación
            klines_1m = self.client.futures_klines(symbol=symbol, interval='1m', limit=60)
            klines_3m = self.client.futures_klines(symbol=symbol, interval='3m', limit=40)
            klines_5m = self.client.futures_klines(symbol=symbol, interval='5m', limit=30)
            
            if not all([klines_1m, klines_3m, klines_5m]):
                return None, None, None
            
            def create_dataframe(klines):
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df if not df['close'].isna().any() else None
            
            df_1m = create_dataframe(klines_1m)
            df_3m = create_dataframe(klines_3m)
            df_5m = create_dataframe(klines_5m)
            
            return df_1m, df_3m, df_5m
            
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo klines para {symbol}: {e}")
            return None, None, None

    def scalping_signal_advanced(self, df_1m, df_3m, df_5m, symbol):
        """Señales avanzadas de scalping"""
        if not all([df_1m is not None, df_3m is not None, df_5m is not None]):
            return None

        try:
            # Análisis en 1m (entrada)
            close_1m = df_1m['close']
            high_1m = df_1m['high']
            low_1m = df_1m['low']
            volume_1m = df_1m['volume']
            
            # Indicadores rápidos para scalping
            ema8_1m = EMAIndicator(close_1m, window=8).ema_indicator()
            ema13_1m = EMAIndicator(close_1m, window=13).ema_indicator()
            ema21_1m = EMAIndicator(close_1m, window=21).ema_indicator()
            
            # RSI rápido
            rsi_1m = RSIIndicator(close_1m, window=7).rsi()
            
            # Stochastic para momentum
            stoch = StochasticOscillator(high_1m, low_1m, close_1m, window=14, smooth_window=3)
            stoch_k = stoch.stoch()
            stoch_d = stoch.stoch_signal()
            
            # Volume confirmation
            volume_sma = volume_1m.rolling(window=10).mean()
            
            # Bollinger Bands para volatilidad
            bb_1m = BollingerBands(close_1m, window=20, window_dev=2)
            bb_upper = bb_1m.bollinger_hband()
            bb_lower = bb_1m.bollinger_lband()
            bb_middle = bb_1m.bollinger_mavg()
            
            # Análisis en 3m (confirmación)
            close_3m = df_3m['close']
            ema21_3m = EMAIndicator(close_3m, window=21).ema_indicator()
            rsi_3m = RSIIndicator(close_3m, window=14).rsi()
            
            # ADX para fuerza de tendencia
            adx_3m = ADXIndicator(df_3m['high'], df_3m['low'], df_3m['close'], window=14)
            adx_value = adx_3m.adx()
            
            # Análisis en 5m (filtro de tendencia)
            close_5m = df_5m['close']
            ema50_5m = EMAIndicator(close_5m, window=50).ema_indicator()
            
            latest = -1
            current_price = close_1m.iloc[latest]
            
            # Verificar que tenemos datos suficientes
            required_indicators = [
                ema8_1m, ema13_1m, ema21_1m, rsi_1m, stoch_k, stoch_d,
                volume_sma, bb_upper, bb_lower, bb_middle, ema21_3m, 
                rsi_3m, adx_value, ema50_5m
            ]
            
            if any(pd.isna(indicator.iloc[latest]) for indicator in required_indicators):
                return None

            # Filtros de mercado
            market_regime = bot_state['market_regime']
            
            # Calcular momentum y volatilidad
            price_momentum = (current_price - close_1m.iloc[-5]) / close_1m.iloc[-5] * 100
            volume_ratio = volume_1m.iloc[latest] / volume_sma.iloc[latest]
            bb_position = (current_price - bb_lower.iloc[latest]) / (bb_upper.iloc[latest] - bb_lower.iloc[latest])
            
            # Condiciones base
            trending_up_1m = ema8_1m.iloc[latest] > ema13_1m.iloc[latest] > ema21_1m.iloc[latest]
            trending_down_1m = ema8_1m.iloc[latest] < ema13_1m.iloc[latest] < ema21_1m.iloc[latest]
            trending_up_3m = close_3m.iloc[latest] > ema21_3m.iloc[latest]
            trending_down_3m = close_3m.iloc[latest] < ema21_3m.iloc[latest]
            strong_trend = adx_value.iloc[latest] > 25
            
            # Señales LONG optimizadas para scalping
            long_conditions = [
                trending_up_1m,  # Tendencia alcista en 1m
                trending_up_3m,  # Confirmación en 3m
                current_price > ema50_5m.iloc[latest],  # Filtro de tendencia 5m
                30 < rsi_1m.iloc[latest] < 70,  # RSI en zona favorable
                rsi_3m.iloc[latest] > 45,  # RSI 3m no sobrevendido
                stoch_k.iloc[latest] > stoch_d.iloc[latest],  # Stochastic alcista
                stoch_k.iloc[latest] > 20,  # Stochastic no sobrevendido
                volume_ratio > 1.2,  # Volumen por encima del promedio
                strong_trend or market_regime == 'trending',  # Tendencia fuerte
                0.2 < bb_position < 0.8,  # No en extremos de BB
                price_momentum > -0.5  # Momentum no muy negativo
            ]
            
            # Señales SHORT optimizadas para scalping
            short_conditions = [
                trending_down_1m,  # Tendencia bajista en 1m
                trending_down_3m,  # Confirmación en 3m
                current_price < ema50_5m.iloc[latest],  # Filtro de tendencia 5m
                30 < rsi_1m.iloc[latest] < 70,  # RSI en zona favorable
                rsi_3m.iloc[latest] < 55,  # RSI 3m no sobrecomprado
                stoch_k.iloc[latest] < stoch_d.iloc[latest],  # Stochastic bajista
                stoch_k.iloc[latest] < 80,  # Stochastic no sobrecomprado
                volume_ratio > 1.2,  # Volumen por encima del promedio
                strong_trend or market_regime == 'trending',  # Tendencia fuerte
                0.2 < bb_position < 0.8,  # No en extremos de BB
                price_momentum < 0.5  # Momentum no muy positivo
            ]
            
            # Ajustar umbrales según régimen de mercado
            min_conditions = 8 if market_regime == 'trending' else 9
            
            if sum(long_conditions) >= min_conditions:
                if self._can_send_signal(symbol, 'buy'):
                    logger.info(f"SCALP LONG: {symbol} | Condiciones: {sum(long_conditions)}/{len(long_conditions)} | Régimen: {market_regime}")
                    return "buy"
                    
            elif sum(short_conditions) >= min_conditions:
                if self._can_send_signal(symbol, 'sell'):
                    logger.info(f"SCALP SHORT: {symbol} | Condiciones: {sum(short_conditions)}/{len(short_conditions)} | Régimen: {market_regime}")
                    return "sell"
                    
            return None

        except Exception as e:
            logger.error(f"ERROR: Error en análisis de scalping para {symbol}: {e}")
            return None

    def _can_send_signal(self, symbol, side):
        """Verifica si se puede enviar una señal con control avanzado"""
        key = f"{symbol}_{side}"
        last_signal_time = bot_state['last_signals'].get(key, 0)
        current_time = time.time()
        
        # Cooldown básico
        if current_time - last_signal_time < SIGNAL_COOLDOWN:
            return False
        
        # Verificar win rate del símbolo
        recent_performance = bot_state['symbol_performance'].get(symbol, [])
        if len(recent_performance) >= 3:
            recent_win_rate = sum(1 for p in recent_performance[-3:] if p > 0) / 3
            if recent_win_rate < 0.3:
                return False
        
        # Verificar límites de riesgo
        if not self._check_risk_limits():
            return False
            
        bot_state['last_signals'][key] = current_time
        return True

    def _check_risk_limits(self):
        """Verifica límites de riesgo"""
        try:
            # Verificar pérdida diaria
            if bot_state['daily_pnl'] < -abs(self.get_available_usdt() * DAILY_LOSS_LIMIT):
                logger.warning("WARNING: Límite de pérdida diaria alcanzado")
                return False
            
            # Verificar win rate
            total_trades = bot_state['win_count'] + bot_state['loss_count']
            if total_trades > 10:
                win_rate = bot_state['win_count'] / total_trades
                if win_rate < WIN_RATE_THRESHOLD:
                    logger.warning(f"WARNING: Win rate bajo: {win_rate:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Error verificando límites de riesgo: {e}")
            return False

    def get_open_positions(self):
        """Obtiene posiciones abiertas"""
        try:
            positions = self.client.futures_account()['positions']
            open_positions = []
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    pnl = float(pos.get('unrealizedPnl', 0))
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
        """Obtiene balance USDT"""
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    return float(balance['balance'])
            return 0
        except Exception as e:
            logger.error(f"ERROR: Error obteniendo balance: {e}")
            return 0

    def send_signal_to_finandy(self, symbol, side):
        """Envía señal a Finandy con retry logic"""
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
                    timeout=15,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    logger.info(f"SUCCESS: SEÑAL ENVIADA: {symbol} | {side.upper()}")
                    bot_state['daily_trades'] += 1
                    
                    # Registrar trade
                    trade_data = {
                        'symbol': symbol,
                        'side': side,
                        'timestamp': datetime.now(),
                        'sent': True
                    }
                    bot_state['recent_trades'].append(trade_data)
                    
                    return True
                else:
                    logger.warning(f"WARNING: Respuesta Finandy: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"ERROR: Error enviando señal (intento {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        return False

    def update_performance_tracking(self):
        """Actualiza seguimiento de rendimiento"""
        try:
            positions = self.get_open_positions()
            
            # Actualizar PnL diario
            daily_pnl = sum(pos['pnl'] for pos in positions)
            bot_state['daily_pnl'] = daily_pnl
            
            # Actualizar estadísticas por símbolo (esto requeriría lógica adicional)
            # Por ahora, solo logging
            if positions:
                logger.info(f"PERFORMANCE: PnL diario: ${daily_pnl:.2f} | Posiciones: {len(positions)}")
                
        except Exception as e:
            logger.error(f"ERROR: Error actualizando performance: {e}")

    def reset_daily_counters(self):
        """Resetea contadores diarios"""
        today = datetime.now().date()
        if bot_state['last_reset'] != today:
            # Guardar estadísticas del día anterior
            if bot_state['daily_trades'] > 0:
                win_rate = bot_state['win_count'] / (bot_state['win_count'] + bot_state['loss_count']) if (bot_state['win_count'] + bot_state['loss_count']) > 0 else 0
                logger.info(f"DAILY SUMMARY: Trades: {bot_state['daily_trades']} | Win Rate: {win_rate:.2%} | PnL: ${bot_state['daily_pnl']:.2f}")
            
            # Reset
            bot_state['daily_trades'] = 0
            bot_state['daily_pnl'] = 0.0
            bot_state['win_count'] = 0
            bot_state['loss_count'] = 0
            bot_state['failed_symbols'].clear()
            bot_state['last_reset'] = today
            
            logger.info("RESET: Contadores diarios reseteados")

    def run_scalping_cycle(self):
        """Ejecuta ciclo de scalping"""
        try:
            self.reset_daily_counters()
            self.update_performance_tracking()

            if bot_state['daily_trades'] >= 20:
                logger.info("LIMIT: Límite diario de trades alcanzado")
                return

            if not self._check_risk_limits():
                logger.warning("WARNING: Límites de riesgo alcanzados")
                return

            total_usdt = self.get_available_usdt()
            capital_por_operacion = self.capital_requerido_por_moneda(total_usdt)

            open_positions = self.get_open_positions()
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info(f"LIMIT: Límite de posiciones alcanzado ({len(open_positions)}/{MAX_OPEN_TRADES})")
                return

            posiciones_disponibles = MAX_OPEN_TRADES - len(open_positions)
            capital_requerido_total = capital_por_operacion * posiciones_disponibles

            if total_usdt < capital_por_operacion:
                logger.warning(f"WARNING: Capital insuficiente incluso para una operación: ${total_usdt:.2f} < ${capital_por_operacion:.2f}")
                return

            symbols = self.get_scalping_symbols()
            if not symbols:
                logger.error("ERROR: No se pudieron obtener símbolos")
                return

            open_symbols = {pos['symbol'] for pos in open_positions}
            signals_sent = 0
            max_signals_per_cycle = posiciones_disponibles

            logger.info(f"SCALP SCAN: Analizando {len(symbols)} símbolos para scalping...")

            for symbol in symbols:
                if signals_sent >= max_signals_per_cycle:
                    break

                if symbol in open_symbols or symbol in bot_state['failed_symbols']:
                    continue

                # Verificar si hay capital suficiente para una nueva operación
                capital_restante = self.get_available_usdt()
                if capital_restante < capital_por_operacion:
                    logger.warning(f"SKIP: Capital insuficiente para abrir {symbol}: ${capital_restante:.2f} < ${capital_por_operacion:.2f}")
                    continue

                try:
                    df_1m, df_3m, df_5m = self.get_klines_scalping(symbol)
                    signal = self.scalping_signal_advanced(df_1m, df_3m, df_5m, symbol)

                    if signal:
                        if self.send_signal_to_finandy(symbol, signal):
                            signals_sent += 1
                            open_symbols.add(symbol)

                except Exception as e:
                    logger.error(f"ERROR: Error procesando {symbol}: {e}")
                    bot_state['failed_symbols'].add(symbol)

                time.sleep(API_CALL_DELAY)

            logger.info(f"SCALP CYCLE: {signals_sent} señales enviadas | Régimen: {bot_state['market_regime']}")

        except Exception as e:
            logger.error(f"ERROR: Error en ciclo de scalping: {e}")


def main():
    """Función principal del bot de scalping"""
    logger.info("START: Iniciando Scalping Bot Avanzado...")
    
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        logger.info("NETWORK: Conectividad verificada")
    except Exception as e:
        logger.error(f"ERROR: Sin conexión a internet: {e}")
        return
    
    try:
        bot = ScalpingBot()
        logger.info("SUCCESS: Bot de scalping inicializado")
        
        while True:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("SCALP CYCLE: Iniciando ciclo de scalping...")
            
            try:
                bot.run_scalping_cycle()
            except Exception as e:
                logger.error(f"ERROR: Error en ciclo: {e}")
                time.sleep(30)
                continue
            
            execution_time = time.time() - start_time
            logger.info(f"TIMING: Ciclo completado en {execution_time:.2f}s")
            
            # Espera más corta para scalping
            sleep_time = max(SCAN_INTERVAL - execution_time, 5)
            logger.info(f"WAIT: Esperando {sleep_time:.1f}s hasta próximo ciclo...")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("STOP: Bot detenido por usuario")
    except Exception as e:
        logger.error(f"ERROR: Error crítico: {e}")
        logger.info("RESTART: Reiniciando en 60 segundos...")
        time.sleep(60)
        main()

if __name__ == "__main__":
    main()