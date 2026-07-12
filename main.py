import asyncio, json, websockets, pandas as pd, threading, requests
from datetime import datetime, timezone, timedelta
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
PRODUCT_ID = "BTC-USD"

app = Flask(__name__)
CORS(app)

history_log = deque(maxlen=100)
latest_data = {
    "prediction_early": "NONE", "confidence_early": 0,
    "prediction_late": "NONE", "confidence_late": 0,
    "accuracy": 0, "price": 0, "block_start": "", "rsi": 50, "net_liquidity": 0,
    "correct": None, "total_blocks": 0, "patterns": [], "whale_signal": 0
}

class BTCPredictor:
    def __init__(self):
        self.trades = []
        self.block_start_price = None
        self.block_start_time = None
        self.model = PassiveAggressiveClassifier(C=1.0, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.correct = 0
        self.total = 0
        self.last_features = None
        self.last_minute_trades = []
        self.whale_memory = defaultdict(lambda: {"buys": 0, "sells": 0, "wins": 0, "losses": 0, "last_seen": None})
        self.WHALE_THRESHOLD = 0.5
        self.feature_names = [
            'return_so_far', 'volatility', 'buy_volume', 'sell_volume', 'trade_count',
            'max_drawdown', 'max_runup', 'minute_of_block', 'liquidity_imbalance',
            'volume_imbalance', 'wave_count', 'wave_amplitude', 'momentum_shift',
            'macd_histogram', 'macd_slope', 'histogram_slope', 'rsi_slope',
            'whale_buy_pressure', 'whale_sell_pressure', 'whale_net_signal'
        ]

    def get_block_times(self, dt):
        minute = (dt.minute // 15) * 15
        start = dt.replace(minute=minute, second=0, microsecond=0)
        end = start + timedelta(minutes=15)
        return start, end

    def track_whales_last_minute(self):
        if not self.last_minute_trades:
            return 0, 0, 0
        whale_buys = defaultdict(float)
        whale_sells = defaultdict(float)
        for t in self.last_minute_trades:
            size = float(t['size'])
            if size >= self.WHALE_THRESHOLD:
                order_id = t.get('maker_order_id', t.get('taker_order_id', 'unknown'))
                if t['side'] == 'buy':
                    whale_buys[order_id] += size
                else:
                    whale_sells[order_id] += size
        total_whale_buy = sum(whale_buys.values())
        total_whale_sell = sum(whale_sells.values())
        whale_signal = 0
        for whale_id, size in whale_buys.items():
            stats = self.whale_memory[whale_id]
            if stats['wins'] + stats['losses'] > 2:
                win_rate = stats['wins'] / (stats['wins'] + stats['losses'])
                whale_signal += size * (win_rate - 0.5) * 2
        for whale_id, size in whale_sells.items():
            stats = self.whale_memory[whale_id]
            if stats['wins'] + stats['losses'] > 2:
                win_rate = stats['wins'] / (stats['wins'] + stats['losses'])
                whale_signal -= size * (win_rate - 0.5) * 2
        return total_whale_buy, total_whale_sell, whale_signal

    def update_whale_memory(self, actual):
        block_went_up = actual == 1
        for t in self.last_minute_trades:
            size = float(t['size'])
            if size >= self.WHALE_THRESHOLD:
                order_id = t.get('maker_order_id', t.get('taker_order_id', 'unknown'))
                whale = self.whale_memory[order_id]
                if t['side'] == 'buy':
                    whale['buys'] += size
                    if block_went_up: whale['wins'] += 1
                    else: whale['losses'] += 1
                else:
                    whale['sells'] += size
                    if not block_went_up: whale['wins'] += 1
                    else: whale['losses'] += 1
                whale['last_seen'] = datetime.now(timezone.utc)

    def extract_features(self):
        if len(self.trades) < 26: return None
        df = pd.DataFrame(self.trades)
        df['price'] = df['price'].astype(float)
        df['size'] = df['size'].astype(float)
        prices = df['price'].values
        current_price = prices[-1]
        start_price = self.block_start_price

        try:
            r = requests.get('https://api.exchange.coinbase.com/products/BTC-USD/book?level=2', timeout=2)
            book = r.json()
            buy_walls = sum(float(size) for price, size, _ in book['bids'] if float(price) > current_price * 0.995)
            sell_walls = sum(float(size) for price, size, _ in book['asks'] if float(price) < current_price * 1.005)
            liquidity_imbalance = (buy_walls - sell_walls) / (buy_walls + sell_walls + 1e-9)
        except:
            buy_walls, sell_walls, liquidity_imbalance = 0, 0, 0

        peaks, _ = find_peaks(prices, distance=5, prominence=np.std(prices)*0.1)
        troughs, _ = find_peaks(-prices, distance=5, prominence=np.std(prices)*0.1)
        wave_count = len(peaks) + len(troughs)
        amplitude = (np.mean(prices[peaks]) - np.mean(prices[troughs])) / start_price if len(peaks) > 0 and len(troughs) > 0 else 0

        ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd.iloc[-1] - signal.iloc[-1]
        macd_slope = macd.iloc[-1] - macd.iloc[-5] if len(macd) > 5 else 0
        histogram_slope = histogram - (macd.iloc[-2] - signal.iloc[-2]) if len(macd) > 1 else 0

        deltas = pd.Series(prices).diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=14).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        rsi_slope = rsi.iloc[-1] - rsi.iloc[-5] if len(rsi) > 5 else 0

        if len(prices) > 10:
            recent_slope = np.polyfit(range(10), prices[-10:], 1)[0]
            early_slope = np.polyfit(range(10), prices[:10], 1)[0]
            momentum_shift = (recent_slope - early_slope) / start_price
        else:
            momentum_shift = 0

        returns = df['price'].pct_change().dropna()
        buy_vol = df[df['side'] == 'buy']['size'].sum()
        sell_vol = df[df['side'] == 'sell']['size'].sum()
        whale_buy_pressure, whale_sell_pressure, whale_net_signal = self.track_whales_last_minute()

        features = {
            'return_so_far': (current_price - start_price) / start_price,
            'volatility': returns.std() if len(returns) > 1 else 0,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'trade_count': len(df),
            'max_drawdown': (df['price'].min() - start_price) / start_price,
            'max_runup': (df['price'].max() - start_price) / start_price,
            'minute_of_block': (datetime.now(timezone.utc) - self.block_start_time).seconds / 60,
            'liquidity_imbalance': liquidity_imbalance,
            'volume_imbalance': (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9),
            'wave_count': wave_count,
            'wave_amplitude': amplitude,
            'momentum_shift': momentum_shift,
            'macd_histogram': histogram / start_price,
            'macd_slope': macd_slope / start_price,
            'histogram_slope': histogram_slope / start_price,
            'rsi': rsi_current,
            'rsi_slope': rsi_slope,
            'whale_buy_pressure': whale_buy_pressure,
            'whale_sell_pressure': whale_sell_pressure,
            'whale_net_signal': whale_net_signal
        }

        self.current_price = current_price
        self.buy_walls = buy_walls
        self.sell_walls = sell_walls
        self.last_feature_dict = features
        return np.array(list(features.values())).reshape(1, -1)

    def predict(self):
        features = self.extract_features()
        if features is None or not self.trained:
            self.last_features = features
            return None, 0.5
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        confidence = abs(self.model.decision_function(features_scaled)[0])
        confidence = 1 / (1 + np.exp(-confidence))
        self.last_features = features
        return int(prediction), confidence

    def update_model(self, features, actual):
        if features is None: return
        if not self.trained:
            self.scaler.partial_fit(features)
            self.model.partial_fit(features, [actual], classes=[0, 1])
            self.trained = True
        else:
            self.scaler.partial_fit(features)
            self.model.partial_fit(features, [actual])

    def get_top_patterns(self):
        if not self.trained or self.last_features is None: return []
        coef = self.model.coef_[0]
        features = self.last_features[0]
        impacts = [(self.feature_names[i], coef[i] * features[i]) for i in range(len(coef))]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": name, "impact": round(val, 4)} for name, val in impacts[:5]]

    def log_result(self, block_start, pred_early, conf_early, pred_late, conf_late, actual, end_price):
        global latest_data, history_log
        self.total += 1
        correct = int(pred_late == actual) if pred_late is not None else None
        if correct is not None: self.correct += correct
        accuracy = self.correct / self.total * 100 if self.total > 0 else 0
        self.update_whale_memory(actual)

        log_entry = {
            'block_start_utc': block_start.strftime('%Y-%m-%d %H:%M:%S'),
            'start_price': self.block_start_price,
            'end_price': end_price,
            'pred_early': 'UP' if pred_early == 1 else 'DOWN' if pred_early == 0 else 'NONE',
            'conf_early': round(conf_early * 100, 1),
            'pred_late': 'UP' if pred_late == 1 else 'DOWN' if pred_late == 0 else 'NONE',
            'conf_late': round(conf_late * 100, 1),
            'actual': 'UP' if actual == 1 else 'DOWN',
            'correct_late': correct,
            'running_accuracy': round(accuracy, 2),
            'whale_buy': round(self.last_feature_dict.get('whale_buy_pressure', 0), 2),
            'whale_sell': round(self.last_feature_dict.get('whale_sell_pressure', 0), 2),
            'whale_signal': round(self.last_feature_dict.get('whale_net_signal', 0), 3)
        }
        history_log.append(log_entry)

        latest_data.update({
            "prediction_early": log_entry['pred_early'],
            "confidence_early": log_entry['conf_early'],
            "prediction_late": log_entry['pred_late'],
            "confidence_late": log_entry['conf_late'],
            "accuracy": log_entry['running_accuracy'],
            "price": end_price,
            "block_start": block_start.strftime('%H:%M UTC'),
            "rsi": round(self.last_feature_dict.get('rsi', 50), 1),
            "net_liquidity": round((self.buy_walls - self.sell_walls) / 1e6, 2),
            "correct": correct,
            "total_blocks": self.total,
            "patterns": self.get_top_patterns(),
            "whale_signal": log_entry['whale_signal']
        })

        print(f"\n{'='*70}\nBlock: {block_start.strftime('%H:%M')} - ${end_price:.2f}")
        print(f"EARLY @2min: {log_entry['pred_early']} @ {log_entry['conf_early']}%")
        print(f"LATE @13min: {log_entry['pred_late']} @ {log_entry['conf_late']}% -> {'✅' if correct else '❌'}")
        print(f"WHALES: Buy {log_entry['whale_buy']} BTC | Sell {log_entry['whale_sell']} BTC | Signal: {log_entry['whale_signal']}")
        print(f"Actual: {log_entry['actual']} | Accuracy: {self.correct}/{self.total} = {accuracy:.1f}%")
        print(f"Whales Tracked: {len(self.whale_memory)}\n{'='*70}\n")

async def run_bot():
    bot = BTCPredictor()
    async with websockets.connect(COINBASE_WS) as ws:
        await ws.send(json.dumps({"type": "subscribe", "product_ids": [PRODUCT_ID], "channels": ["matches"]}))
        print(f"PRIVATE BOT ACTIVE: Whale tracking + RL")
        current_block_start = None
        pred_early, conf_early = None, 0.5
        pred_late, conf_late = None, 0.5

        async for message in ws:
            data = json.loads(message)
            if data.get('type')!= 'match': continue
            trade_time = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
            block_start, block_end = bot.get_block_times(trade_time)

            if block_start!= current_block_start:
                if current_block_start and bot.trades:
                    end_price = float(bot.trades[-1]['price'])
                    actual = 1 if end_price > bot.block_start_price else 0
                    bot.log_result(current_block_start, pred_early, conf_early, pred_late, conf_late, actual, end_price)
                    if bot.last_features is not None:
                        bot.update_model(bot.last_features, actual)
                current_block_start = block_start
                bot.trades = []
                bot.last_minute_trades = []
                bot.block_start_price = float(data['price'])
                bot.block_start_time = block_start
                pred_early, conf_early = None, 0.5
                pred_late, conf_late = None, 0.5
                print(f"\n[{block_start.strftime('%H:%M UTC')}] New block. Start: ${bot.block_start_price:.2f}")

            trade_data = {'price': float(data['price']), 'size': float(data['size']), 'side': data['side'],
                         'time': trade_time, 'maker_order_id': data.get('maker_order_id'), 'taker_order_id': data.get('taker_order_id')}
            bot.trades.append(trade_data)

            minutes_into_block = (trade_time - current_block_start).seconds / 60
            if minutes_into_block >= 14:
                bot.last_minute_trades.append(trade_data)

            if 2 <= minutes_into_block < 3 and pred_early is None:
                pred_early, conf_early = bot.predict()
                if pred_early is not None:
                    print(f"[{trade_time.strftime('%H:%M:%S')}] EARLY @2min: {'UP' if pred_early else 'DOWN'} @ {conf_early*100:.1f}%")

            if 13 <= minutes_into_block < 14 and pred_late is None:
                pred_late, conf_late = bot.predict()
                if pred_late is not None:
                    print(f"[{trade_time.strftime('%H:%M:%S')}] LATE @13min: {'UP' if pred_late else 'DOWN'} @ {conf_late*100:.1f}%")

@app.route('/data')
def get_data():
    return jsonify(latest_data)

@app.route('/history')
def get_history():
    return jsonify(list(history_log))

@app.route('/patterns')
def get_patterns():
    return jsonify({"patterns": latest_data.get("patterns", []), "total_blocks": latest_data.get("total_blocks", 0)})

@app.route('/')
def health():
    return "BTC Predictor Online - Private"

def start_bot():
    asyncio.run(run_bot())

if __name__ == "__main__":
    threading.Thread(target=start_bot, daemon=True).start()
    app.run(host='0.0.0.0', port=10000)
