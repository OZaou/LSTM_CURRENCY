# data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import ta  # Librairie pour les indicateurs techniques
import copy

class ForexDataset:
    def __init__(self, ticker, start_date, interval, features, time_step, test_size, validation_size):
        self.ticker = ticker
        self.start_date = start_date
        self.interval = interval
        self.features = features.copy()
        self.original_features = features.copy()  
        self.time_step = time_step
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def load_data(self):
        try:
            logging.info(f"Téléchargement des données pour {self.ticker}")
            data = yf.download(self.ticker, start=self.start_date, interval=self.interval)
            if data.empty:
                raise ValueError(f"Aucune donnée trouvée pour {self.ticker}")
            self.data = data[self.features].dropna()
            self.adjust_for_market_hours()
            self.add_technical_indicators()
            logging.info(f"Données pour {self.ticker} chargées avec succès")
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des données pour {self.ticker}: {e}")
            raise

    def adjust_for_market_hours(self):
        # Ajuster les heures de marché si nécessaire
        self.data = self.data.between_time('00:00', '23:59')

    def add_technical_indicators(self):
        # Ajouter des indicateurs techniques
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close']).rsi()
        self.data['MACD'] = ta.trend.MACD(self.data['Close']).macd()
        self.data['EMA'] = ta.trend.EMAIndicator(self.data['Close']).ema_indicator()
        self.data = self.data.dropna()
        self.features = self.original_features + ['RSI', 'MACD', 'EMA']  # Mettre à jour les features

    def preprocess_data(self):
        data_values = self.data[self.features].values
        data_scaled = self.scaler.fit_transform(data_values)
        X, y = self.create_sequences(data_scaled)
        self.split_data(X, y)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.time_step):
            X.append(data[i:(i + self.time_step)])
            y.append(data[i + self.time_step, self.features.index('Close')])
        return np.array(X), np.array(y)

    def split_data(self, X, y):
        test_split = int(len(X) * (1 - self.test_size))
        val_split = int(test_split * (1 - self.validation_size))
        self.train_data = (X[:val_split], y[:val_split])
        self.validation_data = (X[val_split:test_split], y[val_split:test_split])
        self.test_data = (X[test_split:], y[test_split:])
        logging.info(f"Données divisées en ensembles d'entraînement, de validation et de test pour {self.ticker}")

    def get_input_size(self):
        return len(self.features)
