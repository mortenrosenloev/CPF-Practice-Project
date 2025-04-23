import numpy as np
import pandas as pd
import talib
import tpqoa
import datetime
import q

lags = 10  # same as used for model training
timeframe = '5min'  # timeframe should match the model training

class KNNTrader(tpqoa.tpqoa):
    """
    Class that performs the following steps:
    1. Establish API connectin to OANDA.
    2. Retrieve and resample tick-data.
    3. Add features when sufficient data are gathered.
    4. Normalize data.
    5. Add lags.
    4. Create long/short signals using the imported trained model.
    5. Place orders.
    6. Log activities using `q`.
    9. Close any open trades and save data to csv after a predefined
       number of minutes.
    """
    
    def __init__(self, creds_file, model, scaler, selected_features,
                 units, duration_minutes=60, log_minutes=None, verbose=False):
        super().__init__(creds_file)
        self.model = model # pre-trained model
        self.scaler = scaler # for data normalization
        self.selected_features = selected_features
        self.units = units  # number of trading units
        self.position = 0 # initialize position as neutral
        self.tick_data = pd.DataFrame() # collect streaming data
        self.min_bars = 45 #rolling window (30), slope (5), lags (10)
        if verbose:
            self.suppress = False
        else:
            self.suppress = True

        # Instantiate dataframe for normalized data
        self.data_ = pd.DataFrame()

        # Prepare for API connectivity logging
        self.logging_enabled = isinstance(log_minutes, (int, float)) and log_minutes > 0
        self.logging_freq = log_minutes if self.logging_enabled else 0
        self.last_log = -self.logging_freq  # First log on start of script
        
        # set timer
        self.duration_minutes = duration_minutes
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(minutes=self.duration_minutes)
    
    def on_success(self, time, bid, ask):
        now = datetime.datetime.now()
        
        # Logging of API connectivity
        if self.logging_enabled:
            min_since_start = int((now - self.start_time).total_seconds() // 60)
            if min_since_start - self.last_log >= self.logging_freq:
                msg = f'Status {now}: Live streaming'
                print(msg)
                q(msg)
                self.last_log = min_since_start
        
        # Check timer for shutdown
        if now >= self.end_time:
            # Close any open positions and stop streaming
            print(80*'=')
            print(f'Streaming ended after {self.duration_minutes} minutes')
            print(f'Time: {now} | Closing any open positions')
            print('Closing any open positions')
            print(80*'=')              
            order = self.create_order(self.stream_instrument,
                          units=-self.position * self.units,
                          suppress=self.suppress, ret=True)
            q(now)
            q(order)
            self.data.to_csv(f'data_{now}.csv')
            self.stop_stream = True
            return
        
        # Collect tick data
        mid = (bid + ask) / 2
        df = pd.DataFrame({'bid': bid, 'mid': mid, 'ask': ask},
                          index=[pd.Timestamp(time)])
        self.tick_data = pd.concat([self.tick_data, df])
        
        # Resample data to open, hihg, low, close prices based om mid-price
        self.data = self.tick_data['mid'].resample(
            timeframe, label='right').ohlc().ffill()
        self.data['r'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data = self.data.iloc[:-1]  # the last line is incomplete
        
        # Generate signals
        if len(self.data) > self.min_bars:
            self.min_bars += 1
            self._data_preprocessing()
            pred = self.model.predict(self.X_live)
            self.data.loc[self.data_.index, 'pred'] = pred
            signal = np.where(pred[-1] > 0.5, 1, -1)
            if self.position in [0, -1] and signal == 1:
                # Go long
                print(80*'=')
                print(f'Time: {datetime.datetime.now()} | Signal = {signal}  | GOING LONG')
                print(80*'=')
                order = self.create_order(
                    self.stream_instrument,
                    units=(1 - self.position) * self.units,
                    suppress=self.suppress, ret=True
                )
                q(datetime.datetime.now())
                q(signal)
                q(self.data.iloc[-(lags+1):-1])
                q(order)
                self.position = 1
            elif self.position in [0, 1] and signal == -1:
                # Go short
                print(80*'=')
                print(f'Time: {datetime.datetime.now()} | Signal = {signal} | GOING SHORT')
                print(80*'=')
                order = self.create_order(
                    self.stream_instrument,
                    units=-(1 + self.position) * self.units,
                    suppress=self.suppress, ret=True
                )
                q(datetime.datetime.now())
                q(signal)
                q(self.data.iloc[-(lags+1):-1])
                q(order)
                self.position = -1
    
    def _data_preprocessing(self):
        self._add_features()
        self._normalize()
        self._add_lags()
        self.X_live = self.data_[self.selected_features].iloc[[-1]].copy()

    def _add_features(self):
        # Use reduced dattaframe for features calcs for effeciency
        df = self.data.iloc[-self.min_bars:].copy()
        # Trend
        df['SMA10'] = df['close'].rolling(10).mean()
        df['SMA30'] = df['close'].rolling(30).mean()
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'],
                              timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'],
                                      timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'],
                                        timeperiod=14)
        
        # Momentum
        df['RSI_9'] = talib.RSI(df['close'], timeperiod=9)
        macd, macdsignal, macdhist = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['MACD_hist'] = macdhist

        # Volatility
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'],
                              timeperiod=14)
        df['VOL10'] = df['r'].rolling(10).mean()
        df['VOL30'] = df['r'].rolling(30).mean()
        
        # Slope
        df['SMA10_SLOPE'] = np.arctan((df['SMA10'].shift(1)
                                         - df['SMA10'].shift(5)) / 4)
        df['SMA30_SLOPE'] = np.arctan((df['SMA30'].shift(1)
                                         - df['SMA30'].shift(5)) / 4)
        df['VOL10_SLOPE'] = np.arctan((df['VOL10'].shift(1)
                                         - df['VOL10'].shift(5)) / 4)
        df['VOL30_SLOPE'] = np.arctan((df['VOL30'].shift(1)
                                         - df['VOL30'].shift(5)) / 4)
        
        # Price - Action
        df['CO'] = df['close'] / df['open'] - 1
        df['OHLC4'] = (df['open'] + df['high']
                         + df['low'] + df['close']) / 4
        
        # Calendar - cyclical (sin/cos) time of day
        tod = df.index.hour + df.index.minute / 60
        df['TOD_SIN'] = np.sin(2 * np.pi * tod  /24)
        df['TOD_COS'] = np.cos(2 * np.pi * tod  /24)

        # List of features
        exclude = ['open', 'high', 'low', 'close', 'volume', 'd']
        self.base_features = [x for x in df.columns if x not in exclude]
        # Calendar features should not be lagged
        self.no_lag_features = ['TOD_SIN', 'TOD_COS']
        
        # Dataframe for normalized data
        self.data_ = df.copy()

    def _normalize(self):
        self.data_ = pd.DataFrame(
            self.scaler.transform(self.data_[self.base_features]),
            index=self.data_.index,
            columns=self.base_features
        )

    def _add_lags(self):
        self.lag_features = []       
        lagged_dfs = []
        for feat in self.base_features:
            for lag in range(1, lags + 1):
                col_ = f'{feat}_lag{lag}'
                if feat not in self.no_lag_features:
                    self.lag_features.append(col_)
                    lagged_col = self.data_[feat].shift(lag).rename(col_)
                    lagged_dfs.append(lagged_col)
                elif lag == 1:   # celendar features are only lagges once
                    self.lag_features.append(col_)
                    lagged_col = self.data_[feat].shift(lag).rename(col_)
                    lagged_dfs.append(lagged_col)
                    #self.data_[col_] = self.data_[feat].shift(lag)
        lagged_df = pd.concat(lagged_dfs, axis=1)
        self.data_ = pd.concat([self.data_, lagged_df], axis=1)
        self.data_.dropna(inplace=True)
                    