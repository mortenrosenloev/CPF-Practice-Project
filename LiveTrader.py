# CPF Algorithmic Trading Practice Project
# Full name: Morten Rosenløv Jensen
# Submission data: 23-04-2025

import os
import joblib
from KNNTrader import KNNTrader

def run_live_trading():    
# Input
    path = 'data'  # path to folder containing OANDA credentials-file
    file = 'oanda.cfg' # name of OANDA credentials-file
    symbol = 'EUR_USD'  # trading instrument
    units = 10000  # number of units to be traded
    
    # Load model and scaler
    bundle = joblib.load('knn_bundle.joblib')
    scaler = bundle['scaler']
    model = bundle['model']
    selected_features = bundle['features']
    
    # Instantiate the MLPTrader
    config_file = os.path.join(path, file)
    trader = KNNTrader(config_file, model, scaler, selected_features,
                       units, duration_minutes=1440, log_minutes=15, verbose=False)
    
    # Execute live testing
    trader.stream_data(symbol)

if __name__ == '__main__':
    run_live_trading()
    