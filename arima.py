import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import threading

# Function to download stock price data
def download_stock_price(symbol):
    # Set start and end dates for historical data
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365) # 1 years data

    # Get historical data for chosen stock from Yahoo Finance using yfinance
    data_frame = yf.download(symbol, start=start, end=end)
    return data_frame

# Function to predict stock prices using ARIMA model
def predict_stock_price(data):
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima

    # Split the data into train and test sets
    x_train = data[:-60]   # use first n-60 days for training
    x_test = data[-60:]    # use last 60 days for testing

    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True)
    model = ARIMA(data, order=stepwise_fit.order)
    model = model.fit()

    start=len(x_train)
    end=len(x_train)+len(x_test)-1
    pred = model.predict(start=start,end=end)

    return pred

# Function to plot the actual and predicted stock prices
def plot_stock_prices(symbol, data_frame, pred_future):
    import datetime
    start_date = datetime.datetime.today()
    dates = [start_date + datetime.timedelta(days=idx) for idx in range(11)]

    pred_future2 = pd.Series(pred_future, index = dates)

    plt.figure(figsize=(10,6), dpi=100)
    data_frame['Close'][-250:].plot(label='Actual Stock Price', legend=True)
    pred_future2.plot(label='Future Predicted Price', legend=True)
    plt.title(symbol)
    plt.legend()
    plt.grid()
    plt.show()

# Function to start the socket server
def start_socket_server(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen()
    print(f"Socket server started on {host}:{port}")
    return s

# Function to handle client connections
def handle_client(conn, addr):
    print(f"New client connected: {addr}")
    symbol = conn.recv(1024).decode()
    print(f"Received symbol: {symbol}")
    data_frame = download_stock_price(symbol)
    data = list(data_frame["Close"])
    pred_future = predict_stock_price(data)
    plot_stock_prices(symbol, data_frame, pred_future)
    conn.sendall(b"Done")
    conn.close()
    print(f"Client disconnected: {addr}")

if __name__ == "__main__":
    # Start the socket server
    s = start_socket_server("localhost", 5052)

    while True:
        # Accept client connections
        conn, addr = s.accept()
        # Handle client connections in a new thread
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
