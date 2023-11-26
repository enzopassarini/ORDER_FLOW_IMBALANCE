import MetaTrader5 as mt5
import pandas as pd

def mt5_connect(login, password, server):
    """
    Connects to MetaTrader 5 and displays basic account information.
    """

    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize() failed")
        mt5.shutdown()
    else: 
        print("Conection stablished with MetaTrader5")
        print(mt5.account_info())

def OpenTicker(ticker):
    """"
    Opens ticker in MT5, making it open for data requests.
    """
    symbol_info = mt5.symbol_info(ticker)
    if not symbol_info.visible:
        mt5.symbol_select(ticker,True)

def open_book(ticker):
    """
    Makes ticker available for book data extraction.
    """
    mt5.market_book_add(ticker)

def extract_bid_ask(ticker):
    """"
    Returns book data for the specified ticker.
    """
    book = mt5.market_book_get(ticker)
    return book

def setup_ticker(ticker):
    """
    Opens ticker for both book and other information pull.
    """
    OpenTicker(ticker)
    open_book(ticker)
