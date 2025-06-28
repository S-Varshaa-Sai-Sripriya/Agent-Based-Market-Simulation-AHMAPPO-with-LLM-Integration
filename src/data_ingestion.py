import logging
import yfinance as yf
import pandas as pd
import os
import random
import numpy as np
import torch
from datetime import datetime
from exceptions import DataIngestionError
from constants import RAW_DATA_PATH, SEED

logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self):
        self.set_seed(SEED)
        
    def set_seed(self, seed: int = SEED):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}.")

    def fetch_historical_data(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Fetch historical stock data using yfinance.
        
        Args:
            ticker: Stock symbol to fetch data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            pd.DataFrame: Historical stock data
            
        Raises:
            DataIngestionError: If data fetching fails or returns empty
        """
        logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date} with interval '{interval}'")
        
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            # First check if data is None
            if stock_data is None:
                logger.error(f"yfinance returned None for {ticker}. Check internet connection or API status.")
                raise DataIngestionError(f"Null data returned for {ticker}")
                
            # Then check if DataFrame is empty
            if stock_data.empty:
                logger.error(f"Empty DataFrame for {ticker}. Check date range and ticker symbol.")
                raise DataIngestionError(f"Empty data for {ticker}")
                
            stock_data.reset_index(inplace=True)
            logger.info(f"Successfully fetched {len(stock_data)} rows for {ticker}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            raise DataIngestionError(f"Data fetch failed: {str(e)}")
    
    def save_raw_data(self, data: pd.DataFrame, path: str = RAW_DATA_PATH):
        """Save raw data to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        logger.info(f"Raw historical data saved at '{path}'.")

    def run(self, ticker: str, start_date: str, end_date: str):
        """Execute the data ingestion pipeline."""
        historical_data = self.fetch_historical_data(ticker, start_date, end_date)
        self.save_raw_data(historical_data)
        return historical_data