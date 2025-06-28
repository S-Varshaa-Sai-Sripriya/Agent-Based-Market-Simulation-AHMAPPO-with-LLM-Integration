import pandas as pd
import logging
from exceptions import DataTransformationError
from constants import RAW_DATA_PATH, CLEANED_DATA_PATH, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the historical stock data."""
        logger.info("Starting data cleaning process.")
        
        try:
            # Convert date column to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
                logger.info("Converted 'Date' column to datetime.")
            
            # Sort data by date in ascending order
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info("Sorted data by date in ascending order.")
            
            # Remove duplicate entries
            initial_len = len(df)
            df.drop_duplicates(subset=['Date'], inplace=True)
            final_len = len(df)
            logger.info(f"Removed {initial_len - final_len} duplicate rows based on 'Date' column.")
            
            # Ensure all necessary columns are present
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise DataTransformationError(f"Missing required columns: {missing_columns}")
            
            # Reset index after cleaning
            df.reset_index(drop=True, inplace=True)
            logger.info("Data cleaning completed successfully.")
            
            return df
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise DataTransformationError(f"Data cleaning failed: {e}")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the stock data."""
        logger.info("Starting missing values handling.")
        
        try:
            # Check for missing values
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            if total_missing == 0:
                logger.info("No missing values found in the data.")
                return df
            
            logger.warning(f"Found {total_missing} missing values in the data.")
            
            # Forward fill to handle missing values
            df.ffill(inplace=True)
            df.bfill(inplace=True)  # In case forward fill doesn't fill initial missing values
            
            # Verify if all missing values are handled
            if df.isnull().sum().sum() == 0:
                logger.info("All missing values have been handled successfully.")
            else:
                logger.error("There are still missing values after handling.")
                raise DataTransformationError("Missing values remain in the data.")
            
            return df
        except Exception as e:
            logger.error(f"Handling missing values failed: {e}")
            raise DataTransformationError(f"Handling missing values failed: {e}")

    def run(self, raw_data_path: str = RAW_DATA_PATH):
        """Execute the data transformation pipeline."""
        try:
            # Load and clean data
            raw_data = pd.read_csv(raw_data_path)
            cleaned_data = self.clean_data(raw_data)
            cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)
            logger.info(f"Cleaned data saved at '{CLEANED_DATA_PATH}'.")
            
            # Handle missing values
            cleaned_data = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['Date'])
            processed_data = self.handle_missing_values(cleaned_data)
            processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
            logger.info(f"Processed data with missing values handled saved at '{PROCESSED_DATA_PATH}'.")
            
            return processed_data
        except Exception as e:
            logger.error(f"Data transformation pipeline failed: {e}")
            raise DataTransformationError(f"Data transformation pipeline failed: {e}")