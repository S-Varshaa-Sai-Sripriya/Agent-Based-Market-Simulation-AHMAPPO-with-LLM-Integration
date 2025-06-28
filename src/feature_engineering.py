import pandas as pd
import logging
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from exceptions import FeatureEngineeringError
from constants import PROCESSED_DATA_PATH, FEATURES_DATA_PATH, COMBINED_DATA_PATH

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the stock data."""
        logger.info("Adding technical indicators to the data.")
        try:
            # Ensure required columns are present
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise FeatureEngineeringError("Missing required columns for technical indicators")
            
            # Add all technical analysis features
            df = add_all_ta_features(
                df, 
                open="Open", 
                high="High", 
                low="Low", 
                close="Close", 
                volume="Volume"
            )
            
            # Select specific indicators we want to keep
            selected_indicators = [
                'Close', '20_MA', '50_MA', '200_MA', 
                'Volatility', 'RSI', 'Momentum'
            ]
            
            # Ensure all selected indicators exist
            available_indicators = [col for col in selected_indicators if col in df.columns]
            missing_indicators = set(selected_indicators) - set(available_indicators)
            
            if missing_indicators:
                logger.warning(f"Missing some technical indicators: {missing_indicators}")
            
            logger.info("Technical indicators added successfully.")
            return df[['Date'] + available_indicators]
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise FeatureEngineeringError(f"Error adding technical indicators: {e}")

    def combine_with_sentiment(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Combine stock data with sentiment data."""
        logger.info("Combining stock data with sentiment data.")
        try:
            # Aggregate sentiment scores per date
            sentiment_mapping = {'Positive': 1, 'Negative': 0}
            sentiment_data['Sentiment_Label'] = sentiment_data['Sentiment'].map(sentiment_mapping)
            
            # Handle any unmapped sentiments
            if sentiment_data['Sentiment_Label'].isnull().any():
                logger.warning("Found unmapped sentiment labels. Filling with mode.")
                mode_sentiment = sentiment_data['Sentiment_Label'].mode()[0]
                sentiment_data['Sentiment_Label'].fillna(mode_sentiment, inplace=True)
            
            # Group by Date and calculate average sentiment
            df_agg_sentiment = sentiment_data.groupby('Date').agg({'Sentiment_Label': 'mean'}).reset_index()
            df_agg_sentiment.rename(columns={'Sentiment_Label': 'Avg_Sentiment'}, inplace=True)
            
            # Merge with stock data
            df_merged = pd.merge(stock_data, df_agg_sentiment, on='Date', how='left')
            
            # Handle missing sentiment values
            missing_sentiment = df_merged['Avg_Sentiment'].isnull().sum()
            if missing_sentiment > 0:
                logger.warning(f"Found {missing_sentiment} dates with no sentiment data. Filling with 0.5 (neutral).")
                df_merged['Avg_Sentiment'].fillna(0.5, inplace=True)  # Neutral sentiment
            
            logger.info("Successfully combined stock data with sentiment data.")
            return df_merged
        except Exception as e:
            logger.error(f"Error combining with sentiment data: {e}")
            raise FeatureEngineeringError(f"Error combining with sentiment data: {e}")

    def run(self):
        """Execute the feature engineering pipeline."""
        try:
            # Load processed data
            processed_data = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['Date'])
            
            # Add technical indicators
            features_data = self.add_technical_indicators(processed_data)
            features_data.to_csv(FEATURES_DATA_PATH, index=False)
            logger.info(f"Features data saved at '{FEATURES_DATA_PATH}'.")
            
            # Generate sample sentiment data (in a real scenario, this would come from actual sentiment analysis)
            sentiment_data = self._generate_sample_sentiment_data()
            
            # Combine with sentiment
            combined_data = self.combine_with_sentiment(features_data, sentiment_data)
            combined_data.to_csv(COMBINED_DATA_PATH, index=False)
            logger.info(f"Combined data saved at '{COMBINED_DATA_PATH}'.")
            
            return combined_data
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            raise FeatureEngineeringError(f"Feature engineering pipeline failed: {e}")

    def _generate_sample_sentiment_data(self) -> pd.DataFrame:
        """Generate sample sentiment data for demonstration purposes."""
        # This is the same sample data generation code from the original notebook
        # In a real application, this would come from actual sentiment analysis
        import random
        from datetime import datetime, timedelta
        
        sample_headlines = ["Apple releases new product", "Market reacts to news", ...]  # truncated for brevity
        sample_sources = ["CNN", "BBC", "Reuters", "Bloomberg", "Fox News"]
        
        data = []
        for i in range(100):
            headline = random.choice(sample_headlines)
            source = random.choice(sample_sources)
            random_days = random.randint(0, 180)
            date = datetime.now() - timedelta(days=random_days)
            date_str = date.strftime('%Y-%m-%d')
            sentiment = random.choice(['Positive', 'Negative'])
            data.append({
                'Date': date_str,
                'Headline': headline,
                'Source': source,
                'Sentiment': sentiment
            })
        
        return pd.DataFrame(data)