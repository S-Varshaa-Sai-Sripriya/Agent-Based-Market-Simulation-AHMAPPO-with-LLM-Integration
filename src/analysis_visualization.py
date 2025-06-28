import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from exceptions import AnalysisError
from constants import COMBINED_DATA_PATH, MODEL_RESULTS_PATH

logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        pass

    def load_data(self):
        """Load necessary data for analysis."""
        try:
            df_combined = pd.read_csv(COMBINED_DATA_PATH, parse_dates=['Date'])
            df_results = pd.read_csv(MODEL_RESULTS_PATH)
            
            # Define target variable if not already present
            if 'Target' not in df_combined.columns:
                df_combined = df_combined.sort_values('Date').reset_index(drop=True)
                df_combined['Next_Close'] = df_combined['Close'].shift(-1)
                df_combined['Target'] = (df_combined['Next_Close'] > df_combined['Close']).astype(int)
                df_combined = df_combined.dropna(subset=['Next_Close'])
            
            return df_combined, df_results
        except Exception as e:
            logger.error(f"Error loading data for analysis: {e}")
            raise AnalysisError(f"Error loading data for analysis: {e}")

    def exploratory_data_analysis(self, df: pd.DataFrame):
        """Perform exploratory data analysis."""
        logger.info("Starting Exploratory Data Analysis.")
        try:
            # Display basic statistics
            logger.info("Descriptive statistics:")
            logger.info(df.describe())
            
            # Target variable distribution
            plt.figure(figsize=(6,4))
            sns.countplot(x='Target', data=df, palette='viridis')
            plt.title('Target Variable Distribution')
            plt.xlabel('Target (0: Down, 1: Up)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('target_distribution.png')
            plt.close()
            logger.info("Target variable distribution plot saved.")
            
            # Numerical features boxplots
            numerical_features = ['Close', '20_MA', '50_MA', '200_MA', 'Volatility', 'RSI', 'Momentum', 'Avg_Sentiment']
            df_melted = df.melt(id_vars=['Target'], value_vars=numerical_features, 
                               var_name='Feature', value_name='Value')
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Feature', y='Value', data=df_melted, palette='Set3')
            plt.title('Boxplots of Numerical Features')
            plt.xlabel('Feature')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('numerical_features_boxplots.png')
            plt.close()
            logger.info("Boxplots of numerical features saved.")
            
            logger.info("Exploratory Data Analysis completed.")
        except Exception as e:
            logger.error(f"Error during EDA: {e}")
            raise AnalysisError(f"Error during EDA: {e}")

    def correlation_analysis(self, df: pd.DataFrame):
        """Perform correlation analysis."""
        logger.info("Starting Correlation Analysis.")
        try:
            # Select relevant features including the target
            features = ['Close', '20_MA', '50_MA', '200_MA', 
                       'Volatility', 'RSI', 'Momentum', 'Avg_Sentiment', 'Target']
            df_corr = df[features]
            
            # Compute correlation matrix
            corr_matrix = df_corr.corr()
            
            # Plot heatmap
            plt.figure(figsize=(10,8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            plt.close()
            logger.info("Correlation matrix heatmap saved.")
            
            # Focus on correlations with the target variable
            target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)
            logger.info("Correlation of features with the target variable:")
            logger.info(target_corr)
            
            # Plot correlations with target
            plt.figure(figsize=(8,6))
            sns.barplot(x=target_corr.values, y=target_corr.index, palette='viridis')
            plt.title('Feature Correlations with Target Variable')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('feature_correlation_with_target.png')
            plt.close()
            logger.info("Feature correlations with target plot saved.")
            
            logger.info("Correlation Analysis completed.")
        except Exception as e:
            logger.error(f"Error during correlation analysis: {e}")
            raise AnalysisError(f"Error during correlation analysis: {e}")

    def model_performance_analysis(self, df_results: pd.DataFrame):
        """Analyze model performance metrics."""
        logger.info("Starting Model Performance Analysis.")
        try:
            # Melt the DataFrame for easier plotting
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            df_melted = df_results.melt(id_vars=['Model'], value_vars=metrics, 
                                       var_name='Metric', value_name='Value')
            
            # Plot metrics comparison
            plt.figure(figsize=(10,6))
            sns.barplot(x='Metric', y='Value', hue='Model', data=df_melted, palette='Set2')
            plt.title('Model Performance Comparison')
            plt.ylim(0, 1)
            plt.ylabel('Score')
            plt.xlabel('Metric')
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig('model_performance_comparison.png')
            plt.close()
            logger.info("Model performance comparison plot saved.")
            
            logger.info("Model Performance Analysis completed.")
        except Exception as e:
            logger.error(f"Error during model performance analysis: {e}")
            raise AnalysisError(f"Error during model performance analysis: {e}")

    def visualize_sentiment_impact(self, df: pd.DataFrame):
        """Visualize the impact of sentiment on stock performance."""
        logger.info("Starting Sentiment Impact Visualization.")
        try:
            # Plot Average Sentiment vs. Target
            plt.figure(figsize=(8,6))
            sns.boxplot(x='Target', y='Avg_Sentiment', data=df, palette='Set3')
            plt.title('Average Sentiment by Stock Performance')
            plt.xlabel('Stock Performance (0: Down, 1: Up)')
            plt.ylabel('Average Sentiment')
            plt.tight_layout()
            plt.savefig('avg_sentiment_by_target.png')
            plt.close()
            logger.info("Average sentiment by target plot saved.")
            
            # Scatter plot of Avg_Sentiment vs. Momentum colored by Target
            plt.figure(figsize=(8,6))
            sns.scatterplot(x='Avg_Sentiment', y='Momentum', hue='Target', 
                           data=df, palette='Set1', alpha=0.7)
            plt.title('Momentum vs. Average Sentiment Colored by Stock Performance')
            plt.xlabel('Average Sentiment')
            plt.ylabel('Momentum')
            plt.legend(title='Target', loc='best')
            plt.tight_layout()
            plt.savefig('momentum_vs_sentiment_scatter.png')
            plt.close()
            logger.info("Momentum vs. Average Sentiment scatter plot saved.")
            
            logger.info("Sentiment Impact Visualization completed.")
        except Exception as e:
            logger.error(f"Error during sentiment impact visualization: {e}")
            raise AnalysisError(f"Error during sentiment impact visualization: {e}")

    def run(self):
        """Execute the analysis pipeline."""
        try:
            # Load data
            df_combined, df_results = self.load_data()
            
            # Perform analyses
            self.exploratory_data_analysis(df_combined)
            self.correlation_analysis(df_combined)
            self.model_performance_analysis(df_results)
            self.visualize_sentiment_impact(df_combined)
            
            logger.info("Analysis pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise AnalysisError(f"Analysis pipeline failed: {e}")