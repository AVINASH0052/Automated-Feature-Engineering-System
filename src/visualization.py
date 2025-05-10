import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    """
    A class to create visualizations for feature engineering results.
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')  # Use the updated seaborn style name
        
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 20, save_path: str = None):
        """
        Plot feature importance scores.
        
        Args:
            feature_importance (Dict[str, float]): Dictionary of feature importance scores
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(importance), y=list(features))
        plt.title('Top Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        
    def plot_feature_correlation(self, df: pd.DataFrame, target_col: str = 'target',
                               save_path: str = None):
        """
        Plot correlation matrix of features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            save_path (str): Path to save the plot
        """
        # Convert categorical columns to numeric
        df_numeric = df.copy()
        for column in df_numeric.columns:
            if df_numeric[column].dtype == 'object':
                df_numeric[column] = pd.Categorical(df_numeric[column]).codes
                
        # Calculate correlation matrix
        corr_matrix = df_numeric.corr()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 feature_groups: Dict[str, List[str]],
                                 save_path: str = None,
                                 max_categories: int = 10):
        """
        Plot distributions of features by type.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_groups (Dict[str, List[str]]): Dictionary of feature groups
            save_path (str): Path to save the plot
            max_categories (int): Maximum number of categories to show in categorical plots
        """
        for ftype, features in feature_groups.items():
            if not features:
                continue
                
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5*n_rows))
            
            for i, feature in enumerate(features, 1):
                plt.subplot(n_rows, n_cols, i)
                
                if ftype in ['numerical', 'categorical_numerical']:
                    sns.histplot(data=df, x=feature, kde=True)
                elif ftype == 'categorical':
                    # Get value counts and limit categories if needed
                    value_counts = df[feature].value_counts()
                    if len(value_counts) > max_categories:
                        top_categories = value_counts.nlargest(max_categories).index
                        data = df[df[feature].isin(top_categories)]
                    else:
                        data = df
                    sns.countplot(data=data, x=feature)
                elif ftype == 'datetime':
                    sns.histplot(data=df, x=feature, bins=30)
                    
                plt.title(f'{feature} Distribution')
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f'{save_path}_{ftype}.png')
            plt.close()  # Close the figure to free memory
            
    def plot_interactive_feature_importance(self, feature_importance: Dict[str, float],
                                          top_n: int = 20):
        """
        Create an interactive plot of feature importance using Plotly.
        
        Args:
            feature_importance (Dict[str, float]): Dictionary of feature importance scores
            top_n (int): Number of top features to display
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        # Create interactive plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(importance),
                y=list(features),
                orientation='h',
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title='Interactive Feature Importance Plot',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            width=800
        )
        
        fig.show() 