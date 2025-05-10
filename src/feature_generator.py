import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from datetime import datetime

class FeatureGenerator:
    """
    A class to generate new features based on detected feature types.
    """
    
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        
    def generate_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Generate new numerical features using polynomial combinations and statistical measures.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (List[str]): List of numerical column names
            
        Returns:
            pd.DataFrame: Dataframe with new numerical features
        """
        new_df = df.copy()
        
        # Generate polynomial features
        if len(numerical_cols) > 1:
            poly_features = self.poly.fit_transform(df[numerical_cols])
            poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
            new_df[poly_cols] = poly_features
            
        # Generate statistical features
        for col in numerical_cols:
            new_df[f'{col}_squared'] = df[col] ** 2
            new_df[f'{col}_cubed'] = df[col] ** 3
            new_df[f'{col}_log'] = np.log1p(df[col])
            
        return new_df
    
    def generate_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Generate new categorical features using encoding and combinations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_cols (List[str]): List of categorical column names
            
        Returns:
            pd.DataFrame: Dataframe with new categorical features
        """
        new_df = df.copy()
        
        # One-hot encoding for categorical features
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            new_df = pd.concat([new_df, dummies], axis=1)
            
        # Create interaction features between categorical columns
        if len(categorical_cols) > 1:
            for i in range(len(categorical_cols)):
                for j in range(i+1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    new_df[f'{col1}_{col2}_interaction'] = df[col1] + '_' + df[col2]
                    
        return new_df
    
    def generate_datetime_features(self, df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """
        Generate new features from datetime columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_cols (List[str]): List of datetime column names
            
        Returns:
            pd.DataFrame: Dataframe with new datetime features
        """
        new_df = df.copy()
        
        for col in datetime_cols:
            new_df[f'{col}_year'] = df[col].dt.year
            new_df[f'{col}_month'] = df[col].dt.month
            new_df[f'{col}_day'] = df[col].dt.day
            new_df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            new_df[f'{col}_quarter'] = df[col].dt.quarter
            
        return new_df
    
    def generate_all_features(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Generate all possible features based on feature groups.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_groups (Dict[str, List[str]]): Dictionary of feature groups
            
        Returns:
            pd.DataFrame: Dataframe with all new features
        """
        new_df = df.copy()
        
        # Generate features for each type
        if feature_groups['numerical']:
            new_df = self.generate_numerical_features(new_df, feature_groups['numerical'])
            
        if feature_groups['categorical']:
            new_df = self.generate_categorical_features(new_df, feature_groups['categorical'])
            
        if feature_groups['datetime']:
            new_df = self.generate_datetime_features(new_df, feature_groups['datetime'])
            
        return new_df 