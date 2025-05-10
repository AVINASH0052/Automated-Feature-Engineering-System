import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FeatureSelector:
    """
    A class to select the most important features using Random Forest.
    """
    
    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features by encoding categorical variables and scaling numerical ones.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        processed_df = df.copy()
        
        for column in processed_df.columns:
            if column == self.target_col:
                continue
                
            # Handle categorical variables
            if processed_df[column].dtype == 'object':
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
        
        return processed_df
        
    def select_features(self, df: pd.DataFrame, test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Select the most important features using Random Forest.
        
        Args:
            df (pd.DataFrame): Input dataframe with target column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: Selected features and their importance scores
        """
        # Preprocess data
        processed_df = self._preprocess_features(df)
        
        # Prepare data
        X = processed_df.drop(columns=[self.target_col])
        y = processed_df[self.target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance))
        
        # Select top features (importance > 0.01)
        selected_features = [f for f, imp in self.feature_importance.items() if imp > 0.01]
        
        return df[selected_features + [self.target_col]], self.feature_importance 