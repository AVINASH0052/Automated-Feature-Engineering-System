import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class FeatureDetector:
    """
    A class to detect and categorize features in a dataset.
    """
    
    def __init__(self):
        self.feature_types = {}
        self.numerical_threshold = 10  # Threshold for categorical vs numerical
        
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect the type of each feature in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, str]: Dictionary mapping feature names to their types
        """
        for column in df.columns:
            # Skip target variable if specified
            if column == 'target':
                continue
                
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(df[column]):
                unique_values = df[column].nunique()
                
                if unique_values <= self.numerical_threshold:
                    self.feature_types[column] = 'categorical_numerical'
                else:
                    self.feature_types[column] = 'numerical'
                    
            # Check if column is categorical
            elif pd.api.types.is_object_dtype(df[column]):
                self.feature_types[column] = 'categorical'
                
            # Check if column is datetime
            elif pd.api.types.is_datetime64_dtype(df[column]):
                self.feature_types[column] = 'datetime'
                
            else:
                self.feature_types[column] = 'unknown'
                
        return self.feature_types
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Group features by their detected types.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping feature types to lists of feature names
        """
        feature_groups = {
            'numerical': [],
            'categorical': [],
            'categorical_numerical': [],
            'datetime': [],
            'unknown': []
        }
        
        for feature, ftype in self.feature_types.items():
            feature_groups[ftype].append(feature)
            
        return feature_groups 