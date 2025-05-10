import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_detector import FeatureDetector
from src.feature_generator import FeatureGenerator
from src.feature_selector import FeatureSelector
from src.visualization import Visualizer

def load_adult_dataset():
    """
    Load and preprocess the Adult Income dataset.
    """
    # Column names for the Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    
    # Load data
    df = pd.read_csv('data/adult.csv', names=columns)
    
    # Clean data
    df = df.replace(' ?', np.nan)  # Replace ? with NaN
    df = df.dropna()  # Remove rows with missing values
    
    # Convert target to binary
    df['target'] = (df['income'] == ' >50K').astype(int)
    df = df.drop('income', axis=1)
    
    return df

class AutomatedFeatureEngineering:
    """
    Main class to orchestrate the automated feature engineering pipeline.
    """
    
    def __init__(self, target_col: str = 'target'):
        self.target_col = target_col
        self.detector = FeatureDetector()
        self.generator = FeatureGenerator()
        self.selector = FeatureSelector(target_col)
        self.visualizer = Visualizer()
        
    def process_data(self, df: pd.DataFrame, 
                    save_plots: bool = True,
                    plot_dir: str = 'plots') -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Process the data through the entire feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            save_plots (bool): Whether to save plots
            plot_dir (str): Directory to save plots
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: Processed dataframe and feature importance
        """
        # Create plot directory if it doesn't exist
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Step 1: Detect feature types
        print("Step 1: Detecting feature types...")
        feature_types = self.detector.detect_feature_types(df)
        feature_groups = self.detector.get_feature_groups()
        
        print("\nFeature groups detected:")
        for ftype, features in feature_groups.items():
            if features:
                print(f"\n{ftype.upper()}:")
                print(", ".join(features))
        
        # Step 2: Generate new features
        print("\nStep 2: Generating new features...")
        df_with_new_features = self.generator.generate_all_features(df, feature_groups)
        print(f"Original features: {df.shape[1]}")
        print(f"After feature generation: {df_with_new_features.shape[1]}")
        
        # Step 3: Select important features
        print("\nStep 3: Selecting important features...")
        selected_df, feature_importance = self.selector.select_features(df_with_new_features)
        print(f"Features after selection: {selected_df.shape[1] - 1}")  # Subtract 1 for target
        
        # Update feature groups for selected features
        selected_feature_groups = {
            'numerical': [],
            'categorical': [],
            'categorical_numerical': [],
            'datetime': [],
            'unknown': []
        }
        
        for feature in selected_df.columns:
            if feature == self.target_col:
                continue
            for ftype, features in feature_groups.items():
                if feature in features:
                    selected_feature_groups[ftype].append(feature)
                    break
        
        # Step 4: Create visualizations
        print("\nStep 4: Creating visualizations...")
        if save_plots:
            self.visualizer.plot_feature_importance(
                feature_importance,
                save_path=os.path.join(plot_dir, 'feature_importance.png')
            )
            
            self.visualizer.plot_feature_correlation(
                selected_df,
                save_path=os.path.join(plot_dir, 'correlation_matrix.png')
            )
            
            self.visualizer.plot_feature_distributions(
                selected_df,
                selected_feature_groups,
                save_path=os.path.join(plot_dir, 'feature_distributions')
            )
            
            # Create interactive plot
            self.visualizer.plot_interactive_feature_importance(feature_importance)
            
        return selected_df, feature_importance
    
    def save_results(self, df: pd.DataFrame, feature_importance: Dict[str, float],
                    output_dir: str = 'output'):
        """
        Save the processed data and feature importance.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            feature_importance (Dict[str, float]): Feature importance scores
            output_dir (str): Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save processed data
        df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        })
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        print(f"\nResults saved to {output_dir}/")

def main():
    """
    Main function to run the automated feature engineering pipeline.
    """
    print("Loading Adult Income dataset...")
    df = load_adult_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Initialize the pipeline
    pipeline = AutomatedFeatureEngineering(target_col='target')
    
    # Process the data
    processed_df, feature_importance = pipeline.process_data(df)
    
    # Save results
    pipeline.save_results(processed_df, feature_importance)
    
    print("\nFeature engineering pipeline completed successfully!")

if __name__ == "__main__":
    main() 