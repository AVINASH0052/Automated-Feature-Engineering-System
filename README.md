# Automated Feature Engineering System

This project implements an automated feature engineering system that can detect, generate, and select important features from any dataset. The system uses machine learning techniques to identify the most relevant features and create new ones to improve model performance.

## Features

- **Automatic Feature Detection**: Identifies numerical, categorical, and datetime features
- **Feature Generation**: Creates new features using various transformations and combinations
- **Neural Network-based Feature Selection**: Uses a deep learning model to select the most important features
- **Interactive Visualizations**: Provides detailed visualizations of feature importance and relationships

## Project Structure

```
.
├── src/
│   ├── feature_detector.py    # Feature type detection
│   ├── feature_generator.py   # Feature generation
│   ├── feature_selector.py    # Feature selection
│   ├── visualization.py       # Visualization tools
│   └── main.py               # Main pipeline
├── data/                     # Data directory
├── notebooks/               # Jupyter notebooks
├── plots/                   # Generated plots
├── output/                  # Output files
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd automated-feature-engineering
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `data/` directory.

2. Update the target column name in `main.py` if needed:
```python
pipeline = AutomatedFeatureEngineering(target_col='your_target_column')
```

3. Run the pipeline:
```bash
python src/main.py
```

4. Check the results in:
- `plots/` directory for visualizations
- `output/` directory for processed data and feature importance scores

## Components

### Feature Detector
- Automatically identifies feature types
- Supports numerical, categorical, and datetime features
- Groups features by type for processing

### Feature Generator
- Creates polynomial features for numerical data
- Generates interaction terms
- Handles categorical encoding
- Extracts datetime components

### Feature Selector
- Uses a neural network to evaluate feature importance
- Selects features based on importance scores
- Provides feature rankings

### Visualizer
- Creates feature importance plots
- Generates correlation matrices
- Shows feature distributions
- Provides interactive Plotly visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{automated-feature-engineering,
  author = {Your Name},
  title = {Automated Feature Engineering System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/automated-feature-engineering}
}
``` 