# NeuroGenetic: Neural Network Architecture Optimization using Genetic Algorithms

This project implements a genetic algorithm to optimize neural network architectures for manufacturing data analysis. It combines the power of artificial neural networks (ANN) with genetic algorithms to find optimal network architectures and hyperparameters.

## Features

- Genetic Algorithm for Neural Network Architecture Optimization
- Support for multiple layer configurations (1-4 layers)
- Various activation functions (ReLU, tanh, sigmoid, elu)
- Multiple optimizer options (Adam, Adagrad, RMSprop, SGD)
- Automatic hyperparameter tuning
- Performance metrics tracking (RMSE, MAE, R²)
- **Comprehensive Visualization Suite** with 7 different plot types
- Real-time progress tracking during execution
- Detailed analysis of best solutions and convergence behavior

## Project Structure

```
NeuroGenetic/
├── main_NeuroGenetic.py    # Main genetic algorithm implementation
├── genetic.py             # Genetic algorithm operators
├── ann.py                 # Neural network implementation
├── data_input.py          # Data preprocessing
├── visualization.py       # Comprehensive visualization suite
├── demo_visualization.py  # Demo script for visualization features
└── README.md              # Project documentation
```

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone ..........
cd NeuroGenetic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Genetic Algorithm

1. Prepare your data in CSV format (see `data_manufacturing.csv` for example)
2. Run the genetic algorithm:
```bash
python main_NeuroGenetic.py
```

### Visualization Demo

To see the visualization capabilities with sample data:
```bash
python demo_visualization.py
```

### Generated Visualizations

The algorithm automatically generates 7 comprehensive visualization files:

1. **`fitness_evolution.png`** - Evolution of fitness metrics across generations
2. **`hyperparameter_analysis.png`** - Hyperparameter impact analysis  
3. **`best_solutions_analysis.png`** - Top 10 solutions detailed analysis
4. **`convergence_analysis.png`** - GA convergence behavior
5. **`correlation_heatmap.png`** - Correlation matrix of all variables
6. **`genetic_algorithm_summary.txt`** - Comprehensive text summary
7. **`realtime_progress.png`** - Real-time progress during execution

## Parameters

- `num_generations`: Number of generations (default: 4)
- `sol_per_pop`: Solutions per population (default: 40)
- `num_parents_mating`: Number of parents for mating (default: 8)
- `mutation_percent`: Mutation percentage (default: 50)

## Results

The algorithm outputs:
- Best neural network architecture
- Performance metrics (RMSE, MAE, R²)
- Training and validation results
- Final results saved to `results.csv`
- **7 comprehensive visualization plots** showing:
  - Evolution of fitness across generations
  - Hyperparameter impact analysis
  - Best solutions comparison
  - Convergence behavior
  - Variable correlations
  - Real-time progress tracking
- Detailed text summary report

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
