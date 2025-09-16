#!/usr/bin/env python3
"""
Demo script to showcase the visualization capabilities of the Genetic Algorithm
This script creates sample data and generates all visualization plots
"""

import numpy as np
import pandas as pd
import visualization

def create_sample_data():
    """Create sample genetic algorithm results for demonstration"""
    print("Creating sample genetic algorithm results...")
    
    # Sample hyperparameter ranges
    layers_list = [1, 2, 3, 4]
    batch_list = [10, 25, 50, 100, 200]
    optimizers = ['Adam', 'Adagrad', 'RMSprop', 'sgd']
    activations = ['relu', 'tanh', 'sigmoid', 'elu']
    epochs_list = [50, 100, 150, 200]
    dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    training_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Generate sample data for 4 generations with 40 solutions each
    sample_data = []
    
    # Add header
    sample_data.append(['Layers', 'Neurons', 'batch', 'optimiser', 'keras', 'epochs', 'dropout', 'train %', 'activation', 'RMSE', 'VAL_RMSE', 'Objective', 'mae', 'val_mae', 'R2', 'R2_v'])
    
    for generation in range(4):
        for solution in range(40):
            # Generate random hyperparameters
            layers = np.random.choice(layers_list)
            batch = np.random.choice(batch_list)
            optimizer = np.random.choice(optimizers)
            kernel_init = np.random.choice(['uniform', 'normal'])
            epochs = np.random.choice(epochs_list)
            dropout = np.random.choice(dropout_list)
            train_percent = np.random.choice(training_list)
            activation = np.random.choice(activations)
            
            # Generate neurons based on layers
            if layers == 1:
                neurons = np.random.randint(1, 21)
            elif layers == 2:
                neurons = (np.random.randint(1, 21), np.random.randint(1, 21))
            elif layers == 3:
                neurons = (np.random.randint(1, 21), np.random.randint(1, 21), np.random.randint(1, 21))
            else:
                neurons = (np.random.randint(1, 21), np.random.randint(1, 21), np.random.randint(1, 21), np.random.randint(1, 21))
            
            # Generate performance metrics with some improvement over generations
            base_rmse = 0.3 + np.random.normal(0, 0.1)
            base_val_rmse = 0.35 + np.random.normal(0, 0.1)
            
            # Add generation-based improvement
            improvement_factor = 1 - (generation * 0.1)  # 10% improvement per generation
            rmse = max(0.01, base_rmse * improvement_factor)
            val_rmse = max(0.01, base_val_rmse * improvement_factor)
            
            # Calculate objective function
            objective = ((1 - rmse) * 0.5 + (1 - val_rmse) * 0.5)
            
            # Generate other metrics
            mae = rmse * 0.8 + np.random.normal(0, 0.02)
            val_mae = val_rmse * 0.8 + np.random.normal(0, 0.02)
            r2 = max(0, 1 - rmse + np.random.normal(0, 0.05))
            r2_val = max(0, 1 - val_rmse + np.random.normal(0, 0.05))
            
            sample_data.append([
                layers, neurons, batch, optimizer, kernel_init, epochs, 
                dropout, train_percent, activation, rmse, val_rmse, 
                objective, mae, val_mae, r2, r2_val
            ])
    
    return sample_data

def main():
    """Main demonstration function"""
    print("="*80)
    print("GENETIC ALGORITHM VISUALIZATION DEMO")
    print("="*80)
    print("This demo creates sample genetic algorithm results and generates")
    print("comprehensive visualizations to showcase the plotting capabilities.")
    print("="*80)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Create visualizer
    print("\nInitializing visualizer with sample data...")
    visualizer = visualization.GeneticAlgorithmVisualizer(sample_data)
    
    # Generate all visualizations
    print("\nGenerating comprehensive visualization suite...")
    visualizer.create_all_visualizations()
    
    # Also generate real-time progress plot with sample data
    print("\nGenerating real-time progress visualization...")
    generation_tracker = {
        'generations': [1, 2, 3, 4],
        'best_objectives': [0.65, 0.72, 0.78, 0.82],
        'avg_objectives': [0.58, 0.64, 0.69, 0.73],
        'worst_objectives': [0.45, 0.52, 0.58, 0.61]
    }
    visualization.plot_realtime_progress(generation_tracker)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Generated visualization files:")
    print("📈 fitness_evolution.png - Shows how fitness metrics evolve across generations")
    print("🔍 hyperparameter_analysis.png - Analyzes impact of different hyperparameters")
    print("🏆 best_solutions_analysis.png - Detailed analysis of top 10 solutions")
    print("📉 convergence_analysis.png - Shows GA convergence behavior")
    print("🔥 correlation_heatmap.png - Correlation matrix of all variables")
    print("📋 genetic_algorithm_summary.txt - Comprehensive text summary")
    print("⚡ realtime_progress.png - Real-time progress during execution")
    print("="*80)
    print("\nTo run the actual genetic algorithm with your data:")
    print("python main_NeuroGenetic.py")
    print("="*80)

if __name__ == "__main__":
    main()

