import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_realtime_progress(generation_tracker, save_path='realtime_progress.png'):
    """Plot real-time progress of the genetic algorithm during execution"""
    if not generation_tracker['generations']:
        print("No progress data available for plotting")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot best, average, and worst objectives
    plt.plot(generation_tracker['generations'], generation_tracker['best_objectives'], 
             'o-', linewidth=3, markersize=8, label='Best Objective', color='green')
    plt.plot(generation_tracker['generations'], generation_tracker['avg_objectives'], 
             'o-', linewidth=2, markersize=6, label='Average Objective', color='blue')
    plt.plot(generation_tracker['generations'], generation_tracker['worst_objectives'], 
             'o-', linewidth=2, markersize=6, label='Worst Objective', color='red')
    
    # Fill area between best and worst
    plt.fill_between(generation_tracker['generations'], 
                     generation_tracker['worst_objectives'], 
                     generation_tracker['best_objectives'], 
                     alpha=0.2, color='gray', label='Objective Range')
    
    plt.title('Real-time Genetic Algorithm Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotations
    for i in range(1, len(generation_tracker['generations'])):
        improvement = generation_tracker['best_objectives'][i] - generation_tracker['best_objectives'][i-1]
        if improvement > 0:
            plt.annotate(f'+{improvement:.4f}', 
                        xy=(generation_tracker['generations'][i], generation_tracker['best_objectives'][i]),
                        xytext=(generation_tracker['generations'][i], generation_tracker['best_objectives'][i] + 0.01),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                        fontsize=9, ha='center', color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Real-time progress plot saved as {save_path}")

class GeneticAlgorithmVisualizer:
    def __init__(self, results_data=None):
        """
        Initialize the visualizer with results data
        Args:
            results_data: List of lists containing generation results
        """
        self.results_data = results_data
        self.df = None
        if results_data:
            self._prepare_dataframe()
    
    def _prepare_dataframe(self):
        """Convert results data to pandas DataFrame for easier analysis"""
        if not self.results_data:
            return
        
        # Skip header row
        data = self.results_data[1:] if self.results_data[0][0] == 'Layers' else self.results_data
        
        self.df = pd.DataFrame(data, columns=[
            'Layers', 'Neurons', 'Batch', 'Optimizer', 'Kernel_Init', 
            'Epochs', 'Dropout', 'Train_Percent', 'Activation', 
            'RMSE', 'VAL_RMSE', 'Objective', 'MAE', 'VAL_MAE', 'R2', 'R2_Val'
        ])
        
        # Convert numeric columns
        numeric_cols = ['Layers', 'Batch', 'Epochs', 'Dropout', 'Train_Percent', 
                       'RMSE', 'VAL_RMSE', 'Objective', 'MAE', 'VAL_MAE', 'R2', 'R2_Val']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Add generation column (assuming equal distribution across generations)
        if len(self.df) > 0:
            solutions_per_gen = len(self.df) // 4  # Assuming 4 generations
            self.df['Generation'] = [i // solutions_per_gen + 1 for i in range(len(self.df))]
    
    def plot_fitness_evolution(self, save_path='fitness_evolution.png'):
        """Plot the evolution of fitness metrics across generations"""
        if self.df is None:
            print("No data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fitness Evolution Across Generations', fontsize=16, fontweight='bold')
        
        # RMSE Evolution
        gen_rmse = self.df.groupby('Generation')['RMSE'].agg(['mean', 'min', 'max', 'std'])
        axes[0,0].plot(gen_rmse.index, gen_rmse['mean'], 'o-', linewidth=2, markersize=8, label='Mean RMSE')
        axes[0,0].fill_between(gen_rmse.index, 
                              gen_rmse['mean'] - gen_rmse['std'], 
                              gen_rmse['mean'] + gen_rmse['std'], 
                              alpha=0.3, label='±1 Std Dev')
        axes[0,0].set_title('Training RMSE Evolution')
        axes[0,0].set_xlabel('Generation')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Validation RMSE Evolution
        gen_val_rmse = self.df.groupby('Generation')['VAL_RMSE'].agg(['mean', 'min', 'max', 'std'])
        axes[0,1].plot(gen_val_rmse.index, gen_val_rmse['mean'], 'o-', linewidth=2, markersize=8, label='Mean Val RMSE')
        axes[0,1].fill_between(gen_val_rmse.index, 
                              gen_val_rmse['mean'] - gen_val_rmse['std'], 
                              gen_val_rmse['mean'] + gen_val_rmse['std'], 
                              alpha=0.3, label='±1 Std Dev')
        axes[0,1].set_title('Validation RMSE Evolution')
        axes[0,1].set_xlabel('Generation')
        axes[0,1].set_ylabel('Validation RMSE')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Objective Function Evolution
        gen_obj = self.df.groupby('Generation')['Objective'].agg(['mean', 'min', 'max', 'std'])
        axes[1,0].plot(gen_obj.index, gen_obj['mean'], 'o-', linewidth=2, markersize=8, label='Mean Objective')
        axes[1,0].fill_between(gen_obj.index, 
                              gen_obj['mean'] - gen_obj['std'], 
                              gen_obj['mean'] + gen_obj['std'], 
                              alpha=0.3, label='±1 Std Dev')
        axes[1,0].set_title('Objective Function Evolution')
        axes[1,0].set_xlabel('Generation')
        axes[1,0].set_ylabel('Objective Value')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # R² Evolution
        gen_r2 = self.df.groupby('Generation')['R2_Val'].agg(['mean', 'min', 'max', 'std'])
        axes[1,1].plot(gen_r2.index, gen_r2['mean'], 'o-', linewidth=2, markersize=8, label='Mean R²')
        axes[1,1].fill_between(gen_r2.index, 
                              gen_r2['mean'] - gen_r2['std'], 
                              gen_r2['mean'] + gen_r2['std'], 
                              alpha=0.3, label='±1 Std Dev')
        axes[1,1].set_title('Validation R² Evolution')
        axes[1,1].set_xlabel('Generation')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fitness evolution plot saved as {save_path}")
    
    def plot_hyperparameter_analysis(self, save_path='hyperparameter_analysis.png'):
        """Analyze and visualize hyperparameter distributions and their impact"""
        if self.df is None:
            print("No data available for plotting")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Hyperparameter Analysis and Impact on Performance', fontsize=16, fontweight='bold')
        
        # Top performers analysis
        top_10_percent = int(len(self.df) * 0.1)
        best_solutions = self.df.nlargest(top_10_percent, 'Objective')
        
        # Layers distribution
        layer_counts = self.df['Layers'].value_counts().sort_index()
        axes[0,0].bar(layer_counts.index, layer_counts.values, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Layer Counts')
        axes[0,0].set_xlabel('Number of Layers')
        axes[0,0].set_ylabel('Frequency')
        
        # Best solutions layer distribution
        best_layer_counts = best_solutions['Layers'].value_counts().sort_index()
        axes[0,0].bar(best_layer_counts.index, best_layer_counts.values, alpha=0.8, color='red', width=0.4)
        axes[0,0].legend(['All Solutions', 'Top 10% Solutions'])
        
        # Batch size analysis
        batch_impact = self.df.groupby('Batch')['Objective'].mean().sort_index()
        axes[0,1].bar(batch_impact.index, batch_impact.values, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Batch Size vs Average Objective')
        axes[0,1].set_xlabel('Batch Size')
        axes[0,1].set_ylabel('Average Objective')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Optimizer analysis
        opt_impact = self.df.groupby('Optimizer')['Objective'].mean().sort_values(ascending=False)
        axes[0,2].bar(range(len(opt_impact)), opt_impact.values, alpha=0.7, color='orange')
        axes[0,2].set_title('Optimizer vs Average Objective')
        axes[0,2].set_xlabel('Optimizer')
        axes[0,2].set_ylabel('Average Objective')
        axes[0,2].set_xticks(range(len(opt_impact)))
        axes[0,2].set_xticklabels(opt_impact.index, rotation=45)
        
        # Activation function analysis
        act_impact = self.df.groupby('Activation')['Objective'].mean().sort_values(ascending=False)
        axes[1,0].bar(range(len(act_impact)), act_impact.values, alpha=0.7, color='purple')
        axes[1,0].set_title('Activation Function vs Average Objective')
        axes[1,0].set_xlabel('Activation Function')
        axes[1,0].set_ylabel('Average Objective')
        axes[1,0].set_xticks(range(len(act_impact)))
        axes[1,0].set_xticklabels(act_impact.index, rotation=45)
        
        # Dropout analysis
        dropout_impact = self.df.groupby('Dropout')['Objective'].mean().sort_index()
        axes[1,1].plot(dropout_impact.index, dropout_impact.values, 'o-', linewidth=2, markersize=8)
        axes[1,1].set_title('Dropout Rate vs Average Objective')
        axes[1,1].set_xlabel('Dropout Rate')
        axes[1,1].set_ylabel('Average Objective')
        axes[1,1].grid(True, alpha=0.3)
        
        # Epochs analysis
        epoch_impact = self.df.groupby('Epochs')['Objective'].mean().sort_index()
        axes[1,2].bar(epoch_impact.index, epoch_impact.values, alpha=0.7, color='brown')
        axes[1,2].set_title('Epochs vs Average Objective')
        axes[1,2].set_xlabel('Number of Epochs')
        axes[1,2].set_ylabel('Average Objective')
        
        # RMSE vs Validation RMSE scatter
        axes[2,0].scatter(self.df['RMSE'], self.df['VAL_RMSE'], alpha=0.6, c=self.df['Objective'], cmap='viridis')
        axes[2,0].set_title('Training vs Validation RMSE')
        axes[2,0].set_xlabel('Training RMSE')
        axes[2,0].set_ylabel('Validation RMSE')
        axes[2,0].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect correlation line
        cbar = plt.colorbar(axes[2,0].collections[0], ax=axes[2,0])
        cbar.set_label('Objective Value')
        
        # R² vs Validation R² scatter
        axes[2,1].scatter(self.df['R2'], self.df['R2_Val'], alpha=0.6, c=self.df['Objective'], cmap='plasma')
        axes[2,1].set_title('Training vs Validation R²')
        axes[2,1].set_xlabel('Training R²')
        axes[2,1].set_ylabel('Validation R²')
        axes[2,1].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect correlation line
        cbar = plt.colorbar(axes[2,1].collections[0], ax=axes[2,1])
        cbar.set_label('Objective Value')
        
        # Objective distribution
        axes[2,2].hist(self.df['Objective'], bins=30, alpha=0.7, color='teal', edgecolor='black')
        axes[2,2].axvline(self.df['Objective'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["Objective"].mean():.4f}')
        axes[2,2].axvline(self.df['Objective'].max(), color='green', linestyle='--', linewidth=2, label=f'Max: {self.df["Objective"].max():.4f}')
        axes[2,2].set_title('Objective Function Distribution')
        axes[2,2].set_xlabel('Objective Value')
        axes[2,2].set_ylabel('Frequency')
        axes[2,2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Hyperparameter analysis plot saved as {save_path}")
    
    def plot_best_solutions_analysis(self, save_path='best_solutions_analysis.png'):
        """Analyze the best performing solutions in detail"""
        if self.df is None:
            print("No data available for plotting")
            return
        
        # Get top 10 solutions
        top_10 = self.df.nlargest(10, 'Objective')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Top 10 Best Solutions Analysis', fontsize=16, fontweight='bold')
        
        # Best solutions performance comparison
        x_pos = range(len(top_10))
        axes[0,0].bar(x_pos, top_10['Objective'], alpha=0.7, color='gold', edgecolor='black')
        axes[0,0].set_title('Top 10 Solutions - Objective Values')
        axes[0,0].set_xlabel('Solution Rank')
        axes[0,0].set_ylabel('Objective Value')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([f'#{i+1}' for i in range(len(top_10))])
        
        # Add value labels on bars
        for i, v in enumerate(top_10['Objective']):
            axes[0,0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Architecture comparison (layers vs neurons)
        colors = ['red', 'blue', 'green', 'orange']
        for i, (idx, row) in enumerate(top_10.iterrows()):
            layers = int(row['Layers'])
            neurons_str = str(row['Neurons'])
            # Parse neurons based on layers
            if layers == 1:
                neurons = [int(neurons_str)]
            else:
                neurons = eval(neurons_str) if isinstance(neurons_str, str) else [neurons_str]
            
            axes[0,1].bar(i, layers, alpha=0.7, color=colors[i % len(colors)], 
                         label=f'Sol #{i+1}: {layers}L-{neurons}')
        
        axes[0,1].set_title('Top 10 Solutions - Architecture (Layers)')
        axes[0,1].set_xlabel('Solution Rank')
        axes[0,1].set_ylabel('Number of Layers')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f'#{i+1}' for i in range(len(top_10))])
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Performance metrics comparison
        metrics = ['RMSE', 'VAL_RMSE', 'R2', 'R2_Val']
        x = np.arange(len(top_10))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[1,0].bar(x + i*width, top_10[metric], width, label=metric, alpha=0.8)
        
        axes[1,0].set_title('Top 10 Solutions - Performance Metrics')
        axes[1,0].set_xlabel('Solution Rank')
        axes[1,0].set_ylabel('Metric Value')
        axes[1,0].set_xticks(x + width * 1.5)
        axes[1,0].set_xticklabels([f'#{i+1}' for i in range(len(top_10))])
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Hyperparameter distribution in top solutions
        hyperparams = ['Batch', 'Optimizer', 'Activation', 'Dropout']
        hyperparam_data = []
        hyperparam_labels = []
        
        for param in hyperparams:
            if param in ['Batch', 'Dropout']:
                values = top_10[param].tolist()
            else:
                values = top_10[param].tolist()
            hyperparam_data.extend(values)
            hyperparam_labels.extend([param] * len(values))
        
        # Create a summary table
        summary_data = []
        for i, (idx, row) in enumerate(top_10.iterrows()):
            summary_data.append([
                f"#{i+1}",
                f"{int(row['Layers'])} layers",
                str(row['Neurons']),
                f"{int(row['Batch'])}",
                row['Optimizer'],
                row['Activation'],
                f"{row['Dropout']:.1f}",
                f"{row['Objective']:.4f}"
            ])
        
        # Create table
        table = axes[1,1].table(cellText=summary_data,
                               colLabels=['Rank', 'Layers', 'Neurons', 'Batch', 'Optimizer', 'Activation', 'Dropout', 'Objective'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1,1].axis('off')
        axes[1,1].set_title('Top 10 Solutions Summary Table')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Best solutions analysis plot saved as {save_path}")
    
    def plot_convergence_analysis(self, save_path='convergence_analysis.png'):
        """Analyze convergence behavior of the genetic algorithm"""
        if self.df is None:
            print("No data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Genetic Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Best, worst, and average objective per generation
        gen_stats = self.df.groupby('Generation')['Objective'].agg(['min', 'max', 'mean', 'std'])
        
        axes[0,0].plot(gen_stats.index, gen_stats['max'], 'o-', linewidth=2, markersize=8, label='Best', color='green')
        axes[0,0].plot(gen_stats.index, gen_stats['mean'], 'o-', linewidth=2, markersize=8, label='Average', color='blue')
        axes[0,0].plot(gen_stats.index, gen_stats['min'], 'o-', linewidth=2, markersize=8, label='Worst', color='red')
        axes[0,0].fill_between(gen_stats.index, 
                              gen_stats['mean'] - gen_stats['std'], 
                              gen_stats['mean'] + gen_stats['std'], 
                              alpha=0.3, label='±1 Std Dev')
        axes[0,0].set_title('Objective Function Convergence')
        axes[0,0].set_xlabel('Generation')
        axes[0,0].set_ylabel('Objective Value')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Diversity analysis (standard deviation of objective)
        axes[0,1].plot(gen_stats.index, gen_stats['std'], 'o-', linewidth=2, markersize=8, color='purple')
        axes[0,1].set_title('Population Diversity (Std Dev of Objective)')
        axes[0,1].set_xlabel('Generation')
        axes[0,1].set_ylabel('Standard Deviation')
        axes[0,1].grid(True, alpha=0.3)
        
        # Improvement rate
        improvement = gen_stats['max'].diff().fillna(0)
        axes[1,0].bar(gen_stats.index, improvement, alpha=0.7, color='orange')
        axes[1,0].set_title('Improvement Rate per Generation')
        axes[1,0].set_xlabel('Generation')
        axes[1,0].set_ylabel('Improvement in Best Objective')
        axes[1,0].grid(True, alpha=0.3)
        
        # Cumulative best
        cumulative_best = gen_stats['max'].cummax()
        axes[1,1].plot(gen_stats.index, cumulative_best, 'o-', linewidth=3, markersize=10, color='darkgreen')
        axes[1,1].set_title('Cumulative Best Objective')
        axes[1,1].set_xlabel('Generation')
        axes[1,1].set_ylabel('Best Objective So Far')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (gen, val) in enumerate(cumulative_best.items()):
            if i > 0 and val > cumulative_best.iloc[i-1]:
                axes[1,1].annotate(f'↑{val-cumulative_best.iloc[i-1]:.4f}', 
                                 xy=(gen, val), xytext=(gen, val+0.01),
                                 arrowprops=dict(arrowstyle='->', color='red'),
                                 fontsize=8, ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Convergence analysis plot saved as {save_path}")
    
    def plot_correlation_heatmap(self, save_path='correlation_heatmap.png'):
        """Create a correlation heatmap of all numeric variables"""
        if self.df is None:
            print("No data available for plotting")
            return
        
        # Select only numeric columns for correlation
        numeric_cols = ['Layers', 'Batch', 'Epochs', 'Dropout', 'Train_Percent', 
                       'RMSE', 'VAL_RMSE', 'Objective', 'MAE', 'VAL_MAE', 'R2', 'R2_Val']
        corr_data = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.3f')
        plt.title('Correlation Heatmap of All Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Correlation heatmap saved as {save_path}")
    
    def generate_summary_report(self, save_path='genetic_algorithm_summary.txt'):
        """Generate a comprehensive text summary of the results"""
        if self.df is None:
            print("No data available for summary")
            return
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GENETIC ALGORITHM OPTIMIZATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Solutions Evaluated: {len(self.df)}\n")
            f.write(f"Number of Generations: {self.df['Generation'].max()}\n")
            f.write(f"Solutions per Generation: {len(self.df) // self.df['Generation'].max()}\n\n")
            
            # Best solution
            best_solution = self.df.loc[self.df['Objective'].idxmax()]
            f.write("BEST SOLUTION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Objective Value: {best_solution['Objective']:.6f}\n")
            f.write(f"Architecture: {int(best_solution['Layers'])} layers, {best_solution['Neurons']} neurons\n")
            f.write(f"Hyperparameters:\n")
            f.write(f"  - Batch Size: {int(best_solution['Batch'])}\n")
            f.write(f"  - Optimizer: {best_solution['Optimizer']}\n")
            f.write(f"  - Activation: {best_solution['Activation']}\n")
            f.write(f"  - Dropout: {best_solution['Dropout']:.2f}\n")
            f.write(f"  - Epochs: {int(best_solution['Epochs'])}\n")
            f.write(f"  - Training Split: {best_solution['Train_Percent']:.2f}\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  - Training RMSE: {best_solution['RMSE']:.6f}\n")
            f.write(f"  - Validation RMSE: {best_solution['VAL_RMSE']:.6f}\n")
            f.write(f"  - Training R²: {best_solution['R2']:.6f}\n")
            f.write(f"  - Validation R²: {best_solution['R2_Val']:.6f}\n")
            f.write(f"  - Training MAE: {best_solution['MAE']:.6f}\n")
            f.write(f"  - Validation MAE: {best_solution['VAL_MAE']:.6f}\n\n")
            
            # Generation analysis
            f.write("GENERATION ANALYSIS:\n")
            f.write("-"*40 + "\n")
            gen_stats = self.df.groupby('Generation')['Objective'].agg(['min', 'max', 'mean', 'std'])
            for gen, stats in gen_stats.iterrows():
                f.write(f"Generation {int(gen)}:\n")
                f.write(f"  - Best Objective: {stats['max']:.6f}\n")
                f.write(f"  - Average Objective: {stats['mean']:.6f}\n")
                f.write(f"  - Worst Objective: {stats['min']:.6f}\n")
                f.write(f"  - Standard Deviation: {stats['std']:.6f}\n\n")
            
            # Hyperparameter analysis
            f.write("HYPERPARAMETER ANALYSIS:\n")
            f.write("-"*40 + "\n")
            
            # Best performing hyperparameters
            f.write("Best Performing Hyperparameters:\n")
            f.write(f"  - Most Common Layers: {self.df['Layers'].mode().iloc[0]}\n")
            f.write(f"  - Best Batch Size: {self.df.groupby('Batch')['Objective'].mean().idxmax()}\n")
            f.write(f"  - Best Optimizer: {self.df.groupby('Optimizer')['Objective'].mean().idxmax()}\n")
            f.write(f"  - Best Activation: {self.df.groupby('Activation')['Objective'].mean().idxmax()}\n")
            f.write(f"  - Best Dropout: {self.df.groupby('Dropout')['Objective'].mean().idxmax():.2f}\n")
            f.write(f"  - Best Epochs: {self.df.groupby('Epochs')['Objective'].mean().idxmax()}\n\n")
            
            # Convergence analysis
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            initial_best = gen_stats.iloc[0]['max']
            final_best = gen_stats.iloc[-1]['max']
            improvement = final_best - initial_best
            improvement_pct = (improvement / initial_best) * 100
            
            f.write(f"Initial Best Objective: {initial_best:.6f}\n")
            f.write(f"Final Best Objective: {final_best:.6f}\n")
            f.write(f"Total Improvement: {improvement:.6f} ({improvement_pct:.2f}%)\n")
            
            # Check for convergence
            if len(gen_stats) > 1:
                last_improvement = gen_stats['max'].iloc[-1] - gen_stats['max'].iloc[-2]
                f.write(f"Last Generation Improvement: {last_improvement:.6f}\n")
                if abs(last_improvement) < 0.001:
                    f.write("Status: CONVERGED (minimal improvement in last generation)\n")
                else:
                    f.write("Status: NOT CONVERGED (still improving)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Summary report saved as {save_path}")
    
    def create_all_visualizations(self, save_dir='./'):
        """Create all visualization plots and save them"""
        print("Generating comprehensive visualization suite...")
        
        # Create all plots
        self.plot_fitness_evolution(f"{save_dir}/fitness_evolution.png")
        self.plot_hyperparameter_analysis(f"{save_dir}/hyperparameter_analysis.png")
        self.plot_best_solutions_analysis(f"{save_dir}/best_solutions_analysis.png")
        self.plot_convergence_analysis(f"{save_dir}/convergence_analysis.png")
        self.plot_correlation_heatmap(f"{save_dir}/correlation_heatmap.png")
        self.generate_summary_report(f"{save_dir}/genetic_algorithm_summary.txt")
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETED!")
        print("="*60)
        print("Generated files:")
        print("1. fitness_evolution.png - Evolution of fitness metrics across generations")
        print("2. hyperparameter_analysis.png - Hyperparameter impact analysis")
        print("3. best_solutions_analysis.png - Top 10 solutions detailed analysis")
        print("4. convergence_analysis.png - GA convergence behavior")
        print("5. correlation_heatmap.png - Correlation matrix of all variables")
        print("6. genetic_algorithm_summary.txt - Comprehensive text summary")
        print("="*60)