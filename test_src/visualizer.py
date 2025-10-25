"""
Visualization tools for probability analysis results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from transformers import AutoTokenizer


class ProbabilityVisualizer:
    """
    Visualization tools for probability analysis results.
    """
    
    def __init__(self, tokenizer, colormap: str = "viridis", dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            tokenizer: Tokenizer for converting token IDs to text
            colormap: Matplotlib colormap name
            dpi: DPI for saved plots
        """
        self.tokenizer = tokenizer
        self.colormap = colormap
        self.dpi = dpi
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_probability_heatmap(
        self,
        results: Dict,
        output_dir: str,
        plot_top_k: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        show_colorbar: bool = True
    ) -> str:
        """
        Create a heatmap showing probability distributions over time.
        
        Args:
            results: Generation results dictionary
            output_dir: Directory to save the plot
            plot_top_k: Number of top tokens to show
            figsize: Figure size (width, height)
            show_colorbar: Whether to show colorbar
            
        Returns:
            Path to saved plot
        """
        prob_stats = results["probability_statistics"]
        max_length = prob_stats["max_length"]
        vocab_size = prob_stats["vocab_size"]
        
        # Get top-k tokens for each timestep
        top_tokens_per_timestep = []
        top_probs_per_timestep = []
        
        for timestep in range(max_length):
            # Get top-k tokens at this timestep
            timestep_probs = prob_stats["probability_means"][timestep]
            top_k_indices = np.argsort(timestep_probs)[-plot_top_k:][::-1]
            top_k_probs = timestep_probs[top_k_indices]
            
            top_tokens_per_timestep.append(top_k_indices)
            top_probs_per_timestep.append(top_k_probs)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for heatmap
        heatmap_data = np.array(top_probs_per_timestep).T
        
        # Create the heatmap
        im = ax.imshow(heatmap_data, cmap=self.colormap, aspect='auto')
        
        # Set labels
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Top-K Tokens')
        ax.set_title('Probability Distribution Heatmap')
        
        # Set x-axis ticks
        ax.set_xticks(range(0, max_length, max(1, max_length // 10)))
        
        # Set y-axis ticks with token text
        y_labels = []
        for i in range(plot_top_k):
            if i < len(top_tokens_per_timestep[0]):
                token_id = top_tokens_per_timestep[0][i]
                token_text = self.tokenizer.decode([token_id])
                y_labels.append(f"{token_text[:10]}...")
            else:
                y_labels.append("")
        ax.set_yticks(range(plot_top_k))
        ax.set_yticklabels(y_labels, fontsize=8)
        
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "probability_heatmap.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_entropy_over_time(
        self,
        results: Dict,
        output_dir: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Plot entropy over time to show uncertainty evolution.
        
        Args:
            results: Generation results dictionary
            output_dir: Directory to save the plot
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot
        """
        prob_stats = results["probability_statistics"]
        max_length = prob_stats["max_length"]
        
        timesteps = range(max_length)
        entropy_means = prob_stats["entropy_means"]
        entropy_stds = prob_stats["entropy_stds"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean entropy with error bars
        ax.errorbar(timesteps, entropy_means, yerr=entropy_stds, 
                   capsize=3, capthick=1, alpha=0.7)
        ax.plot(timesteps, entropy_means, 'o-', linewidth=2, markersize=4)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Over Time (Uncertainty Evolution)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "entropy_over_time.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_token_frequencies(
        self,
        results: Dict,
        output_dir: str,
        plot_top_k: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Plot token frequency distributions over time.
        
        Args:
            results: Generation results dictionary
            output_dir: Directory to save the plot
            plot_top_k: Number of top tokens to show
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot
        """
        prob_stats = results["probability_statistics"]
        max_length = prob_stats["max_length"]
        token_probabilities = prob_stats["token_probabilities"]
        
        # Get most frequent tokens across all timesteps
        all_token_probs = token_probabilities.flatten()
        top_token_indices = np.argsort(all_token_probs)[-plot_top_k:][::-1]
        
        # Create subplots for each top token
        n_cols = 4
        n_rows = (plot_top_k + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, token_idx in enumerate(top_token_indices):
            row = i // n_cols
            col = i % n_cols
            
            if row < n_rows and col < n_cols:
                ax = axes[row, col]
                
                # Plot frequency over time for this token
                frequencies = token_probabilities[:, token_idx]
                ax.plot(range(max_length), frequencies, 'o-', linewidth=2, markersize=3)
                
                # Format token text for title
                token_text = self.tokenizer.decode([token_idx])
                ax.set_title(f"'{token_text[:15]}...'", fontsize=10)
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(plot_top_k, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].set_visible(False)
        
        plt.suptitle('Token Frequency Over Time', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "token_frequencies.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_generation_samples(
        self,
        results: Dict,
        output_dir: str,
        num_samples_to_show: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Plot a sample of generated sequences.
        
        Args:
            results: Generation results dictionary
            output_dir: Directory to save the plot
            num_samples_to_show: Number of sample sequences to display
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot
        """
        all_generated_tokens = results["all_generated_tokens"]
        prompt = results["prompt"]
        
        # Select random samples to show
        num_samples = len(all_generated_tokens)
        sample_indices = np.random.choice(num_samples, 
                                        min(num_samples_to_show, num_samples), 
                                        replace=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each sample sequence
        for i, sample_idx in enumerate(sample_indices):
            tokens = all_generated_tokens[sample_idx]
            # Decode tokens to text
            text = self.tokenizer.decode(tokens)
            # Truncate for display
            display_text = text[:50] + "..." if len(text) > 50 else text
            
            ax.text(0, i, f"Sample {sample_idx}: {display_text}", 
                   fontsize=10, verticalalignment='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(sample_indices) - 0.5)
        ax.set_xlabel('Generated Sequences')
        ax.set_title(f'Generated Text Samples (Prompt: "{prompt[:30]}...")')
        ax.set_yticks(range(len(sample_indices)))
        ax.set_yticklabels([f"Sample {idx}" for idx in sample_indices])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "generation_samples.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_summary_report(
        self,
        results: Dict,
        output_dir: str,
        plot_paths: List[str]
    ) -> str:
        """
        Create a summary report with all visualizations.
        
        Args:
            results: Generation results dictionary
            output_dir: Directory to save the report
            plot_paths: List of paths to generated plots
            
        Returns:
            Path to saved report
        """
        prob_stats = results["probability_statistics"]
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Probability Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metadata {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Probability Analysis Report</h1>
            
            <div class="metadata">
                <h2>Experiment Metadata</h2>
                <p><strong>Prompt:</strong> {results['prompt']}</p>
                <p><strong>Number of Samples:</strong> {results['num_samples']}</p>
                <p><strong>Max New Tokens:</strong> {results['max_new_tokens']}</p>
                <p><strong>Temperature:</strong> {results['generation_params']['temperature']}</p>
                <p><strong>Top-p:</strong> {results['generation_params']['top_p']}</p>
                <p><strong>Top-k:</strong> {results['generation_params']['top_k']}</p>
            </div>
            
            <div class="stats">
                <h2>Statistics Summary</h2>
                <p><strong>Max Sequence Length:</strong> {prob_stats['max_length']}</p>
                <p><strong>Vocabulary Size:</strong> {prob_stats['vocab_size']}</p>
                <p><strong>Average Entropy (first 10 steps):</strong> {np.mean(prob_stats['entropy_means'][:10]):.4f}</p>
                <p><strong>Average Entropy (last 10 steps):</strong> {np.mean(prob_stats['entropy_means'][-10:]):.4f}</p>
            </div>
        """
        
        # Add plots
        for plot_path in plot_paths:
            plot_name = os.path.basename(plot_path)
            html_content += f"""
            <div class="plot">
                <h2>{plot_name.replace('_', ' ').replace('.png', '').title()}</h2>
                <img src="{plot_name}" alt="{plot_name}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "analysis_report.html")
        with open(report_path, "w") as f:
            f.write(html_content)
        
        return report_path
