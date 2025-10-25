"""
Main test script for probability analysis.
"""

import os
import sys
import argparse
from typing import List, Optional

from config import TestConfig, load_test_config
from generator import ProbabilityGenerator
from visualizer import ProbabilityVisualizer
from utils import set_seed, load_prompts_from_file, ensure_output_directory, print_experiment_info


def run_probability_analysis(config: TestConfig) -> dict:
    """
    Run probability analysis with the given configuration.
    
    Args:
        config: Test configuration
        
    Returns:
        Dictionary containing analysis results
    """
    # Set random seed
    set_seed(config.seed)
    
    # Print experiment info
    print_experiment_info(config)
    
    # Initialize generator
    generator = ProbabilityGenerator(
        model_name=config.model_name,
        device=config.device,
        seed=config.seed
    )
    
    # Load prompts
    prompts = [config.prompt]
    if config.prompt_file and os.path.exists(config.prompt_file):
        print(f"📄 Loading prompts from file: {config.prompt_file}")
        file_prompts = load_prompts_from_file(config.prompt_file)
        prompts.extend(file_prompts)
    
    # Ensure output directory exists
    ensure_output_directory(config.output_dir)
    
    # Run analysis for each prompt
    all_results = {}
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n🎯 Analyzing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt[:50]}...'")
        
        # Generate with probabilities
        results = generator.generate_with_probabilities(
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            num_samples=config.num_samples,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            pad_token_id=config.pad_token_id
        )
        
        # Save results
        prompt_output_dir = os.path.join(config.output_dir, f"prompt_{prompt_idx}")
        generator.save_results(results, prompt_output_dir, config.save_format)
        
        # Create visualizations if requested
        if config.save_visualizations:
            print("📊 Creating visualizations...")
            visualizer = ProbabilityVisualizer(
                tokenizer=generator.tokenizer,
                colormap=config.colormap,
                dpi=config.dpi
            )
            
            plot_paths = []
            
            # Create probability heatmap
            heatmap_path = visualizer.plot_probability_heatmap(
                results, prompt_output_dir, 
                plot_top_k=config.plot_top_k,
                figsize=(config.plot_width, config.plot_height),
                show_colorbar=config.colorbar
            )
            plot_paths.append(heatmap_path)
            
            # Create entropy plot
            entropy_path = visualizer.plot_entropy_over_time(
                results, prompt_output_dir,
                figsize=(config.plot_width, config.plot_height)
            )
            plot_paths.append(entropy_path)
            
            # Create token frequency plots
            freq_path = visualizer.plot_token_frequencies(
                results, prompt_output_dir,
                plot_top_k=config.plot_top_k,
                figsize=(config.plot_width, config.plot_height)
            )
            plot_paths.append(freq_path)
            
            # Create generation samples plot
            samples_path = visualizer.plot_generation_samples(
                results, prompt_output_dir,
                figsize=(config.plot_width, config.plot_height)
            )
            plot_paths.append(samples_path)
            
            # Create summary report
            report_path = visualizer.create_summary_report(
                results, prompt_output_dir, plot_paths
            )
            print(f"📋 Summary report: {report_path}")
        
        all_results[f"prompt_{prompt_idx}"] = results
    
    return all_results


def main():
    """Main function for running probability analysis."""
    parser = argparse.ArgumentParser(description="Run probability analysis on language models")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--prompt", help="Override prompt from command line")
    parser.add_argument("--num-samples", type=int, help="Override number of samples")
    parser.add_argument("--max-tokens", type=int, help="Override max new tokens")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_test_config(args.config)
    
    # Apply command line overrides
    if args.prompt:
        config.prompt = args.prompt
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.max_tokens:
        config.max_new_tokens = args.max_tokens
    
    # Run analysis
    try:
        results = run_probability_analysis(config)
        print("\n✅ Probability analysis completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
        
        # Print summary statistics
        import numpy as np
        for prompt_key, result in results.items():
            prob_stats = result["probability_statistics"]
            print(f"\n📊 {prompt_key} Summary:")
            print(f"   Max sequence length: {prob_stats['max_length']}")
            print(f"   Average entropy: {np.mean(prob_stats['entropy_means']):.4f}")
            print(f"   Entropy std: {np.std(prob_stats['entropy_means']):.4f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
