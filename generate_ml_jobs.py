#!/usr/bin/env python3
"""
ML Job Generator for GPU Scheduler
Helps create job files for machine learning experiments with parameter sweeps and model comparisons.
"""

import argparse
import json
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any


class MLJobGenerator:
    """Helper class for generating ML experiment jobs."""
    
    def __init__(self):
        self.job_lines = []
        
    def add_parameter_sweep(self, script_path: str, conda_env: str,
                          parameter_grid: Dict[str, List[str]], 
                          base_args: Optional[List[str]] = None,
                          priority: int = 5) -> int:
        """
        Generates jobs for a parameter sweep experiment.
        
        Returns:
            Number of jobs generated
        """
        if base_args is None:
            base_args = []
            
        # Generate all combinations of parameters
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        jobs_added = 0
        for combination in itertools.product(*param_values):
            # Build arguments for this combination
            args = base_args.copy()
            
            # Add parameter values
            for param_name, param_value in zip(param_names, combination):
                args.extend([param_name, str(param_value)])
            
            # Create job line
            args_str = ' '.join(args)
            job_line = f"{priority},{script_path},{conda_env},{args_str}"
            self.job_lines.append(job_line)
            jobs_added += 1
            
        print(f"Generated {jobs_added} jobs for parameter sweep")
        return jobs_added

    def add_model_comparison(self, script_path: str, conda_env: str,
                           models: List[str], datasets: List[str],
                           modes: List[str], base_args: Optional[List[str]] = None,
                           priorities: Optional[Dict[str, int]] = None,
                           batch_sizes: Optional[Dict[str, int]] = None) -> int:
        """
        Generates jobs for comparing multiple models across datasets and modes.
        
        Returns:
            Number of jobs generated
        """
        if base_args is None:
            base_args = []
        if priorities is None:
            priorities = {}
        if batch_sizes is None:
            batch_sizes = {}
            
        jobs_added = 0
        for model in models:
            for dataset in datasets:
                for mode in modes:
                    args = base_args.copy()
                    args.extend(['--llm', model])
                    args.extend(['--dataset', dataset])
                    args.extend(['--mode', mode])
                    args.extend(['--cuda', '0'])  # Will be replaced by scheduler
                    
                    # Add model-specific batch size if available
                    if model in batch_sizes:
                        args.extend(['--batch_size', str(batch_sizes[model])])
                    
                    # Get priority for this model
                    priority = priorities.get(model, 5)
                    
                    args_str = ' '.join(args)
                    job_line = f"{priority},{script_path},{conda_env},{args_str}"
                    self.job_lines.append(job_line)
                    jobs_added += 1
                    
        print(f"Generated {jobs_added} jobs for model comparison")
        return jobs_added

    def add_custom_jobs(self, jobs_config: List[Dict[str, Any]]) -> int:
        """
        Adds custom jobs from a configuration list.
        
        Args:
            jobs_config: List of job dictionaries with keys:
                - script_path: str
                - conda_env: str  
                - args: List[str]
                - priority: int (optional, default 5)
        
        Returns:
            Number of jobs generated
        """
        jobs_added = 0
        for job_config in jobs_config:
            script_path = job_config['script_path']
            conda_env = job_config['conda_env']
            args = job_config.get('args', [])
            priority = job_config.get('priority', 5)
            
            args_str = ' '.join(args)
            job_line = f"{priority},{script_path},{conda_env},{args_str}"
            self.job_lines.append(job_line)
            jobs_added += 1
            
        print(f"Generated {jobs_added} custom jobs")
        return jobs_added

    def save_jobs(self, filename: str, append: bool = False, add_comments: bool = True) -> bool:
        """
        Saves generated job lines to a file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            mode = 'a' if append else 'w'
            with open(filename, mode) as f:
                if add_comments:
                    from datetime import datetime
                    f.write(f"\n# Generated ML jobs - {datetime.now().isoformat()}\n")
                    f.write(f"# Total jobs: {len(self.job_lines)}\n\n")
                
                for job_line in self.job_lines:
                    f.write(job_line + '\n')
                    
                if add_comments:
                    f.write("\n# End generated jobs\n\n")
                    
            print(f"Saved {len(self.job_lines)} jobs to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving jobs to {filename}: {e}")
            return False

    def clear_jobs(self):
        """Clears the current job list."""
        self.job_lines = []
        
    def get_job_count(self) -> int:
        """Returns the number of jobs currently generated."""
        return len(self.job_lines)

    def preview_jobs(self, limit: int = 10):
        """Prints a preview of the generated jobs."""
        print(f"\nPreview of {min(limit, len(self.job_lines))} jobs:")
        print("-" * 80)
        for i, job_line in enumerate(self.job_lines[:limit]):
            print(f"{i+1:3d}: {job_line}")
        
        if len(self.job_lines) > limit:
            print(f"... and {len(self.job_lines) - limit} more jobs")
        print("-" * 80)


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_file}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description='Generate ML experiment jobs for GPU scheduler',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, help='JSON configuration file for experiments')
    parser.add_argument('--output', type=str, default='generated_jobs.txt', 
                       help='Output jobs file (default: generated_jobs.txt)')
    parser.add_argument('--append', action='store_true', help='Append to existing file')
    parser.add_argument('--preview', type=int, default=10, help='Number of jobs to preview (0 to disable)')
    
    # Parameter sweep options
    parser.add_argument('--sweep-script', type=str, help='Script path for parameter sweep')
    parser.add_argument('--sweep-env', type=str, help='Conda environment for parameter sweep')
    parser.add_argument('--sweep-params', type=str, help='JSON string of parameter grid for sweep')
    parser.add_argument('--sweep-priority', type=int, default=5, help='Priority for sweep jobs')
    
    # Model comparison options  
    parser.add_argument('--comp-script', type=str, help='Script path for model comparison')
    parser.add_argument('--comp-env', type=str, help='Conda environment for model comparison')
    parser.add_argument('--comp-models', type=str, nargs='+', help='List of models to compare')
    parser.add_argument('--comp-datasets', type=str, nargs='+', help='List of datasets to use')
    parser.add_argument('--comp-modes', type=str, nargs='+', help='List of evaluation modes')
    
    args = parser.parse_args()
    
    generator = MLJobGenerator()
    
    # Load from config file if provided
    if args.config:
        config = load_config_from_file(args.config)
        
        # Handle parameter sweeps from config
        if 'parameter_sweeps' in config:
            for sweep_config in config['parameter_sweeps']:
                generator.add_parameter_sweep(
                    script_path=sweep_config['script_path'],
                    conda_env=sweep_config['conda_env'],
                    parameter_grid=sweep_config['parameter_grid'],
                    base_args=sweep_config.get('base_args', []),
                    priority=sweep_config.get('priority', 5)
                )
        
        # Handle model comparisons from config
        if 'model_comparisons' in config:
            for comp_config in config['model_comparisons']:
                generator.add_model_comparison(
                    script_path=comp_config['script_path'],
                    conda_env=comp_config['conda_env'],
                    models=comp_config['models'],
                    datasets=comp_config['datasets'],
                    modes=comp_config['modes'],
                    base_args=comp_config.get('base_args', []),
                    priorities=comp_config.get('priorities', {}),
                    batch_sizes=comp_config.get('batch_sizes', {})
                )
        
        # Handle custom jobs from config
        if 'custom_jobs' in config:
            generator.add_custom_jobs(config['custom_jobs'])
    
    # Handle command line parameter sweep
    if args.sweep_script and args.sweep_env and args.sweep_params:
        try:
            parameter_grid = json.loads(args.sweep_params)
            generator.add_parameter_sweep(
                script_path=args.sweep_script,
                conda_env=args.sweep_env,
                parameter_grid=parameter_grid,
                priority=args.sweep_priority
            )
        except json.JSONDecodeError as e:
            print(f"Error parsing sweep parameters JSON: {e}")
            return 1
    
    # Handle command line model comparison
    if args.comp_script and args.comp_env and args.comp_models and args.comp_datasets and args.comp_modes:
        generator.add_model_comparison(
            script_path=args.comp_script,
            conda_env=args.comp_env,
            models=args.comp_models,
            datasets=args.comp_datasets,
            modes=args.comp_modes
        )
    
    # Check if any jobs were generated
    if generator.get_job_count() == 0:
        print("No jobs generated. Please provide configuration file or command line parameters.")
        print("\nExample usage:")
        print("  python generate_ml_jobs.py --config experiments.json")
        print("  python generate_ml_jobs.py --comp-script evaluate.py --comp-env myenv \\")
        print("    --comp-models qwen25-7b llama3-11b --comp-datasets d32k df402k \\")
        print("    --comp-modes zeroshot zeroshot-cot")
        return 1
    
    # Preview jobs if requested
    if args.preview > 0:
        generator.preview_jobs(args.preview)
    
    # Save jobs
    success = generator.save_jobs(args.output, args.append)
    if success:
        print(f"\nSuccessfully generated {generator.get_job_count()} jobs!")
        print(f"Jobs saved to: {args.output}")
        print(f"Run with: python scheduler.py start --jobs-file {args.output}")
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main()) 