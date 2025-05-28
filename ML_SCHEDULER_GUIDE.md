# GPU Scheduler for Machine Learning Research

## New ML-Focused Features

The GPU scheduler has been enhanced with several features specifically designed for machine learning research workflows:

### 🧪 **Experiment Tracking & Organization**

The scheduler now automatically organizes your ML experiments:

- **Experiment Grouping**: Jobs are automatically grouped by experiment type based on script name and key hyperparameters
- **Hyperparameter Extraction**: Common ML parameters (--llm, --dataset, --mode, etc.) are automatically detected and tracked
- **Organized Output**: Each experiment gets its own directory structure under `experiment_results/`
- **Result Tracking**: Output files (models, logs, results) are automatically detected and cataloged

### 📁 **Automatic Output Organization**

```
experiment_results/
├── evaluate_normvio_unified_model=qwen25-7b-dataset=d32k-mode=zeroshot/
│   ├── job_abc123_zeroshot/
│   │   ├── execution_abc123.log
│   │   ├── execution_abc123.script
│   │   ├── results.json
│   │   └── model_outputs.csv
│   └── job_def456_zeroshot/
└── hyperparameter_tuning_model=bert-base/
    └── job_ghi789_lr_001/
```

### 📊 **Enhanced Experiment Logging**

The scheduler creates detailed experiment logs in `experiment_tracker.log`:

```json
{
  "timestamp": "2024-05-28T15:30:00Z",
  "event": "experiment_start", 
  "experiment_id": "evaluate_normvio_unified_model=qwen25-7b-dataset=d32k-mode=zeroshot",
  "job_id": "abc123",
  "hyperparameters": {
    "model": "qwen25-7b",
    "dataset": "d32k", 
    "mode": "zeroshot",
    "batch_size": "20"
  }
}
```

### 🔧 **Job Generation Helpers**

#### Built-in Methods (Programmatic)

```python
from scheduler import GPUJobScheduler

scheduler = GPUJobScheduler()

# Generate parameter sweep jobs
jobs = scheduler.generate_parameter_sweep_jobs(
    script_path="experiments/tune_model.py",
    conda_env="ml-env",
    parameter_grid={
        "--lr": ["0.001", "0.01", "0.1"],
        "--batch_size": ["16", "32", "64"],
        "--model": ["bert", "roberta"]
    },
    base_args=["--epochs", "10"],
    priority=3
)

# Generate model comparison jobs
jobs = scheduler.generate_model_comparison_jobs(
    script_path="experiments/evaluate.py",
    conda_env="ml-env", 
    models=["qwen25-7b", "llama3-11b"],
    datasets=["d32k", "df402k"],
    modes=["zeroshot", "zeroshot-cot"],
    priorities={"qwen25-7b": 1, "llama3-11b": 2},
    batch_sizes={"qwen25-7b": 20, "llama3-11b": 15}
)

# Save to jobs file
scheduler.save_jobs_to_file(jobs, "experiments.txt")
```

#### Command Line Tool

```bash
# Generate from configuration file
python generate_ml_jobs.py --config example_ml_experiments.json --output my_jobs.txt

# Generate model comparison from command line
python generate_ml_jobs.py \
  --comp-script experiments/evaluate.py \
  --comp-env table-talk \
  --comp-models qwen25-7b llama3-11b \
  --comp-datasets d32k df402k \
  --comp-modes zeroshot zeroshot-cot \
  --output comparison_jobs.txt

# Generate parameter sweep
python generate_ml_jobs.py \
  --sweep-script experiments/tune.py \
  --sweep-env ml-env \
  --sweep-params '{"--lr": ["0.001", "0.01"], "--batch_size": ["16", "32"]}' \
  --output sweep_jobs.txt
```

### 📈 **Enhanced Monitoring & Metrics**

New experiment-focused metrics:

```python
# Get experiment status
status = scheduler.get_experiment_status()
print(f"Active experiments: {status['active_experiments']}")

# Get enhanced performance metrics
metrics = scheduler.get_performance_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Total experiment jobs: {metrics['total_experiment_jobs']}")
```

### 🎯 **Smart Output Detection**

The scheduler automatically detects and catalogs:

- **Model files**: `*.pt`, `*.pth`, `*.pkl`
- **Results**: `*.json`, `*.csv`, `*.xlsx`
- **Logs**: `*.log`, `*.txt`
- **Any custom output files** in the job's output directory

### 🔄 **Improved Job Execution**

- **Automatic `--output_dir` injection**: Jobs automatically get an output directory parameter
- **Better error detection**: Enhanced log analysis for Python errors and success patterns
- **Conda environment handling**: Robust conda activation for different environments
- **GPU allocation**: Automatic `--cuda` parameter management

## Usage Examples for Your Research

### Example 1: Large Model Comparison Study

```bash
# Create a comprehensive model comparison
python generate_ml_jobs.py --config - << EOF
{
  "model_comparisons": [{
    "script_path": "/data3/zkachwal/Zero-shot-s2/experiments/evaluate_AI_qwen.py",
    "conda_env": "zeroshot_s2", 
    "models": ["qwen25-7b", "qwen25-32b", "llama3-11b"],
    "datasets": ["d32k", "df402k", "genimage2k"],
    "modes": ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"],
    "priorities": {"qwen25-32b": 2, "qwen25-7b": 1, "llama3-11b": 1},
    "batch_sizes": {"qwen25-32b": 5, "qwen25-7b": 20, "llama3-11b": 15}
  }]
}
EOF

# Run the experiments
python scheduler.py start --jobs-file generated_jobs.txt --screen
```

### Example 2: Hyperparameter Sweep

```bash
# Generate parameter sweep for learning rate and batch size
python generate_ml_jobs.py \
  --sweep-script experiments/finetune_model.py \
  --sweep-env table-talk \
  --sweep-params '{
    "--lr": ["1e-5", "5e-5", "1e-4"],
    "--batch_size": ["8", "16", "32"],
    "--warmup_steps": ["100", "500", "1000"]
  }' \
  --sweep-priority 3

# This generates 27 jobs (3×3×3 combinations)
```

### Example 3: Using with Your Existing Workflow

```bash
# Convert your existing jobs2.txt pattern
cat jobs2.txt | head -5
# 5,/data3/zkachwal/Zero-shot-s2/experiments/evaluate_AI_qwen.py,zeroshot_s2,--n 1 --mode zeroshot --dataset d32k --llm qwen25-7b --batch_size 20

# Now run with enhanced scheduler
python scheduler.py start --jobs-file jobs2.txt --screen

# Check experiment progress
python -c "
from scheduler import GPUJobScheduler
import json
with open('experiment_tracker.log') as f:
    for line in f:
        entry = json.loads(line)
        if entry['event'] == 'experiment_completion':
            print(f\"Experiment {entry['experiment_id']}: {'SUCCESS' if entry['success'] else 'FAILED'}\")
"
```

## Configuration File Format

The `example_ml_experiments.json` shows the full configuration format:

```json
{
  "parameter_sweeps": [
    {
      "script_path": "path/to/script.py",
      "conda_env": "environment_name",
      "parameter_grid": {
        "--param1": ["value1", "value2"],
        "--param2": ["value3", "value4"]
      },
      "base_args": ["--fixed_param", "fixed_value"],
      "priority": 3
    }
  ],
  "model_comparisons": [
    {
      "script_path": "path/to/evaluate.py", 
      "conda_env": "environment_name",
      "models": ["model1", "model2"],
      "datasets": ["dataset1", "dataset2"],
      "modes": ["mode1", "mode2"],
      "priorities": {"model1": 1, "model2": 2},
      "batch_sizes": {"model1": 20, "model2": 10}
    }
  ],
  "custom_jobs": [
    {
      "script_path": "path/to/custom.py",
      "conda_env": "environment_name", 
      "args": ["--custom", "args"],
      "priority": 5
    }
  ]
}
```

## Advanced Features

### Duration Estimation
The scheduler can estimate job duration based on historical data:

```python
duration = scheduler.estimate_job_duration("experiments/evaluate.py", {"model": "qwen25-7b"})
print(f"Estimated duration: {duration/3600:.1f} hours")
```

### Experiment Status Monitoring
```python
status = scheduler.get_experiment_status()
for exp_id, info in status['experiments'].items():
    print(f"{exp_id}: {info['total_jobs']} jobs")
```

### Health Monitoring
```python
health = scheduler.get_health_status()
print(f"System healthy: {health['healthy']}")
print(f"GPU utilization: {health['gpu_utilization_percent']:.1f}%")
```

## Migration from Old Workflow

1. **Keep existing jobs files**: The enhanced scheduler is fully backward compatible
2. **Add experiment tracking**: New jobs automatically get experiment tracking 
3. **Organized outputs**: Results are now organized in `experiment_results/`
4. **Enhanced logging**: More detailed success/failure detection
5. **Better monitoring**: Real-time experiment progress tracking

The scheduler maintains full backward compatibility while adding powerful new features for ML research workflows! 