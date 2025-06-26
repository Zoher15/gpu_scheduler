# GPU Job Scheduler

A sophisticated Python-based GPU job scheduling system that intelligently manages GPU resources across multiple jobs, with support for fractional GPU allocation, LLM-aware scheduling, and GNU Screen integration.

## 🚀 Features

### Core Scheduling Features
- **Intelligent GPU Assignment**: Automatically assigns jobs to available GPUs based on memory and load thresholds
- **Fractional GPU Support**: Supports fractional GPU allocation (e.g., 0.5 GPU for smaller models)
- **LLM-Aware Scheduling**: Automatically determines GPU requirements based on the `--llm` argument in job commands
- **Priority Queue**: Jobs are processed based on priority (lower number = higher priority)
- **Dynamic Load Balancing**: Continuously monitors GPU utilization and reassigns jobs as needed
- **Job Retry Logic**: Automatic retry with exponential backoff for failed job assignments

### Job Execution Modes
- **Direct Execution**: Run jobs directly in subprocesses with real-time output streaming
- **GNU Screen Integration**: Execute jobs in detached screen sessions for long-running tasks
- **Conda Environment Support**: Automatic conda environment activation for jobs
- **Custom GPU ID Injection**: Automatically modifies `--cuda` arguments with assigned GPU IDs

### File-Based Management
- **Dynamic Job File Monitoring**: Automatically loads new jobs from a monitored file
- **State Persistence**: Saves and restores GPU states, paused GPUs, and job status
- **External State Control**: Modify GPU pause/resume states externally via state file
- **Comprehensive Logging**: Detailed logging with per-job output files

### Control and Monitoring
- **Real-time Status Monitoring**: View GPU status, allocation, and job queue information
- **GPU Pause/Resume**: Temporarily disable specific GPUs without stopping the scheduler
- **Screen Session Management**: List and manage active GNU Screen sessions
- **Job Deduplication**: Prevents duplicate job execution using content-based hashing

## 📋 Requirements

- Python 3.7+
- GPUtil (`pip install gputil`)
- GNU Screen (for screen mode: `sudo apt-get install screen`)
- `script` command (usually pre-installed on Linux systems)

## 🛠️ Installation

1. Ensure all dependencies are installed:
```bash
pip install gputil
sudo apt-get install screen  # If using screen mode
```

2. Clone or download `scheduler.py` and `llm_config.json`

3. Make the scheduler executable:
```bash
chmod +x scheduler.py
```

## 🚀 Quick Start

### 1. Start the Scheduler
```bash
python scheduler.py start --gpus 8 --screen
```

### 2. Add Jobs to the Queue
Create a `jobs.txt` file or use the add command:
```bash
python scheduler.py add script.py --conda myenv --args "--llm qwen2.5-7b-inst --mode train"
```

### 3. Monitor Status
```bash
python scheduler.py status
```

### 4. Control GPUs
```bash
# Pause GPU 0
python scheduler.py pause 0

# Resume GPU 0  
python scheduler.py resume 0
```

## 📝 Usage

### Starting the Scheduler

```bash
python scheduler.py start [OPTIONS]
```

**Options:**
- `--gpus NUM`: Number of GPUs to manage (default: auto-detected)
- `--jobs-file PATH`: Path to jobs file for monitoring (default: `jobs.txt`)
- `--llm-config PATH`: Path to LLM configuration file (default: `llm_config.json`)
- `--state-file PATH`: Path to state persistence file (default: `gpu_scheduler_state.json`)
- `--screen`: Enable GNU Screen sessions for job execution
- `--mem-threshold FLOAT`: GPU memory threshold (0.0-1.0, default: 0.8)
- `--load-threshold FLOAT`: GPU load threshold (0.0-1.0, default: 0.8)
- `--monitor-interval INT`: Jobs file check interval in seconds (default: 30)
- `--state-interval INT`: State file check interval in seconds (default: 20)
- `--max-assign-attempts INT`: Maximum job assignment attempts (default: 5)
- `--assign-retry-wait INT`: Base wait time between assignment attempts (default: 5)

### Job File Format

The jobs file (`jobs.txt`) uses a CSV-like format:

```
priority,script_path,conda_env,arguments,allowed_gpus
```

**Example:**
```
# Comments start with #
0,/path/to/train.py,pytorch_env,--llm qwen2.5-7b-inst --epochs 100,0-3
1,/path/to/eval.py,pytorch_env,--llm qwen2.5-3b-inst --dataset test,
2,/path/to/inference.py,,--llm qwen2.5-32b-inst --batch_size 1,4,5,6,7
```

**Fields:**
- `priority`: Lower numbers = higher priority
- `script_path`: Full path to Python script
- `conda_env`: Conda environment name (empty for base environment)
- `arguments`: Script arguments (the scheduler will add/modify `--cuda` with assigned GPU IDs)
- `allowed_gpus`: Optional GPU restrictions (comma-separated IDs, ranges like "0-3", or empty for any GPU)

### Adding Jobs

#### Using the add command:
```bash
python scheduler.py add /path/to/script.py \
    --conda myenv \
    --args "--llm qwen2.5-7b-inst --epochs 100" \
    --priority 0 \
    --gpus "0,1,2,3" \
    --output-file jobs.txt
```

#### Directly editing the jobs file:
```bash
echo "0,/path/to/script.py,myenv,--llm qwen2.5-7b-inst --epochs 100,0-3" >> jobs.txt
```

### LLM Configuration

The `llm_config.json` file defines GPU requirements for different LLM models:

```json
{
    "default_requirement": 1.0,
    "models": {
        "qwen2.5-3b-inst": 0.5,
        "qwen2.5-7b-inst": 1.0,
        "qwen2.5-32b-inst": 1.0,
        "qwen2.5-vl-3b-inst": 1.0,
        "qwen2.5-vl-7b-inst": 1.0,
        "qwen2.5-vl-32b-inst": 1.0
    }
}
```

- `default_requirement`: Default GPU requirement for unknown models
- `models`: Specific GPU requirements per model (fractional values supported)

### Status and Monitoring

#### Check GPU Status:
```bash
python scheduler.py status
```

#### List Active Screen Sessions:
```bash
python scheduler.py screens
```

#### View Logs:
- Main scheduler log: `gpu_scheduler.log`
- Individual job logs: `/tmp/gpu_scheduler_logs/job_<job_id>_<timestamp>.log`

### GPU Control

#### Pause a GPU:
```bash
python scheduler.py pause <gpu_id>
```

#### Resume a GPU:
```bash
python scheduler.py resume <gpu_id>
```

## 🔧 Advanced Features

### Screen Mode
When enabled with `--screen`, jobs run in GNU Screen sessions:
- Sessions are named: `gpujob_<gpu_ids>_<mode>_<job_id>`
- Attach to a session: `screen -r <session_name>`
- Jobs continue running if you disconnect
- Full TTY support with color output preserved

### Fractional GPU Allocation
The scheduler supports fractional GPU usage:
- A GPU with 0.5 allocation can run two 0.5-requirement jobs
- Useful for smaller models that don't need a full GPU
- Automatic load balancing across partially allocated GPUs

### Job Deduplication
Jobs are hashed based on their content to prevent duplicates:
- Same script + arguments + environment = same hash
- Failed jobs can be retried (hash is removed on failure)
- Successful jobs won't be re-queued if added again

### State Persistence
The scheduler maintains state across restarts:
- GPU allocations and pause states are saved
- Job status and assignments are preserved
- External tools can modify the state file for control

### Dynamic Monitoring
The scheduler continuously monitors:
- Jobs file for new entries (configurable interval)
- GPU utilization and availability
- State file for external pause/resume commands
- Screen sessions for completion

## 🐛 Troubleshooting

### Common Issues

1. **"GNU Screen command not found"**
   ```bash
   sudo apt-get install screen
   ```

2. **"script command not found"**
   ```bash
   sudo apt-get install util-linux
   ```

3. **Jobs not starting**
   - Check GPU thresholds (`--mem-threshold`, `--load-threshold`)
   - Verify conda environment exists
   - Check script paths are absolute
   - Review `gpu_scheduler.log` for errors

4. **State file issues**
   - Delete `gpu_scheduler_state.json` to reset state
   - Ensure proper file permissions
   - Check disk space for log directory

### Log Files
- **Main log**: `gpu_scheduler.log` - Overall scheduler activity
- **Job logs**: `/tmp/gpu_scheduler_logs/` - Individual job outputs
- **Screen logs**: Available when using `--screen` mode

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the scheduler. 