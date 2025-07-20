#!/usr/bin/env python3
"""
GPU Job Scheduler - Refactored Version
Extremely DRY, modular architecture with clear separation of concerns.
"""

import os
import time
import subprocess
import threading
import queue
import GPUtil
import logging
import shlex
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import hashlib
import math
import argparse
import uuid
import re


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration container"""
    # File paths
    jobs_file: str = "jobs.txt"
    llm_config_file: str = "llm_config.json"
    state_file: str = "gpu_scheduler_state.json"
    
    # Intervals
    file_monitor_interval: int = 30
    state_check_interval: int = 20
    
    # Job assignment
    max_assignment_attempts: int = 5
    assignment_retry_wait: int = 5
    
    # GPU thresholds
    gpu_memory_threshold: float = 0.8
    gpu_load_threshold: float = 0.8
    gpu_allocation_precision: float = 0.01
    
    # Logging
    screen_log_dir: Path = field(default_factory=lambda: Path("/tmp/gpu_scheduler_logs"))
    log_file: str = "gpu_scheduler.log"
    
    # Defaults
    default_llm_requirement: float = 1.0


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GPUState(Enum):
    """GPU allocation state"""
    AVAILABLE = "available"
    BUSY = "busy"
    PAUSED = "paused"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Job:
    """Immutable job representation replacing the 10-element tuple"""
    # Core identification
    job_id: str
    priority: int
    script_path: str
    
    # Execution details
    conda_env: Optional[str] = None
    execution_type: str = "python"  # "python", "torchrun-8gpu", "torchrun-4gpu", etc.
    args: List[str] = field(default_factory=list)
    
    # GPU requirements
    allowed_gpus: Optional[List[int]] = None
    required_gpus: float = 1.0
    llm_name: Optional[str] = None
    
    # Metadata
    job_hash: Optional[str] = None
    original_line: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    
    # Runtime
    assigned_gpus: List[int] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate job after creation"""
        if not self.script_path or not Path(self.script_path).is_file():
            raise ValueError(f"Invalid script path: {self.script_path}")
        if self.required_gpus <= 0:
            raise ValueError(f"Invalid GPU requirement: {self.required_gpus}")
    
    @property
    def is_torchrun_job(self) -> bool:
        """Check if this is a torchrun distributed job"""
        return self.execution_type.startswith('torchrun')
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def _copy_with_changes(self, **changes) -> 'Job':
        """Create copy with specified field changes using dataclasses.replace"""
        from dataclasses import replace
        return replace(self, **changes)
    
    def with_assigned_gpus(self, gpu_ids: List[int]) -> 'Job':
        """Return new job with assigned GPUs"""
        return self._copy_with_changes(assigned_gpus=gpu_ids)
    
    def with_status(self, status: JobStatus, timestamp: Optional[datetime] = None) -> 'Job':
        """Return new job with updated status"""
        ts = timestamp or datetime.now()
        return self._copy_with_changes(
            status=status,
            start_time=ts if status == JobStatus.RUNNING else self.start_time,
            end_time=ts if status in [JobStatus.COMPLETED, JobStatus.FAILED] else self.end_time
        )


@dataclass
class GPUInfo:
    """GPU state information"""
    gpu_id: int
    allocation: float
    is_paused: bool
    memory_util: float = 0.0
    load_util: float = 0.0
    
    @property
    def state(self) -> GPUState:
        """Get GPU state"""
        if self.is_paused:
            return GPUState.PAUSED
        elif self.allocation > 0.01:
            return GPUState.BUSY
        else:
            return GPUState.AVAILABLE
    
    @property
    def available_capacity(self) -> float:
        """Get available capacity (0.0 to 1.0)"""
        return max(0.0, 1.0 - self.allocation)


# =============================================================================
# UTILITY FUNCTIONS (MAXIMUM DRY)
# =============================================================================

def setup_logging(log_file: str = "gpu_scheduler.log") -> logging.Logger:
    """Setup centralized logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GPUJobScheduler")


class StateManager:
    """Handles all state persistence with automatic error handling"""
    
    def __init__(self, state_file: str, logger: logging.Logger):
        self.state_file = Path(state_file)
        self.logger = logger
    
    def load(self, default: Any = None) -> Any:
        """Load state from file with error handling"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load {self.state_file}: {e}")
            return default or {}
    
    def save(self, data: Any) -> bool:
        """Save state to file atomically with error handling"""
        try:
            temp_file = self.state_file.with_suffix(f".tmp_{os.getpid()}")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.state_file)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save {self.state_file}: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            return False
    
    def safe_operation(self, operation_name: str, operation_func):
        """Execute operation with automatic error logging"""
        try:
            return operation_func()
        except Exception as e:
            self.logger.error(f"{operation_name} error: {e}")
            return None


# Legacy functions for compatibility
def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """Legacy function - use StateManager instead"""
    logger = logging.getLogger("GPUJobScheduler")
    manager = StateManager(str(file_path), logger)
    return manager.load(default)


def safe_json_save(file_path: Path, data: Any) -> bool:
    """Legacy function - use StateManager instead"""
    logger = logging.getLogger("GPUJobScheduler")
    manager = StateManager(str(file_path), logger)
    return manager.save(data)


def extract_arg_value(args: List[str], key: str) -> Optional[str]:
    """Extract argument value from command line args"""
    try:
        idx = args.index(key)
        return args[idx + 1] if idx + 1 < len(args) else None
    except (ValueError, IndexError):
        return None


def sanitize_filename(value: str, max_length: int = 20) -> str:
    """Sanitize string for use in filenames"""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value))
    return sanitized[:max_length]


def calculate_job_hash(job: Job) -> str:
    """Calculate consistent hash for job deduplication"""
    hasher = hashlib.md5()
    hasher.update(str(job.priority).encode())
    hasher.update(job.script_path.encode())
    hasher.update(str(job.conda_env or '').encode())
    hasher.update(job.execution_type.encode())
    hasher.update(str(sorted(job.args)).encode())
    hasher.update(str(sorted(job.allowed_gpus or [])).encode())
    hasher.update(str(job.required_gpus).encode())
    return hasher.hexdigest()


def run_command(cmd: List[str], capture_output: bool = True, check: bool = True, 
                timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Common subprocess execution with error handling"""
    try:
        return subprocess.run(cmd, capture_output=capture_output, text=True, 
                            check=check, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.getLogger("GPUJobScheduler").error(f"Command timed out: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        logging.getLogger("GPUJobScheduler").error(f"Command failed: {' '.join(cmd)}, Error: {e}")
        raise

class ScreenManager:
    """Handles all screen session operations"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @staticmethod
    def get_screen_name(job_id: str) -> str:
        """Generate screen session name for job"""
        return f"job_{job_id[:8]}"
    
    def get_screen_list(self) -> List[str]:
        """Get list of all screen sessions with error handling"""
        try:
            result = subprocess.run(
                ["screen", "-list"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout.split('\n')
        except Exception as e:
            self.logger.error(f"Failed to get screen list: {e}")
            return []
    
    def screen_exists(self, screen_name: str) -> bool:
        """Check if screen session exists"""
        screen_lines = self.get_screen_list()
        return any(screen_name in line for line in screen_lines)
    
    def get_active_job_screens(self) -> List[Dict[str, str]]:
        """Get list of active job screen sessions"""
        screens = []
        screen_lines = self.get_screen_list()
        
        for line in screen_lines:
            if 'job_' in line:
                # Parse screen list format: "\t1077430.job_86b46b04\t(07/16/2025 06:18:03 PM)\t(Detached)"
                parts = line.split('\t')
                if len(parts) >= 4:  # Should have ['', 'session.job_id', '(date)', '(status)']
                    full_session = parts[1]  # "1077430.job_86b46b04" 
                    status = parts[-1].strip('()')  # "Detached"
                    
                    # Extract just the job part: "job_86b46b04"
                    if '.' in full_session and 'job_' in full_session:
                        screen_name = full_session.split('.', 1)[1]  # "job_86b46b04"
                        job_id = screen_name.replace('job_', '')  # "86b46b04"
                        screens.append({
                            "screen_name": screen_name,
                            "job_id": job_id,
                            "status": status
                        })
        
        return screens


def get_screen_name(job_id: str) -> str:
    """Legacy function - use ScreenManager.get_screen_name() instead"""
    return ScreenManager.get_screen_name(job_id)


# =============================================================================
# COMMAND EXECUTION (DRY CONSOLIDATION)
# =============================================================================

class CommandExecutor:
    """Handles all command execution logic with maximum DRY"""
    
    def __init__(self, conda_manager: 'CondaManager', screen_manager: 'ScreenManager'):
        self.conda_manager = conda_manager
        self.screen_manager = screen_manager
        self.logger = logging.getLogger("CommandExecutor")
    
    def _add_cuda_args_if_needed(self, cmd: List[str], job: Job) -> List[str]:
        """Add CUDA args if GPUs assigned and not already specified"""
        if job.assigned_gpus and not any('--cuda' in str(arg) for arg in job.args):
            cuda_arg = ','.join(map(str, job.assigned_gpus))
            cmd.extend(['--cuda', cuda_arg])
        return cmd
    
    def build_command(self, job: Job) -> List[str]:
        """Build command list based on execution type"""
        if job.execution_type == "python":
            cmd = ["python", "-u", job.script_path]
            cmd = self._add_cuda_args_if_needed(cmd, job)
            cmd.extend(job.args)
            return cmd
        elif job.execution_type.startswith("torchrun"):
            # Parse GPU count from execution type (e.g., "torchrun-8gpu" -> 8)
            if "-" in job.execution_type and "gpu" in job.execution_type:
                gpu_count = job.execution_type.split("-")[1].replace("gpu", "")
                torchrun_args = [f"--nnodes=1", f"--nproc-per-node={gpu_count}"]
            else:
                # Fallback for "torchrun" without GPU specification
                torchrun_args = ["--nnodes=1", "--nproc-per-node=1"]
            
            return ["torchrun"] + torchrun_args + [job.script_path] + job.args
        else:
            # Default to python execution
            self.logger.warning(f"Unknown execution type '{job.execution_type}', defaulting to python")
            cmd = ["python", "-u", job.script_path]
            cmd = self._add_cuda_args_if_needed(cmd, job)
            cmd.extend(job.args)
            return cmd
    
    def create_execution_script(self, job: Job, env: Dict[str, str]) -> Path:
        """Create execution script with conda activation and TTY preservation"""
        script_lines = [
            "#!/bin/bash",
            f"# Job ID: {job.job_id}",
            f"# Hash: {job.job_hash}",
            f"# Original: {job.original_line or 'N/A'}",
            "set -e",
            "# Setup terminal for progress bars with TTY preservation",
            "export TERM=xterm-256color",
            "export PYTHONUNBUFFERED=1",
            "# Don't force terminal size - let tqdm auto-detect",
            "stty sane 2>/dev/null || true",
            f"echo 'Job {job.job_id} starting at $(date)'",
            f"echo 'GPU(s): {env.get('CUDA_VISIBLE_DEVICES', 'N/A')}'",
            f"echo 'Working directory: $(pwd)'",
            f"echo 'Environment check:'",
            "env | grep -E '(CUDA|PATH)' || true"
        ]
        
        # Add conda activation if needed
        if job.conda_env:
            script_lines.extend(self.conda_manager.get_activation_commands(job.conda_env))
        
        # Add main command with TTY preservation using script command
        cmd = self.build_command(job)
        python_cmd_str = shlex.join(cmd)
        
        # Use script command for TTY preservation (better progress bars)
        script_lines.extend([
            f"echo 'Executing with TTY preservation: {python_cmd_str}'",
            f"# Check if script command is available",
            "if command -v script >/dev/null 2>&1; then",
            f"    echo 'Using script command for TTY preservation'",
            f"    script -q -e -c {shlex.quote(python_cmd_str)} /dev/null",
            f"    exit_code=$?",
            "else",
            f"    echo 'script command not available, running directly'",
            f"    {python_cmd_str}",
            f"    exit_code=$?",
            "fi",
            f"echo 'Job {job.job_id} completed at $(date) with exit code: $exit_code'",
            "exit $exit_code"
        ])
        
        # Create temp script
        script_path = Path(f"/tmp/job_{job.job_id[:8]}.sh")
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)
        
        return script_path
    
    def execute_job(self, job: Job, env: Dict[str, str]) -> subprocess.Popen:
        """Execute job in screen session with proper environment"""
        script_path = self.create_execution_script(job, env)
        screen_name = self.screen_manager.get_screen_name(job.job_id)
        
        self.logger.info(f"Executing job {job.job_id}: {job.script_path}")
        self.logger.info(f"Screen session: {screen_name}")
        self.logger.info(f"Assigned GPUs: {job.assigned_gpus}")
        if job.is_torchrun_job:
            self.logger.info(f"Using torchrun for distributed job {job.job_id}")
        
        # Execute job in screen session with TTY preservation for progress bars
        screen_cmd = [
            "screen", "-dmS", screen_name,
            "bash", "-c", f"cd {Path.cwd()} && TERM=xterm-256color PYTHONUNBUFFERED=1 exec {script_path}"
        ]
        
        process = subprocess.Popen(
            screen_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        return process


class CondaManager:
    """Manages conda environment setup (DRY consolidation)"""
    
    def __init__(self):
        self.logger = logging.getLogger("CondaManager")
        self._conda_script_path = self._find_conda_script()
    
    def _find_conda_script(self) -> Optional[str]:
        """Find conda initialization script"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_script = Path(conda_prefix).parent.parent / "etc/profile.d/conda.sh"
            if conda_script.exists():
                return str(conda_script)
        
        # Check common locations
        home_dir = os.environ.get('HOME', '/root')
        for path in [
            f"{home_dir}/anaconda3/etc/profile.d/conda.sh",
            f"{home_dir}/miniconda3/etc/profile.d/conda.sh",
            "/opt/conda/etc/profile.d/conda.sh"
        ]:
            if Path(path).exists():
                return path
        
        return None
    
    def get_activation_commands(self, env_name: str) -> List[str]:
        """Get conda activation commands"""
        if not self._conda_script_path:
            self.logger.warning("Conda script not found, using fallback")
            return [
                "echo 'WARNING: Conda script not found, using fallback'",
                f"conda activate {shlex.quote(env_name)}"
            ]
        
        return [
            f"source {shlex.quote(self._conda_script_path)}",
            "if [ $? -ne 0 ]; then echo 'WARNING: Conda init failed' >&2; fi",
            f"conda activate {shlex.quote(env_name)}",
            "if [ $? -ne 0 ]; then echo 'ERROR: Conda activate failed' >&2; exit 1; fi",
            "echo 'Conda environment activated'"
        ]


# =============================================================================
# GPU MANAGEMENT (DRY CONSOLIDATION)
# =============================================================================

class GPUManager:
    """Manages GPU allocation and monitoring"""
    
    def __init__(self, num_gpus: int, config: Config):
        self.num_gpus = num_gpus
        self.config = config
        self.logger = logging.getLogger("GPUManager")
        
        # State (no locks needed for single worker)
        self.allocations: List[float] = [0.0] * num_gpus
        self.paused_gpus: Set[int] = set()
    
    def get_gpu_info(self, reconcile_state: bool = False, running_jobs: Dict[str, any] = None) -> List[GPUInfo]:
        """Get current GPU information and optionally reconcile state with live data"""
        gpu_utils = []
        freed_count = 0
        
        try:
            gpu_utils = GPUtil.getGPUs()
        except Exception as e:
            self.logger.error(f"Failed to get GPU utilization: {e}")
        
        # Reconcile state if requested - Check for tracked running jobs
        if reconcile_state and gpu_utils and running_jobs is not None:
            for i in range(self.num_gpus):
                if (i not in self.paused_gpus and 
                    self.allocations[i] > 0 and 
                    i < len(gpu_utils)):
                    
                    # Check if any tracked jobs are using this GPU
                    gpu_has_tracked_jobs = any(
                        job.assigned_gpus and i in job.assigned_gpus 
                        for job in running_jobs.values()
                    )
                    
                    # Only reconcile if NO tracked jobs are using this GPU
                    if not gpu_has_tracked_jobs:
                        gpu = gpu_utils[i]
                        # Only reset if GPU shows extremely low utilization AND no tracked jobs
                        if gpu.memoryUtil < 0.05 and gpu.load < 0.05:
                            self.logger.info(f"GPU {i} appears idle with no tracked jobs "
                                           f"(mem: {gpu.memoryUtil:.1%}, load: {gpu.load:.1%}), "
                                           f"resetting allocation from {self.allocations[i]:.2f}")
                            self.allocations[i] = 0.0
                            freed_count += 1
                    else:
                        # GPU has tracked jobs, don't reset even if temporarily idle
                        self.logger.debug(f"GPU {i} has tracked jobs, skipping reconciliation despite low utilization")
            
            if freed_count > 0:
                self.logger.info(f"Reconciled {freed_count} idle GPUs with no tracked jobs")
        
        return [
            GPUInfo(
                gpu_id=i,
                allocation=self.allocations[i],
                is_paused=i in self.paused_gpus,
                memory_util=gpu_utils[i].memoryUtil if i < len(gpu_utils) else 0.0,
                load_util=gpu_utils[i].load if i < len(gpu_utils) else 0.0
            )
            for i in range(self.num_gpus)
        ]
    
    def can_allocate(self, job: Job) -> Tuple[bool, List[int]]:
        """Check if job can be allocated and return suitable GPUs"""
        return self._find_suitable_gpus(job)
    
    def _find_suitable_gpus(self, job: Job) -> Tuple[bool, List[int]]:
        """Find suitable GPUs for job"""
        candidates = [
            i for i in range(self.num_gpus)
            if i not in self.paused_gpus and 
            (job.allowed_gpus is None or i in job.allowed_gpus)
        ]
        
        if job.required_gpus <= 1.0:
            return self._find_fractional_gpu(job, candidates)
        else:
            return self._find_multi_gpu(job, candidates)
    
    def _find_fractional_gpu(self, job: Job, candidates: List[int]) -> Tuple[bool, List[int]]:
        """Find GPU for fractional allocation"""
        for gpu_id in candidates:
            if self.allocations[gpu_id] + job.required_gpus <= 1.0 + self.config.gpu_allocation_precision:
                if self._passes_utilization_check(gpu_id):
                    return True, [gpu_id]
        return False, []
    
    def _find_multi_gpu(self, job: Job, candidates: List[int]) -> Tuple[bool, List[int]]:
        """Find multiple GPUs for job"""
        needed = math.ceil(job.required_gpus)
        free_gpus = [
            i for i in candidates 
            if self.allocations[i] < self.config.gpu_allocation_precision
        ]
        
        if len(free_gpus) >= needed:
            suitable = [gpu_id for gpu_id in free_gpus[:needed] 
                       if self._passes_utilization_check(gpu_id)]
            if len(suitable) >= needed:
                return True, suitable[:needed]
        
        return False, []
    
    def _passes_utilization_check(self, gpu_id: int) -> bool:
        """Check if GPU passes utilization thresholds"""
        try:
            gpu_utils = GPUtil.getGPUs()
            if gpu_id < len(gpu_utils):
                gpu = gpu_utils[gpu_id]
                return (gpu.memoryUtil < self.config.gpu_memory_threshold and
                        gpu.load < self.config.gpu_load_threshold)
        except Exception:
            pass
        return False  # Conservative: fail if can't check
    
    def allocate(self, job: Job, gpu_ids: List[int]) -> bool:
        """Allocate GPUs to job"""
        allocation = job.required_gpus if job.required_gpus <= 1.0 else 1.0
        for gpu_id in gpu_ids:
            self.allocations[gpu_id] += allocation
        return True
    
    def deallocate(self, job: Job, gpu_ids: List[int]) -> bool:
        """Deallocate GPUs from job"""
        allocation = job.required_gpus if job.required_gpus <= 1.0 else 1.0
        for gpu_id in gpu_ids:
            self.allocations[gpu_id] = max(0.0, self.allocations[gpu_id] - allocation)
        return True
    
    def _validate_gpu_id(self, gpu_id: int) -> bool:
        """Validate GPU ID is within range"""
        return 0 <= gpu_id < self.num_gpus
    
    def pause_gpu(self, gpu_id: int) -> bool:
        """Pause GPU"""
        if not self._validate_gpu_id(gpu_id):
            return False
        self.paused_gpus.add(gpu_id)
        return True
    
    def resume_gpu(self, gpu_id: int) -> bool:
        """Resume GPU"""
        if not self._validate_gpu_id(gpu_id):
            return False
        self.paused_gpus.discard(gpu_id)
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state"""
        return {
            "gpu_status": [round(a, 4) for a in self.allocations],
            "paused_gpus": sorted(list(self.paused_gpus))
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from dict"""
        if "gpu_status" in state:
            status = state["gpu_status"]
            if len(status) == self.num_gpus:
                self.allocations = [max(0.0, min(1.0, float(s))) for s in status]
            else:
                self.logger.warning(f"GPU status length mismatch: {len(status)} vs {self.num_gpus}")
        
        if "paused_gpus" in state:
            self.paused_gpus = set(
                gpu_id for gpu_id in state["paused_gpus"] 
                if 0 <= gpu_id < self.num_gpus
            )


# =============================================================================
# JOB PARSING & VALIDATION (DRY CONSOLIDATION)
# =============================================================================

class JobParser:
    """Parses job definitions with consistent logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("JobParser")
        self.llm_requirements = self._load_llm_config()
    
    def _load_llm_config(self) -> Dict[str, float]:
        """Load LLM requirements configuration"""
        config_data = safe_json_load(Path(self.config.llm_config_file), {})
        return config_data.get("models", {})
    
    def _get_gpu_requirement(self, execution_type: str, llm_name: Optional[str]) -> float:
        """Determine GPU requirement for job"""
        # Check for torchrun job first
        if execution_type.startswith("torchrun"):
            if "-" in execution_type and "gpu" in execution_type:
                gpu_count = execution_type.split("-")[1].replace("gpu", "")
                return float(gpu_count)
            else:
                return 1.0  # Default for torchrun without GPU specification
        
        # Use LLM config
        if llm_name and llm_name in self.llm_requirements:
            return float(self.llm_requirements[llm_name])
        
        return self.config.default_llm_requirement
    
    
    def parse_job_line(self, line: str, line_num: int) -> Optional[Job]:
        """Parse single job line with new format: priority,script_path,conda_env,execution_type,args"""
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        try:
            # New format: priority,script_path,conda_env,execution_type,args,allowed_gpus
            parts = [p.strip() for p in line.split(',', maxsplit=5)]
            
            if len(parts) < 4:
                raise ValueError("Need at least priority, script_path, conda_env, and execution_type")
            
            priority = int(parts[0])
            script_path = parts[1]
            conda_env = parts[2] if parts[2] else None
            execution_type = parts[3]
            args_str = parts[4] if len(parts) > 4 and parts[4] else ""
            allowed_gpus_str = parts[5] if len(parts) > 5 and parts[5] else None
            
            # Parse arguments
            args = shlex.split(args_str) if args_str else []
            llm_name = extract_arg_value(args, '--llm')
            required_gpus = self._get_gpu_requirement(execution_type, llm_name)
            
            # Parse allowed GPUs
            allowed_gpus = None
            if allowed_gpus_str:
                allowed_gpus = self._parse_gpu_list(allowed_gpus_str)
            
            job = Job(
                job_id=str(uuid.uuid4()),
                priority=priority,
                script_path=script_path,
                conda_env=conda_env,
                execution_type=execution_type,
                args=args,
                allowed_gpus=allowed_gpus,
                required_gpus=required_gpus,
                llm_name=llm_name,
                original_line=line
            )
            
            job.job_hash = calculate_job_hash(job)
            return job
            
        except Exception as e:
            self.logger.error(f"Failed to parse line {line_num}: '{line}' - {e}")
            return None
    
    def _parse_gpu_list(self, gpu_str: str) -> List[int]:
        """Parse GPU list with ranges (e.g., '0-3,5,7')"""
        gpus = []
        for part in gpu_str.replace(" ", "").split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                gpus.extend(range(start, end + 1))
            else:
                gpus.append(int(part))
        return sorted(list(set(gpus)))


# =============================================================================
# MAIN SCHEDULER (CLEAN ARCHITECTURE)
# =============================================================================

class GPUJobScheduler:
    """Main scheduler with clean, focused responsibilities"""
    
    def __init__(self, config: Config, num_gpus: int = 8):
        self.config = config
        self.num_gpus = num_gpus
        self.logger = setup_logging(config.log_file)
        
        # Components
        self.conda_manager = CondaManager()
        self.gpu_manager = GPUManager(num_gpus, config)
        self.job_parser = JobParser(config)
        self.state_manager = StateManager(config.state_file, self.logger)
        self.screen_manager = ScreenManager(self.logger)
        self.command_executor = CommandExecutor(self.conda_manager, self.screen_manager)
        
        # State (simplified for single worker)
        self.job_queue: List[Job] = []  # Simple list, sorted by priority
        self.running_jobs: Dict[str, Job] = {}
        self.managed_hashes: Set[str] = set()
        self.stop_requested = False  # Simple boolean flag
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load scheduler state using StateManager"""
        state = self.state_manager.load({})
        self.gpu_manager.load_state(state)
    
    def _save_state(self):
        """Save scheduler state using StateManager"""
        state = self.gpu_manager.get_state()
        self.state_manager.save(state)
    
    def add_job(self, job: Job) -> bool:
        """Add job to queue"""
        if job.job_hash in self.managed_hashes:
            return False
        
        self.managed_hashes.add(job.job_hash)
        self.job_queue.append(job)
        # Keep queue sorted by priority (lower number = higher priority)
        self.job_queue.sort(key=lambda j: j.priority)
        
        self.logger.info(f"Added job {job.job_id}: {Path(job.script_path).name} "
                        f"(Priority: {job.priority}, GPUs: {job.required_gpus})")
        return True
    
    def load_jobs_from_file(self, file_path: str) -> int:
        """Load jobs from file"""
        if not Path(file_path).exists():
            return 0
        
        added = 0
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                job = self.job_parser.parse_job_line(line, line_num)
                if job and self.add_job(job):
                    added += 1
        
        self.logger.info(f"Loaded {added} jobs from {file_path}")
        return added
    
    def _try_allocate_and_run(self, job: Job) -> bool:
        """Try to allocate GPUs and run job (simplified for single worker)"""
        # Check if job can be allocated
        can_allocate, gpu_ids = self.gpu_manager.can_allocate(job)
        
        if not can_allocate:
            return False
        
        # Allocate GPUs and update job tracking
        self.gpu_manager.allocate(job, gpu_ids)
        job = job.with_assigned_gpus(gpu_ids).with_status(JobStatus.RUNNING)
        self.running_jobs[job.job_id] = job
        
        self.logger.info(f"Allocated GPUs {gpu_ids} to job {job.job_id[:8]}")
        self._save_state()
        
        # Execute job in parallel (still runs in separate screen session)
        thread = threading.Thread(
            target=self._execute_job,
            args=(job,),
            name=f"Job-{job.job_id[:8]}",
            daemon=True
        )
        thread.start()
        
        return True
    
    def _execute_job(self, job: Job):
        """Execute single job"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, job.assigned_gpus))
        
        try:
            process = self.command_executor.execute_job(job, env)
            screen_name = self.screen_manager.get_screen_name(job.job_id)
            
            # Wait for screen startup to complete
            startup_code = process.wait()
            if startup_code != 0:
                self.logger.error(f"Failed to start screen session for job {job.job_id}")
                final_status = JobStatus.FAILED
            else:
                # Monitor the screen session until it completes
                final_status = self._monitor_screen_session(screen_name, job)
            
        except Exception as e:
            self.logger.error(f"Job {job.job_id} execution failed: {e}")
            final_status = JobStatus.FAILED
        
        finally:
            # Cleanup
            self._cleanup_job(job, final_status)
    
    def _monitor_screen_session(self, screen_name: str, job: Job) -> JobStatus:
        """Monitor screen session until completion"""
        self.logger.info(f"Monitoring screen session {screen_name} for job {job.job_id[:8]}")
        
        while not self.stop_requested:
            try:
                # Check if screen session still exists using ScreenManager
                if not self.screen_manager.screen_exists(screen_name):
                    # Screen session ended, job completed
                    self.logger.info(f"Screen session {screen_name} completed for job {job.job_id[:8]}")
                    return JobStatus.COMPLETED
                
                # Wait before checking again
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error monitoring screen session {screen_name}: {e}")
                return JobStatus.FAILED
        
        # Scheduler is stopping
        return JobStatus.FAILED
    
    def _cleanup_job(self, job: Job, final_status: JobStatus):
        """Cleanup job after execution (simplified for single worker)"""
        # Update job status
        job = job.with_status(final_status)
        
        # Deallocate GPUs and update tracking
        self.gpu_manager.deallocate(job, job.assigned_gpus)
        self.running_jobs.pop(job.job_id, None)
        
        # Remove from managed hashes if failed (allow retry)
        if final_status == JobStatus.FAILED:
            self.managed_hashes.discard(job.job_hash)
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Job {job.job_id} finished: {final_status.value}, GPUs {job.assigned_gpus} freed")
    
    def run(self):
        """Run scheduler main loop (simplified single worker)"""
        self.logger.info(f"Starting scheduler with {self.num_gpus} GPUs")
        
        # Load initial jobs
        if Path(self.config.jobs_file).exists():
            self.load_jobs_from_file(self.config.jobs_file)
        
        last_check = time.time()
        
        # Main scheduler loop
        while not self.stop_requested:
            current_time = time.time()
            
            # Run periodic tasks every 20 seconds
            if current_time - last_check >= self.config.state_check_interval:
                self._run_periodic_tasks()
                last_check = current_time
            
            # Process pending jobs
            self._process_job_queue()
            
            # Brief sleep to prevent busy waiting
            time.sleep(0.1)
        
        self.logger.info("Scheduler stopped")
    
    def _run_periodic_tasks(self):
        """Run all periodic maintenance tasks with unified error handling"""
        # 1. Reload state from file (respects manual pause/resume)
        self.state_manager.safe_operation("State reload", self._load_state)
        
        # 2. Load new jobs from file
        self.state_manager.safe_operation(
            "Job loading", 
            lambda: self.load_jobs_from_file(self.config.jobs_file)
        )
        
        # 3. Reconcile GPU state with live utilization
        def reconcile_gpu_state():
            current_running_jobs = dict(self.running_jobs)
            self.gpu_manager.get_gpu_info(reconcile_state=True, running_jobs=current_running_jobs)
        
        self.state_manager.safe_operation("GPU reconciliation", reconcile_gpu_state)
        
        # 4. Save state to file
        self.state_manager.safe_operation("State save", self._save_state)
    
    def _process_job_queue(self):
        """Process pending jobs in queue"""
        if not self.job_queue:
            return
        
        # Try to run jobs starting from highest priority
        for i, job in enumerate(self.job_queue):
            if self._try_allocate_and_run(job):
                # Job started successfully, remove from queue
                self.job_queue.pop(i)
                break  # Only start one job per iteration to maintain fairness
    
    def stop(self):
        """Stop scheduler"""
        self.logger.info("Stopping scheduler")
        self.stop_requested = True
        
        # Save final state
        self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status (simplified for single worker)"""
        return {
            "gpus": [
                {
                    "id": gpu.gpu_id,
                    "state": gpu.state.value,
                    "allocation": f"{gpu.allocation:.2f}",
                    "memory": f"{gpu.memory_util:.1%}",
                    "load": f"{gpu.load_util:.1%}"
                }
                for gpu in self.gpu_manager.get_gpu_info(reconcile_state=False)
            ],
            "jobs": {
                "queued": len(self.job_queue),
                "running": len(self.running_jobs)
            }
        }
    
    def get_active_screens(self) -> List[Dict[str, str]]:
        """Get list of active job screen sessions using ScreenManager"""
        return self.screen_manager.get_active_job_screens()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="GPU Job Scheduler - Refactored")
    parser.add_argument('--config', default='config.json', help='Configuration file')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start scheduler')
    start_parser.add_argument('--gpus', type=int, default=8, help='Number of GPUs')
    start_parser.add_argument('--jobs-file', help='Jobs file path')
    
    # Status command
    subparsers.add_parser('status', help='Show status')
    
    # Screens command
    subparsers.add_parser('screens', help='Show active job screen sessions')
    
    # Pause command
    pause_parser = subparsers.add_parser('pause', help='Pause GPU')
    pause_parser.add_argument('gpu_id', type=int, help='GPU ID to pause')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume GPU')
    resume_parser.add_argument('gpu_id', type=int, help='GPU ID to resume')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.command == 'start' and args.jobs_file:
        config = Config(jobs_file=args.jobs_file)
    
    # Execute command
    if args.command == 'start':
        scheduler = GPUJobScheduler(config, args.gpus)
        
        try:
            scheduler.run()  # Use new simplified main loop
        except KeyboardInterrupt:
            scheduler.stop()
    
    elif args.command == 'status':
        scheduler = GPUJobScheduler(config)
        status = scheduler.get_status()
        
        print("\n=== GPU Status ===")
        for gpu in status['gpus']:
            print(f"GPU {gpu['id']}: {gpu['state']} "
                  f"(Alloc: {gpu['allocation']}, Mem: {gpu['memory']}, Load: {gpu['load']})")
        
        print(f"\n=== Jobs ===")
        print(f"Queued: {status['jobs']['queued']}")
        print(f"Running: {status['jobs']['running']}")
    
    elif args.command == 'screens':
        scheduler = GPUJobScheduler(config)
        screens = scheduler.get_active_screens()
        
        print("\n=== Active Job Screen Sessions ===")
        if screens:
            for screen in screens:
                print(f"Screen: {screen['screen_name']} | Job: {screen['job_id']} | Status: {screen['status']}")
            print(f"\nTotal: {len(screens)} active job screens")
            print("\nTo attach to a screen: screen -r <screen_name>")
        else:
            print("No active job screens found")
    
    elif args.command == 'pause':
        scheduler = GPUJobScheduler(config)
        if scheduler.gpu_manager.pause_gpu(args.gpu_id):
            scheduler._save_state()
            print(f"GPU {args.gpu_id} paused successfully")
        else:
            print(f"Failed to pause GPU {args.gpu_id} (invalid GPU ID)")
    
    elif args.command == 'resume':
        scheduler = GPUJobScheduler(config)
        if scheduler.gpu_manager.resume_gpu(args.gpu_id):
            scheduler._save_state()
            print(f"GPU {args.gpu_id} resumed successfully")
        else:
            print(f"Failed to resume GPU {args.gpu_id} (invalid GPU ID)")


if __name__ == "__main__":
    main()