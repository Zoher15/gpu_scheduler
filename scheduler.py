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
        """Create copy with specified field changes"""
        fields = {
            'job_id': self.job_id,
            'priority': self.priority,
            'script_path': self.script_path,
            'conda_env': self.conda_env,
            'execution_type': self.execution_type,
            'args': self.args,
            'allowed_gpus': self.allowed_gpus,
            'required_gpus': self.required_gpus,
            'llm_name': self.llm_name,
            'job_hash': self.job_hash,
            'original_line': self.original_line,
            'status': self.status,
            'assigned_gpus': self.assigned_gpus,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        fields.update(changes)
        return Job(**fields)
    
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


def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """Safely load JSON file with error handling"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.getLogger("GPUJobScheduler").warning(f"Failed to load {file_path}: {e}")
        return default


def safe_json_save(file_path: Path, data: Any) -> bool:
    """Safely save JSON file with atomic write"""
    try:
        temp_file = file_path.with_suffix(f".tmp_{os.getpid()}")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, file_path)
        return True
    except Exception as e:
        logging.getLogger("GPUJobScheduler").error(f"Failed to save {file_path}: {e}")
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        return False


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


# =============================================================================
# COMMAND EXECUTION (DRY CONSOLIDATION)
# =============================================================================

class CommandExecutor:
    """Handles all command execution logic with maximum DRY"""
    
    def __init__(self, conda_manager: 'CondaManager'):
        self.conda_manager = conda_manager
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
        """Create execution script with conda activation"""
        script_lines = [
            "#!/bin/bash",
            f"# Job ID: {job.job_id}",
            f"# Hash: {job.job_hash}",
            f"# Original: {job.original_line or 'N/A'}",
            "set -e",
            f"echo 'Job {job.job_id} starting at $(date)'",
            f"echo 'GPU(s): {env.get('CUDA_VISIBLE_DEVICES', 'N/A')}'"
        ]
        
        # Add conda activation if needed
        if job.conda_env:
            script_lines.extend(self.conda_manager.get_activation_commands(job.conda_env))
        
        # Add main command
        cmd = self.build_command(job)
        script_lines.append(shlex.join(cmd))
        
        # Create temp script
        script_path = Path(f"/tmp/job_{job.job_id[:8]}.sh")
        script_path.write_text("\n".join(script_lines))
        script_path.chmod(0o755)
        
        return script_path
    
    def execute_job(self, job: Job, env: Dict[str, str]) -> subprocess.Popen:
        """Execute job with proper environment"""
        script_path = self.create_execution_script(job, env)
        
        self.logger.info(f"Executing job {job.job_id}: {job.script_path}")
        if job.is_torchrun_job:
            self.logger.info(f"Using torchrun for distributed job {job.job_id}")
        
        process = subprocess.Popen(
            [str(script_path)],
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
        self.lock = threading.Lock()
        
        # State
        self.allocations: List[float] = [0.0] * num_gpus
        self.paused_gpus: Set[int] = set()
    
    def get_gpu_info(self, reconcile_state: bool = False) -> List[GPUInfo]:
        """Get current GPU information and optionally reconcile state with live data"""
        gpu_utils = []
        freed_count = 0
        
        try:
            gpu_utils = GPUtil.getGPUs()
        except Exception as e:
            self.logger.error(f"Failed to get GPU utilization: {e}")
        
        with self.lock:
            # Reconcile state if requested - only for very low utilization to avoid false positives
            if reconcile_state and gpu_utils:
                for i in range(self.num_gpus):
                    if (i not in self.paused_gpus and 
                        self.allocations[i] > 0 and 
                        i < len(gpu_utils)):
                        
                        gpu = gpu_utils[i]
                        # Only reset if GPU shows extremely low utilization for a while
                        # This is conservative to avoid disrupting fractional allocations
                        if gpu.memoryUtil < 0.05 and gpu.load < 0.05:
                            self.logger.info(f"GPU {i} appears completely idle (mem: {gpu.memoryUtil:.1%}, "
                                           f"load: {gpu.load:.1%}), resetting allocation from {self.allocations[i]:.2f}")
                            self.allocations[i] = 0.0
                            freed_count += 1
                
                if freed_count > 0:
                    self.logger.info(f"Reconciled {freed_count} completely idle GPUs")
            
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
        with self.lock:
            return self._find_suitable_gpus(job)
    
    def _find_suitable_gpus(self, job: Job) -> Tuple[bool, List[int]]:
        """Find suitable GPUs for job (called within lock)"""
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
        with self.lock:
            allocation = job.required_gpus if job.required_gpus <= 1.0 else 1.0
            for gpu_id in gpu_ids:
                self.allocations[gpu_id] += allocation
            return True
    
    def deallocate(self, job: Job, gpu_ids: List[int]) -> bool:
        """Deallocate GPUs from job"""
        with self.lock:
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
        with self.lock:
            self.paused_gpus.add(gpu_id)
        return True
    
    def resume_gpu(self, gpu_id: int) -> bool:
        """Resume GPU"""
        if not self._validate_gpu_id(gpu_id):
            return False
        with self.lock:
            self.paused_gpus.discard(gpu_id)
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state"""
        with self.lock:
            return {
                "gpu_status": [round(a, 4) for a in self.allocations],
                "paused_gpus": sorted(list(self.paused_gpus))
            }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from dict"""
        with self.lock:
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
        self.command_executor = CommandExecutor(self.conda_manager)
        self.gpu_manager = GPUManager(num_gpus, config)
        self.job_parser = JobParser(config)
        
        # State
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_jobs: Dict[str, Job] = {}
        self.managed_hashes: Set[str] = set()
        self.stop_event = threading.Event()
        
        # Thread safety locks
        self.scheduler_lock = threading.Lock()  # Protects running_jobs and managed_hashes
        self.state_file_lock = threading.Lock()  # Protects state file operations
        
        # Threads
        self.workers: List[threading.Thread] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load scheduler state"""
        state = safe_json_load(Path(self.config.state_file), {})
        self.gpu_manager.load_state(state)
    
    def _save_state(self):
        """Save scheduler state with thread safety"""
        with self.state_file_lock:
            state = self.gpu_manager.get_state()
            safe_json_save(Path(self.config.state_file), state)
    
    def add_job(self, job: Job) -> bool:
        """Add job to queue with thread safety"""
        with self.scheduler_lock:
            if job.job_hash in self.managed_hashes:
                return False
            
            self.managed_hashes.add(job.job_hash)
        
        # Queue operations are thread-safe, do outside lock
        self.job_queue.put((job.priority, job.job_id, job))
        
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
    
    def worker(self):
        """Worker thread main loop"""
        while not self.stop_event.is_set():
            try:
                # Get job from queue
                _, job_id, job = self.job_queue.get(timeout=1.0)
                
                if self.stop_event.is_set():
                    self.job_queue.put((job.priority, job_id, job))
                    break
                
                # Try to allocate GPUs
                success = self._try_allocate_and_run(job)
                
                if not success:
                    # Re-queue job and continue immediately
                    self.job_queue.put((job.priority, job_id, job))
                
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                self.job_queue.task_done()
    
    def _try_allocate_and_run(self, job: Job) -> bool:
        """Try to allocate GPUs and run job atomically"""
        # Critical section: check availability and allocate atomically
        can_allocate, gpu_ids = self.gpu_manager.can_allocate(job)
        
        if not can_allocate:
            return False
        
        # Allocate GPUs
        if not self.gpu_manager.allocate(job, gpu_ids):
            return False
        
        # Update job tracking atomically with GPU allocation
        job = job.with_assigned_gpus(gpu_ids).with_status(JobStatus.RUNNING)
        with self.scheduler_lock:
            self.running_jobs[job.job_id] = job
        
        # Save state and start execution outside critical sections
        self._save_state()
        
        # Start job execution thread
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
            return_code = process.wait()
            
            success = return_code == 0
            final_status = JobStatus.COMPLETED if success else JobStatus.FAILED
            
        except Exception as e:
            self.logger.error(f"Job {job.job_id} execution failed: {e}")
            final_status = JobStatus.FAILED
        
        finally:
            # Cleanup
            self._cleanup_job(job, final_status)
    
    def _cleanup_job(self, job: Job, final_status: JobStatus):
        """Cleanup job after execution"""
        # Update job status
        job = job.with_status(final_status)
        
        # Deallocate GPUs
        self.gpu_manager.deallocate(job, job.assigned_gpus)
        
        # Remove from tracking collections atomically
        with self.scheduler_lock:
            self.running_jobs.pop(job.job_id, None)
            # Remove from managed hashes if failed (allow retry)
            if final_status == JobStatus.FAILED:
                self.managed_hashes.discard(job.job_hash)
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Job {job.job_id} finished: {final_status.value}")
    
    def start(self):
        """Start scheduler"""
        self.logger.info(f"Starting scheduler with {self.num_gpus} GPUs")
        
        # Load initial jobs
        if Path(self.config.jobs_file).exists():
            self.load_jobs_from_file(self.config.jobs_file)
        
        # Start workers
        for i in range(self.num_gpus):
            worker = threading.Thread(
                target=self.worker,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start file monitor
        self.monitor_thread = threading.Thread(
            target=self._monitor_files,
            name="FileMonitor",
            daemon=True
        )
        self.monitor_thread.start()
    
    def _monitor_files(self):
        """Monitor job files for changes and reconcile GPU state"""
        last_file_check = time.time()
        last_state_check = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Check for new jobs periodically
            if current_time - last_file_check >= self.config.file_monitor_interval:
                try:
                    self.load_jobs_from_file(self.config.jobs_file)
                    last_file_check = current_time
                except Exception as e:
                    self.logger.error(f"File monitor error: {e}")
            
            # Reconcile GPU state periodically
            if current_time - last_state_check >= self.config.state_check_interval:
                try:
                    # Use enhanced get_gpu_info to reconcile and save state
                    self.gpu_manager.get_gpu_info(reconcile_state=True)
                    self._save_state()
                    last_state_check = current_time
                except Exception as e:
                    self.logger.error(f"State reconciliation error: {e}")
            
            time.sleep(1)
    
    def stop(self):
        """Stop scheduler"""
        self.logger.info("Stopping scheduler")
        self.stop_event.set()
        
        # Stop workers
        for worker in self.workers:
            worker.join(timeout=10)
        
        # Stop monitor
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save final state
        self._save_state()
        
        self.logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status with thread safety"""
        # Get running job count safely
        with self.scheduler_lock:
            running_count = len(self.running_jobs)
        
        return {
            "gpus": [
                {
                    "id": gpu.gpu_id,
                    "state": gpu.state.value,
                    "allocation": f"{gpu.allocation:.2f}",
                    "memory": f"{gpu.memory_util:.1%}",
                    "load": f"{gpu.load_util:.1%}"
                }
                for gpu in self.gpu_manager.get_gpu_info()
            ],
            "jobs": {
                "queued": self.job_queue.qsize(),
                "running": running_count
            }
        }


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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.command == 'start' and args.jobs_file:
        config = Config(jobs_file=args.jobs_file)
    
    # Execute command
    if args.command == 'start':
        scheduler = GPUJobScheduler(config, args.gpus)
        scheduler.start()
        
        try:
            while True:
                time.sleep(60)
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


if __name__ == "__main__":
    main()