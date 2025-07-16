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
        return (self.args and 
                any(arg.startswith('--nnodes') or arg.startswith('--nproc-per-node') 
                    for arg in self.args) and
                '--cuda' in self.args)
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def with_assigned_gpus(self, gpu_ids: List[int]) -> 'Job':
        """Return new job with assigned GPUs"""
        return Job(
            job_id=self.job_id,
            priority=self.priority,
            script_path=self.script_path,
            conda_env=self.conda_env,
            args=self.args,
            allowed_gpus=self.allowed_gpus,
            required_gpus=self.required_gpus,
            llm_name=self.llm_name,
            job_hash=self.job_hash,
            original_line=self.original_line,
            status=self.status,
            assigned_gpus=gpu_ids,
            start_time=self.start_time,
            end_time=self.end_time
        )
    
    def with_status(self, status: JobStatus, timestamp: Optional[datetime] = None) -> 'Job':
        """Return new job with updated status"""
        ts = timestamp or datetime.now()
        return Job(
            job_id=self.job_id,
            priority=self.priority,
            script_path=self.script_path,
            conda_env=self.conda_env,
            args=self.args,
            allowed_gpus=self.allowed_gpus,
            required_gpus=self.required_gpus,
            llm_name=self.llm_name,
            job_hash=self.job_hash,
            original_line=self.original_line,
            status=status,
            assigned_gpus=self.assigned_gpus,
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
    
    def build_command(self, job: Job) -> List[str]:
        """Build command list based on job type"""
        if job.is_torchrun_job:
            return ["torchrun"] + job.args
        else:
            return ["python", "-u", job.script_path] + job.args
    
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
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get current GPU information"""
        gpu_utils = []
        try:
            gpu_utils = GPUtil.getGPUs()
        except Exception as e:
            self.logger.error(f"Failed to get GPU utilization: {e}")
        
        with self.lock:
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
    
    def pause_gpu(self, gpu_id: int) -> bool:
        """Pause GPU"""
        if not 0 <= gpu_id < self.num_gpus:
            return False
        with self.lock:
            self.paused_gpus.add(gpu_id)
        return True
    
    def resume_gpu(self, gpu_id: int) -> bool:
        """Resume GPU"""
        if not 0 <= gpu_id < self.num_gpus:
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
    
    def _get_gpu_requirement(self, llm_name: Optional[str], args: List[str]) -> float:
        """Determine GPU requirement for job"""
        # Check for torchrun job first
        if self._is_torchrun_args(args):
            return 8.0  # TODO: Make configurable
        
        # Use LLM config
        if llm_name and llm_name in self.llm_requirements:
            return float(self.llm_requirements[llm_name])
        
        return self.config.default_llm_requirement
    
    def _is_torchrun_args(self, args: List[str]) -> bool:
        """Check if arguments indicate torchrun job"""
        has_torchrun = any(arg.startswith('--nnodes') or arg.startswith('--nproc-per-node') 
                          for arg in args)
        has_cuda = '--cuda' in args
        return has_torchrun and has_cuda
    
    def parse_job_line(self, line: str, line_num: int) -> Optional[Job]:
        """Parse single job line with robust CSV handling"""
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        try:
            # Smart CSV parsing based on content
            if '--cuda' in line:
                # Torchrun format: gpu_id,script,env,args (4 fields)
                parts = [p.strip() for p in line.split(',', maxsplit=3)]
            else:
                # Standard format: priority,script,env,args,allowed_gpus (5 fields)
                parts = [p.strip() for p in line.split(',', maxsplit=4)]
            
            if len(parts) < 2:
                raise ValueError("Need at least priority and script")
            
            priority = int(parts[0])
            script_path = parts[1]
            conda_env = parts[2] if len(parts) > 2 and parts[2] else None
            args_str = parts[3] if len(parts) > 3 and parts[3] else ""
            allowed_gpus_str = parts[4] if len(parts) > 4 and parts[4] else None
            
            # Parse arguments
            args = shlex.split(args_str) if args_str else []
            llm_name = extract_arg_value(args, '--llm')
            required_gpus = self._get_gpu_requirement(llm_name, args)
            
            # Parse allowed GPUs
            allowed_gpus = None
            if allowed_gpus_str:
                allowed_gpus = self._parse_gpu_list(allowed_gpus_str)
            
            job = Job(
                job_id=str(uuid.uuid4()),
                priority=priority,
                script_path=script_path,
                conda_env=conda_env,
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
        """Save scheduler state"""
        state = self.gpu_manager.get_state()
        safe_json_save(Path(self.config.state_file), state)
    
    def add_job(self, job: Job) -> bool:
        """Add job to queue"""
        if job.job_hash in self.managed_hashes:
            return False
        
        self.managed_hashes.add(job.job_hash)
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
                    # Re-queue job
                    self.job_queue.put((job.priority, job_id, job))
                    time.sleep(self.config.assignment_retry_wait)
                
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                self.job_queue.task_done()
    
    def _try_allocate_and_run(self, job: Job) -> bool:
        """Try to allocate GPUs and run job"""
        can_allocate, gpu_ids = self.gpu_manager.can_allocate(job)
        
        if not can_allocate:
            return False
        
        # Allocate GPUs
        if not self.gpu_manager.allocate(job, gpu_ids):
            return False
        
        # Save state
        self._save_state()
        
        # Update job and start execution
        job = job.with_assigned_gpus(gpu_ids).with_status(JobStatus.RUNNING)
        self.running_jobs[job.job_id] = job
        
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
        
        # Remove from running jobs
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
        """Monitor job files for changes"""
        last_check = time.time()
        
        while not self.stop_event.is_set():
            if time.time() - last_check >= self.config.file_monitor_interval:
                try:
                    self.load_jobs_from_file(self.config.jobs_file)
                    last_check = time.time()
                except Exception as e:
                    self.logger.error(f"File monitor error: {e}")
            
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
        """Get scheduler status"""
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
                "running": len(self.running_jobs)
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