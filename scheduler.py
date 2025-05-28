# ```python
import os
import time
import subprocess
import threading
import queue # Use PriorityQueue
import GPUtil
import logging
import shlex
import json
from datetime import datetime, timezone # <<< MODIFIED >>> Add timezone
import re
import argparse
import uuid # Import UUID library
from pathlib import Path # Use pathlib
from typing import List, Dict, Tuple, Any, Optional, Set # Add type hints and Set
import hashlib # For hashing job lines
import math # For ceil
import itertools

# --- Constants ---
DEFAULT_JOBS_FILE = "jobs.txt"
DEFAULT_LLM_CONFIG_FILE = "llm_config.json" # Default LLM config file name
DEFAULT_STATE_FILE = "gpu_scheduler_state.json" # Default state file name
RUNNING_JOBS_LOG_FILE = "running_jobs.log" # <<< ADDED >>> Log file for running jobs
DONE_JOBS_LOG_FILE = "done_jobs.log"       # <<< ADDED >>> Log file for completed jobs
EXPERIMENT_RESULTS_DIR = "experiment_results"  # <<< NEW >>> Directory for experiment outputs
FILE_MONITOR_INTERVAL_S = 30 # Check jobs file every 30 seconds
STATE_CHECK_INTERVAL_S = 20 # Check state file for external changes every 20 seconds
# Directory for screen output logs (raw output including escape codes)
SCREEN_LOG_DIR = Path("/tmp/gpu_scheduler_logs")
MAX_ASSIGNMENT_ATTEMPTS = 5 # Max times a worker tries to assign a job before requeuing
ASSIGNMENT_RETRY_WAIT_S = 5 # Base wait time between assignment attempts
GPU_ALLOCATION_PRECISION = 0.01 # For float comparisons
MAX_MANAGED_HASHES = 10000 # Maximum number of job hashes to keep in memory
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5 # Number of consecutive failures before triggering circuit breaker
CIRCUIT_BREAKER_TIMEOUT = 300 # Circuit breaker timeout in seconds (5 minutes)

# <<< NEW >>> Experiment tracking constants
EXPERIMENT_LOG_FILE = "experiment_tracker.log"  # Log for experiment progress
HYPERPARAMETER_PATTERNS = {
    '--llm': 'model',
    '--dataset': 'dataset', 
    '--mode': 'mode',
    '--batch_size': 'batch_size',
    '--context': 'context',
    '--num_sequences': 'num_sequences',
    '--n': 'num_runs',
    '--num': 'num_runs',
    '--lr': 'learning_rate',
    '--epochs': 'epochs',
    '--temperature': 'temperature',
    '--max_tokens': 'max_tokens'
}

# Setup logging (ensure threadName is included)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler("gpu_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GPUJobScheduler")

# Ensure screen log directory exists
try:
    SCREEN_LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create screen log directory {SCREEN_LOG_DIR}: {e}. Screen logs may fail.")


# --- Helper Functions ---
def list_screens() -> List[str]:
    """Lists active GNU Screen sessions matching the scheduler's pattern."""
    try:
        result = subprocess.run(
            ['screen', '-list'],
            capture_output=True,
            text=True,
            check=True
        )
        # Regex to find scheduler-specific screen sessions (PID.gpujob_...)
        # MODIFIED REGEX: Allow digits and commas for GPU IDs
        screen_pattern = re.compile(r'^\s*(\d+\.gpujob_[\d,]+_\S+)\s+\(.*\)', re.MULTILINE)
        matches = screen_pattern.findall(result.stdout)
        # Extract the part after the PID and dot (the session name we set)
        session_names = [match.split('.', 1)[-1] for match in matches if '.' in match]
        return session_names
    except FileNotFoundError:
        logger.warning("GNU Screen command ('screen') not found. Cannot list sessions.")
        return []
    except subprocess.CalledProcessError as e:
        # Handle cases where no screen sessions exist gracefully
        if "No Sockets found" in e.stdout or "No Sockets found" in e.stderr:
            logger.info("No active screen sessions found.")
            return []
        logger.error(f"Error listing screen sessions: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing screen sessions: {e}")
        return []

def _extract_arg_value(args: Optional[List[str]], key: str) -> Optional[str]:
    """Extracts the value of a specific argument (e.g., --llm)."""
    value = None
    if args:
        try:
            key_index = args.index(key)
            if key_index + 1 < len(args):
                value = args[key_index + 1]
        except ValueError:
            pass # Key not found
    return value

def _extract_and_sanitize_key_arg(args: Optional[List[str]], key: str = '--mode') -> str:
    """
    Extracts a specific argument value (defaulting to '--mode') from a list of arguments
    and sanitizes it for use in filenames or tags.
    """
    value = _extract_arg_value(args, key) or "unknown"

    # Sanitize the value: replace non-alphanumeric with underscores
    sanitized_value = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value))
    # Optional: Truncate if values can be very long
    max_len = 20
    return sanitized_value[:max_len]

# --- Main Scheduler Class ---
class GPUJobScheduler:
    """
    Manages a queue of GPU jobs, assigns them to available GPUs based on load,
    memory, and fractional GPU requirements derived from the --llm argument.
    Supports dynamic job file monitoring and GNU Screen execution.
    Automatically modifies '--cuda <gpu_id>' to pass assigned GPU IDs.
    Periodically reloads paused GPU state from the state file to reflect external changes.
    Uses `script` command for screen logging to preserve TTY behavior.
    Logs running and completed jobs to separate files. # <<< MODIFIED >>>
    """
    def __init__(self,
                 num_gpus: int = 8,
                 gpu_memory_threshold: float = 0.8,
                 gpu_load_threshold: float = 0.8,
                 jobs_file_path: Optional[str] = None, # Path to jobs file
                 llm_config_path: str = DEFAULT_LLM_CONFIG_FILE, # Path to LLM config
                 state_file_path: str = DEFAULT_STATE_FILE, # Path to state file
                 monitor_interval: int = FILE_MONITOR_INTERVAL_S, # Interval for file check
                 state_check_interval: int = STATE_CHECK_INTERVAL_S, # Interval for state check
                 max_assignment_attempts: int = MAX_ASSIGNMENT_ATTEMPTS,
                 assignment_retry_wait: int = ASSIGNMENT_RETRY_WAIT_S
                 ):
        """
        Initializes the scheduler.

        Args:
            num_gpus: Number of GPUs the scheduler should manage.
            gpu_memory_threshold: Memory utilization threshold (0.0-1.0).
            gpu_load_threshold: Load utilization threshold (0.0-1.0).
            jobs_file_path: Optional path to the jobs file for monitoring.
            llm_config_path: Path to the LLM configuration JSON file.
            state_file_path: Path to the scheduler state JSON file.
            monitor_interval: Interval in seconds to check the jobs file.
            state_check_interval: Interval in seconds to check the state file for pause changes.
            max_assignment_attempts: Max times a worker tries to assign a job before requeuing.
            assignment_retry_wait: Base wait time (s) between assignment attempts.
        """
        self.use_screen: bool = False # Whether to use GNU Screen for jobs
        self.num_gpus: int = num_gpus
        # --- Job Tuple Format (REVISED) ---
        # (priority, job_id, script_path, conda_env, args, allowed_gpus,
        #  job_hash, required_gpus, llm_name, original_line)
        # required_gpus: float, derived from llm_config based on --llm arg
        # llm_name: str or None
        # original_line: str or None (The raw line from jobs.txt if applicable)
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        # --- GPU Status Tracking ---
        # List of floats representing allocated capacity (0.0 = free, 1.0 = fully busy)
        self.gpu_status: List[float] = [0.0] * num_gpus
        # -------------------------
        self.gpu_memory_threshold: float = gpu_memory_threshold
        self.gpu_load_threshold: float = gpu_load_threshold
        self.lock: threading.Lock = threading.Lock() # Lock for shared resources (queue, status, hashes, paused_gpus)
        self.job_log_lock: threading.Lock = threading.Lock() # <<< ADDED >>> Lock for writing to running/done job logs
        self.stop_event: threading.Event = threading.Event() # Event to signal threads to stop
        self.worker_threads: List[threading.Thread] = [] # List of worker threads
        self.paused_gpus: set[int] = set() # Set of GPU IDs that are manually paused (IN MEMORY)
        self.state_file: Path = Path(state_file_path) # File to persist/read GPU status/paused state
        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_retry_wait = assignment_retry_wait

        # --- Circuit Breaker for Repeated Failures ---
        self.circuit_breaker_failures = 0  # Count of consecutive failures
        self.circuit_breaker_timeout_until = 0  # Timestamp when circuit breaker expires
        # ------------------------------------------------

        # --- Performance Monitoring ---
        self.performance_metrics = {
            'jobs_processed': 0,
            'jobs_successful': 0,
            'jobs_failed': 0,
            'total_gpu_hours_allocated': 0.0,
            'average_queue_wait_time': 0.0,
            'peak_queue_size': 0,
            'assignment_failures': 0,
            'start_time': time.time()
        }
        self.metrics_lock = threading.Lock()
        # --------------------------------

        # --- LLM Configuration ---
        self.llm_config_path = Path(llm_config_path)
        self.llm_requirements: Dict[str, float] = {}
        self.default_llm_requirement: float = 1.0
        self._load_llm_config()
        # -------------------------

        # --- Attributes for dynamic job file handling ---
        self.jobs_file_path: Optional[Path] = Path(jobs_file_path) if jobs_file_path else None
        self.file_monitor_interval: int = monitor_interval
        self.state_check_interval: int = state_check_interval
        self.monitor_thread: Optional[threading.Thread] = None # Combined monitor thread
        # Stores hashes of jobs from the file that have been added to the queue
        # during this scheduler run. Hashes are removed ONLY on failure to allow retries.
        self.managed_job_hashes: Set[str] = set()
        # Maps the hash (from file) to the unique job_id assigned when queued.
        self.hash_to_job_id: Dict[str, str] = {}
        # ----------------------------------------------------

        self.load_state() # Load previous state (paused/busy GPUs) from file into memory
        self._apply_initial_paused_state() # Log the initially loaded paused state

        # <<< NEW >>> Experiment Tracking ---
        self.experiment_results_dir = Path(EXPERIMENT_RESULTS_DIR)
        self.experiment_results_dir.mkdir(exist_ok=True)
        self.experiment_log_file = Path(EXPERIMENT_LOG_FILE)
        self.experiment_groups: Dict[str, List[str]] = {}  # experiment_id -> [job_ids]
        self.job_hyperparameters: Dict[str, Dict[str, str]] = {}  # job_id -> hyperparams
        self.experiment_lock = threading.Lock()
        # ----------------------------------

    def _load_llm_config(self):
        """Loads LLM GPU requirements from the specified JSON file."""
        logger.info(f"Loading LLM configuration from: {self.llm_config_path}")
        if not self.llm_config_path.is_file():
            logger.warning(f"LLM config file not found: {self.llm_config_path}. Using default requirement ({self.default_llm_requirement}) for all LLMs.")
            return

        try:
            with open(self.llm_config_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a JSON object")
            
            # Load and validate default requirement
            if 'default_requirement' in config:
                default_req = float(config['default_requirement'])
                if default_req <= 0:
                    raise ValueError(f"default_requirement must be positive, got: {default_req}")
                self.default_llm_requirement = default_req
                logger.info(f"Loaded default GPU requirement: {self.default_llm_requirement}")

            # Load and validate model requirements
            models = config.get('models', {})
            if not isinstance(models, dict):
                raise ValueError("'models' must be a JSON object")
            
            invalid_models = []
            for llm_name, requirement in models.items():
                try:
                    req_float = float(requirement)
                    if req_float <= 0:
                        invalid_models.append(f"{llm_name}: requirement must be positive, got {req_float}")
                        continue
                    if req_float > 8:  # Reasonable upper bound
                        logger.warning(f"LLM '{llm_name}' has unusually high GPU requirement: {req_float}")
                    self.llm_requirements[llm_name] = req_float
                except (ValueError, TypeError) as e:
                    invalid_models.append(f"{llm_name}: invalid requirement '{requirement}' - {e}")
            
            if invalid_models:
                logger.error(f"Invalid LLM configurations found:\n" + "\n".join(f"  - {error}" for error in invalid_models))
            
            logger.info(f"Successfully loaded {len(self.llm_requirements)} LLM configurations.")
            logger.debug(f"LLM requirements: {self.llm_requirements}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in LLM config file '{self.llm_config_path}': {e}")
        except Exception as e:
            logger.error(f"Failed to load LLM config from '{self.llm_config_path}': {e}")
            logger.info(f"Using default requirement ({self.default_llm_requirement}) for all LLMs.")

    def get_llm_gpu_requirement(self, llm_name: Optional[str]) -> float:
        """
        Returns the GPU requirement for a given LLM.
        If the LLM is not found in the config, returns the default requirement.
        """
        if llm_name is None:
            return self.default_llm_requirement
        return self.llm_requirements.get(llm_name, self.default_llm_requirement)

    def _extract_hyperparameters(self, args: Optional[List[str]]) -> Dict[str, str]:
        """
        Extracts hyperparameters from job arguments based on common ML patterns.
        Returns a dictionary of hyperparameter names and values.
        """
        hyperparams = {}
        if not args:
            return hyperparams
            
        try:
            for i, arg in enumerate(args):
                if arg in HYPERPARAMETER_PATTERNS and i + 1 < len(args):
                    param_name = HYPERPARAMETER_PATTERNS[arg]
                    param_value = args[i + 1]
                    hyperparams[param_name] = param_value
        except Exception as e:
            logger.warning(f"Error extracting hyperparameters from args {args}: {e}")
            
        return hyperparams

    def _generate_experiment_id(self, script_path: str, hyperparams: Dict[str, str]) -> str:
        """
        Generates a unique experiment ID based on script name and key hyperparameters.
        """
        script_name = Path(script_path).stem
        key_params = []
        
        # Priority order for experiment grouping
        param_order = ['model', 'dataset', 'mode', 'context']
        for param in param_order:
            if param in hyperparams:
                key_params.append(f"{param}={hyperparams[param]}")
        
        if key_params:
            return f"{script_name}_{'-'.join(key_params)}"
        else:
            return f"{script_name}_default"

    def _log_experiment_start(self, experiment_id: str, job_id: str, hyperparams: Dict[str, str]):
        """
        Logs the start of an experiment job with hyperparameters.
        """
        try:
            with self.experiment_lock:
                if experiment_id not in self.experiment_groups:
                    self.experiment_groups[experiment_id] = []
                self.experiment_groups[experiment_id].append(job_id)
                self.job_hyperparameters[job_id] = hyperparams
                
            # Log to experiment tracker file
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event': 'experiment_start',
                'experiment_id': experiment_id,
                'job_id': job_id,
                'hyperparameters': hyperparams
            }
            
            with open(self.experiment_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging experiment start: {e}")

    def _log_experiment_completion(self, experiment_id: str, job_id: str, success: bool, 
                                 output_files: Optional[List[str]] = None):
        """
        Logs the completion of an experiment job.
        """
        try:
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'event': 'experiment_completion',
                'experiment_id': experiment_id,
                'job_id': job_id,
                'success': success,
                'output_files': output_files or []
            }
            
            with open(self.experiment_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging experiment completion: {e}")

    def get_experiment_status(self) -> Dict[str, Any]:
        """
        Returns the current status of all experiments.
        """
        with self.experiment_lock:
            status = {
                'active_experiments': len(self.experiment_groups),
                'experiments': {}
            }
            
            for exp_id, job_ids in self.experiment_groups.items():
                status['experiments'][exp_id] = {
                    'total_jobs': len(job_ids),
                    'job_ids': job_ids
                }
                
        return status

    def _calculate_job_hash(self, priority: int, script: str, conda_env: Optional[str], args: Optional[List[str]], allowed_gpus: Optional[List[int]], required_gpus: float) -> str:
        """
        Calculates a stable MD5 hash for a job definition, including LLM requirements.
        """
        hasher = hashlib.md5()
        hasher.update(str(priority).encode())
        hasher.update(str(script).encode())
        hasher.update(str(conda_env or '').encode())
        # Hash based on the *original* arguments provided
        hasher.update(str(sorted(args) if args else []).encode())
        hasher.update(str(sorted(allowed_gpus) if allowed_gpus else []).encode()) # Sort gpus
        hasher.update(str(required_gpus).encode()) # Include GPU requirement in hash
        return hasher.hexdigest()

    # --- State Management ---
    def _apply_initial_paused_state(self):
        """Logs the currently paused GPUs after loading state initially."""
        with self.lock:
            logger.info(f"Initial paused state loaded for GPUs: {self.paused_gpus}")

    def load_state(self):
        """Loads GPU allocation/paused status from the state file into memory."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # --- Load gpu_status (now fractional allocation) ---
                loaded_status = state.get('gpu_status', [])
                # Adjust loaded status to match current num_gpus configuration
                if len(loaded_status) == self.num_gpus:
                    # Validate loaded values are floats between 0.0 and 1.0
                    self.gpu_status = [max(0.0, min(1.0, float(s))) for s in loaded_status]
                elif len(loaded_status) < self.num_gpus:
                    # Add newly detected GPUs as free
                    valid_loaded = [max(0.0, min(1.0, float(s))) for s in loaded_status]
                    self.gpu_status = valid_loaded + [0.0] * (self.num_gpus - len(loaded_status))
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Added new GPUs as FREE (0.0 allocation).")
                else:
                    # Ignore extra GPUs from old state if num_gpus decreased
                    self.gpu_status = [max(0.0, min(1.0, float(s))) for s in loaded_status[:self.num_gpus]]
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Ignoring extra GPUs from state.")

                # --- Load paused GPUs ---
                loaded_paused_list = state.get('paused_gpus', [])
                # Ensure paused GPUs are valid within the current range
                # Store in the in-memory set
                with self.lock:
                    self.paused_gpus = set(gpu_id for gpu_id in loaded_paused_list if 0 <= gpu_id < self.num_gpus)

                logger.info(f"State loaded successfully into memory. Status (Allocation): {self.gpu_status}, Paused: {self.paused_gpus}")
            except Exception as e:
                logger.error(f"Failed to load state from '{self.state_file}': {e}. Starting fresh.", exc_info=True)
                with self.lock:
                    self.gpu_status = [0.0] * self.num_gpus # Initialize as free
                    self.paused_gpus = set()
        else:
            logger.info(f"No state file found at '{self.state_file}'. Starting with all GPUs free and unpaused.")
            with self.lock:
                self.gpu_status = [0.0] * self.num_gpus
                self.paused_gpus = set()
        # Note: We don't persist managed_job_hashes. They are rebuilt on restart by scanning the file.

    def save_state(self):
        """Saves the current IN-MEMORY GPU allocation/paused status to the state file."""
        # This function is now called within the lock where allocation changes happen
        # No need for an extra lock here if called correctly
        try:
            state = {
                # Save allocation status
                "gpu_status": [round(s, 4) for s in self.gpu_status], # Round for cleaner JSON
                "paused_gpus": sorted(list(self.paused_gpus)) # Save the current in-memory paused set
                # Do not save managed_job_hashes here; they are transient
            }
            # Use atomic write if possible (write to temp file, then rename)
            temp_state_file = self.state_file.with_suffix(f".tmp_{os.getpid()}_{threading.get_ident()}") # Add thread id for uniqueness
            with open(temp_state_file, 'w') as f:
                json.dump(state, f, indent=4)
            
            # Validate the written data before committing
            try:
                with open(temp_state_file, 'r') as f:
                    validation_state = json.load(f)
                    # Basic validation checks
                    if (len(validation_state.get('gpu_status', [])) != len(self.gpu_status) or
                        set(validation_state.get('paused_gpus', [])) != self.paused_gpus):
                        raise ValueError("State validation failed: written data doesn't match memory state")
            except Exception as validation_error:
                temp_state_file.unlink()  # Clean up invalid temp file
                raise ValueError(f"State file validation failed: {validation_error}")
            
            os.replace(temp_state_file, self.state_file) # Atomic rename/replace
            logger.debug(f"Scheduler state saved successfully to {self.state_file}.")
        except Exception as e:
            # Critical error if state cannot be saved after allocation
            logger.critical(f"CRITICAL FAILURE: Failed to save state to '{self.state_file}': {e}", exc_info=True)
            # Clean up temp file on error
            if 'temp_state_file' in locals() and temp_state_file.exists():
                try:
                    temp_state_file.unlink()
                except OSError:
                    pass # Ignore cleanup error
            # Raise the exception to signal the failure to the caller (worker thread)
            raise

    def _check_and_apply_external_state_changes(self):
        """
        Reads the state file and updates the in-memory paused_gpus set
        if the file has changed (e.g., due to external pause/resume commands).
        """
        if not self.state_file.exists():
            logger.debug("State file check: File does not exist.")
            return # Nothing to load

        try:
            with open(self.state_file, 'r') as f:
                state_on_disk = json.load(f)

            paused_gpus_on_disk_list = state_on_disk.get('paused_gpus', [])
            # Validate and convert to set
            paused_gpus_on_disk = set(
                gpu_id for gpu_id in paused_gpus_on_disk_list if 0 <= gpu_id < self.num_gpus
            )

            with self.lock:
                # Compare with in-memory set
                if paused_gpus_on_disk != self.paused_gpus:
                    logger.info(f"Detected change in paused GPUs from state file. Updating memory.")
                    logger.info(f"  Old paused (memory): {self.paused_gpus}")
                    logger.info(f"  New paused (disk):   {paused_gpus_on_disk}")
                    self.paused_gpus = paused_gpus_on_disk
                    # No need to save state here, the change came *from* the file.
                    # Future saves will include this updated set.
                else:
                    logger.debug("State file check: No changes detected in paused GPUs.")

        except json.JSONDecodeError:
            logger.warning(f"State file check: Error decoding JSON from '{self.state_file}'. Skipping update.")
        except Exception as e:
            logger.error(f"State file check: Error reading or processing '{self.state_file}': {e}", exc_info=True)


    # --- GPU Control (for internal use by the main scheduler process) ---
    # These methods now only modify the IN-MEMORY state. Saving happens separately.
    # The CLI commands will use a temporary instance to modify the FILE directly.
    def _pause_gpu_internal(self, gpu_id: int) -> bool:
        """Marks a GPU as paused IN MEMORY."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} is already paused in memory.")
                return True
            self.paused_gpus.add(gpu_id)
            logger.info(f"GPU {gpu_id} marked as paused in memory.")
        return True

    def _resume_gpu_internal(self, gpu_id: int) -> bool:
        """Resumes a paused GPU IN MEMORY."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id not in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} was not paused in memory.")
                return True
            self.paused_gpus.remove(gpu_id)
            logger.info(f"GPU {gpu_id} marked as resumed in memory.")
        return True

    # --- Information Retrieval ---
    def get_gpu_status(self) -> List[Dict]:
        """Returns the current status (allocation, state) and utilization of each GPU."""
        status_list = []
        gpus_util = []
        try:
            # Get real-time utilization stats
            gpus_util = GPUtil.getGPUs()
        except Exception as e:
            logger.error(f"Could not get GPU utilization via GPUtil: {e}")

        # Get scheduler's view of state under lock
        with self.lock:
            current_gpu_allocation = list(self.gpu_status) # Get fractional allocation
            # Use the IN-MEMORY paused set for current status reporting
            current_paused_gpus = set(self.paused_gpus)

        for gpu_id in range(self.num_gpus):
            # Determine state based on scheduler's knowledge (IN MEMORY)
            allocation = current_gpu_allocation[gpu_id]
            is_paused = gpu_id in current_paused_gpus
            is_busy = allocation > GPU_ALLOCATION_PRECISION # Consider busy if allocation > ~0

            state = "PAUSED" if is_paused else \
                    f"BUSY ({allocation*100:.0f}%)" if is_busy else \
                    "AVAILABLE"

            memory_util_str = "N/A"
            load_str = "N/A"
            # Add real-time stats if available
            if gpu_id < len(gpus_util):
                try:
                    gpu = gpus_util[gpu_id]
                    memory_util_str = f"{gpu.memoryUtil * 100:.1f}%"
                    load_str = f"{gpu.load * 100:.1f}%"
                except Exception as e:
                    logger.error(f"Error getting utilization for GPU {gpu_id}: {e}")
            status_list.append({
                "gpu_id": gpu_id,
                "state": state, # Shows PAUSED, AVAILABLE, or BUSY(% allocation) based on memory state
                "allocation": f"{allocation:.2f}", # Show precise allocation
                "memory_util": memory_util_str,
                "load": load_str
            })
        return status_list

    def get_job_queue_info(self) -> List[Tuple]:
        """Returns a sorted list of jobs currently in the priority queue."""
        with self.job_queue.mutex: # Lock the queue while accessing its internal list
            # Sort by priority (lower first), then job_id (FIFO tie-breaker)
            # Job tuple: (priority, job_id, script_path, conda_env, args, allowed_gpus,
            #             job_hash, required_gpus, llm_name, original_line)
            return sorted(list(self.job_queue.queue), key=lambda x: (x[0], x[1]))

    # --- Scheduler Lifecycle ---
    def start(self):
        """Start the GPU job scheduler worker threads and the combined monitor thread."""
        if self.worker_threads or self.monitor_thread:
            logger.warning("Scheduler appears to be already started.")
            return

        logger.info(f"Starting GPU Job Scheduler with {self.num_gpus} GPUs...")
        self.stop_event.clear() # Ensure stop event is not set

        # --- Start Combined Monitor Thread ---
        # This thread handles both job file checks and state file checks
        self.monitor_thread = threading.Thread(
            target=self._monitor_files,
            name="FileAndStateMonitor",
            daemon=True # Allow program exit even if this thread is running
        )
        self.monitor_thread.start()
        # -------------------------------------

        # --- Start Worker Threads ---
        num_workers = self.num_gpus # Start one worker per GPU initially
        logger.info(f"Starting {num_workers} worker threads...")
        for i in range(num_workers):
            worker_thread = threading.Thread(target=self.worker, name=f"Worker-{i}", daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        # --------------------------

        logger.info("Scheduler started. Waiting for jobs...")

    def stop(self):
        """Signals all threads to stop and waits for them to finish."""
        logger.info("Stopping GPU Job Scheduler...")
        self.stop_event.set() # Signal all threads to stop

        # Stop the monitor thread first
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info("Waiting for file/state monitor thread to finish...")
            self.monitor_thread.join(timeout=max(5.0, self.state_check_interval + 1.0)) # Wait with timeout
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not finish within timeout.")

        # Stop worker threads by putting sentinel values in the queue
        logger.info("Signaling worker threads to stop...")
        num_sentinels = len(self.worker_threads) # Send one sentinel per worker
        for i in range(num_sentinels):
            try:
                # Sentinel: (priority, job_id=None, ..., required_gpus=0.0, llm_name=None, original_line=None)
                self.job_queue.put((float('inf'), None, None, None, None, None, None, 0.0, None, None)) # Added None for original_line
            except Exception as e:
                logger.error(f"Error putting sentinel value {i+1}/{num_sentinels} in queue: {e}")

        # Wait for worker threads to finish processing remaining items + sentinels
        logger.info("Waiting for worker threads to finish...")
        for thread in self.worker_threads:
            thread.join(timeout=10.0) # Wait with timeout
            if thread.is_alive():
                logger.warning(f"Worker thread {thread.name} did not finish within timeout.")

        # Ensure queue is empty after workers should have finished
        try:
            # Give a very short time for final task_done calls
            self.job_queue.join()
            logger.info("Job queue successfully joined (all tasks processed).")
        except Exception as e:
            logger.error(f"Error joining job queue during shutdown: {e}")
            if not self.job_queue.empty():
                # Log remaining items if queue isn't empty after join
                remaining_items = []
                while not self.job_queue.empty():
                    try:
                        remaining_items.append(self.job_queue.get_nowait())
                    except queue.Empty:
                        break
                logger.warning(f"Job queue is not empty after workers stopped ({len(remaining_items)} items remaining). Items: {remaining_items}")
                # Optionally re-queue them if persistence across restarts is desired (not current design)


        # --- Clean up managed hashes ---
        logger.info("Clearing internal managed job tracking for restart.")
        with self.lock:
            self.managed_job_hashes.clear()
            self.hash_to_job_id.clear()
        # --------------------------------------------------------------

        # Save state one last time (might include allocations from jobs that were running)
        logger.info("Saving final scheduler state (GPU allocation/paused)...")
        try:
            # Need lock for final save
            with self.lock:
                self.save_state() # Save final allocation and paused status
        except Exception as e:
            logger.error(f"Error during final state save: {e}")

        logger.info("Scheduler stopped.")

    # --- Worker Logic (REVISED) ---
    def worker(self):
        """Worker thread logic: Get job, find suitable GPU(s), launch job runner."""
        thread_name = threading.current_thread().name
        logger.info(f"Worker started.")

        while not self.stop_event.is_set():
            # --- Circuit Breaker Check ---
            current_time = time.time()
            with self.lock:
                if current_time < self.circuit_breaker_timeout_until:
                    # Circuit breaker is active, wait before processing
                    remaining_timeout = self.circuit_breaker_timeout_until - current_time
                    logger.debug(f"Circuit breaker active, waiting {remaining_timeout:.1f}s before processing jobs")
                    if self.stop_event.wait(timeout=min(remaining_timeout, 5.0)):
                        break  # Exit if stop event during circuit breaker wait
                    continue  # Skip to next iteration to recheck timeout
            # --------------------------------
            
            job_tuple = None
            job_id = None
            job_hash = None
            script_path = None
            required_gpus = 0.0
            original_line = None
            got_task = False

            try:
                # Get job from queue
                job_tuple = self.job_queue.get(block=True, timeout=1.0)
                got_task = True

                # --- Immediate shutdown check after getting task ---
                if self.stop_event.is_set():
                    logger.info(f"Stop event set after getting task. Re-queueing (if not sentinel) and exiting worker.")
                    if job_tuple[1] is not None: # Check if it's a real job (job_id is not None)
                        try:
                            self.job_queue.put(job_tuple)
                            logger.info(f"Job ID {job_tuple[1]} re-queued due to shutdown.")
                        except Exception as e:
                            logger.critical(f"CRITICAL: Failed to re-queue job ID {job_tuple[1]} during shutdown check: {e}. Job may be lost.")
                    break # Exit the worker loop immediately

                # --- Unpack job details ---
                (priority, job_id, script_path, _, original_args,
                 allowed_gpus_list, job_hash, required_gpus, llm_name, original_line) = job_tuple

                # --- Check for sentinel ---
                if job_id is None:
                    logger.info(f"Received sentinel, exiting.")
                    break # Exit the worker loop

                logger.debug(f"Retrieved job ID {job_id} (Hash: {job_hash}, LLM: {llm_name}, GPUs Req: {required_gpus:.2f})")

                # --- Assignment Loop ---
                assigned = False
                attempts = 0
                assigned_gpu_ids: List[int] = []
                gpus_util_snapshot = [] # Store GPUtil snapshot per attempt

                while not assigned and not self.stop_event.is_set() and attempts < self.max_assignment_attempts:
                    assigned_gpu_ids = [] # Reset for each attempt
                    assignment_successful_this_attempt = False # Flag for breaking outer loop

                    # Get GPUtil snapshot *once* per attempt, outside the lock if possible
                    try:
                        gpus_util_snapshot = GPUtil.getGPUs()
                        logger.debug(f"Job {job_id}, Attempt {attempts+1}: Fetched GPUtil snapshot ({len(gpus_util_snapshot)} GPUs)")
                    except Exception as e:
                        logger.error(f"Job {job_id}, Attempt {attempts+1}: GPUtil check failed: {e}. Cannot check real-time stats.")
                        gpus_util_snapshot = [] # Ensure it's empty list on failure

                    # --- Critical Section: Check and Allocate GPU(s) ---
                    with self.lock:
                        candidate_indices = [
                            i for i in range(self.num_gpus)
                            if i not in self.paused_gpus and \
                               (allowed_gpus_list is None or i in allowed_gpus_list)
                        ]
                        logger.debug(f"Job {job_id}, Attempt {attempts+1}: Candidate GPUs (not paused [memory], allowed): {candidate_indices}")

                        # --- Allocation Strategy ---
                        if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION):
                            # --- Strategy 1: Fractional or Single Full GPU ---
                            found_gpu = False
                            for gpu_id in candidate_indices:
                                current_allocation = self.gpu_status[gpu_id]
                                if (current_allocation + required_gpus) <= (1.0 + GPU_ALLOCATION_PRECISION):
                                    passes_realtime_check = True
                                    # Only perform real-time check if GPU has existing allocation and snapshot is available
                                    if current_allocation > GPU_ALLOCATION_PRECISION:
                                        if gpus_util_snapshot:
                                            if gpu_id < len(gpus_util_snapshot):
                                                gpu = gpus_util_snapshot[gpu_id]
                                                passes_mem = gpu.memoryUtil < self.gpu_memory_threshold
                                                passes_load = gpu.load < self.gpu_load_threshold
                                                passes_realtime_check = passes_mem and passes_load
                                                logger.debug(f"GPU Check [{gpu_id}] Realtime: Mem={gpu.memoryUtil:.2f}<{self.gpu_memory_threshold}({passes_mem}), Load={gpu.load:.2f}<{self.gpu_load_threshold}({passes_load})")
                                            else:
                                                passes_realtime_check = False # Mismatch
                                                logger.warning(f"GPU Check [{gpu_id}] Realtime: Mismatch between configured GPUs ({self.num_gpus}) and GPUtil snapshot ({len(gpus_util_snapshot)}).")
                                        else:
                                            passes_realtime_check = False # Cannot verify
                                            logger.warning(f"GPU Check [{gpu_id}] Realtime: Skipping check as GPUtil snapshot failed.")

                                    # If passes check (or no check needed because GPU is free)
                                    if passes_realtime_check:
                                        logger.info(f"Found suitable fractional/single GPU {gpu_id} for job ID {job_id} (Req: {required_gpus:.2f}, CurrentAlloc: {current_allocation:.2f})")
                                        self.gpu_status[gpu_id] += required_gpus # Allocate
                                        assigned_gpu_ids = [gpu_id]
                                        found_gpu = True
                                        break # Found a suitable GPU, break inner loop
                                    else:
                                         logger.debug(f"GPU Check [{gpu_id}] failed real-time check for fractional job {job_id}.")
                                else:
                                    logger.debug(f"GPU Check [{gpu_id}] insufficient capacity for job ID {job_id} (Req: {required_gpus:.2f}, CurrentAlloc: {current_allocation:.2f})")

                            if found_gpu:
                                assignment_successful_this_attempt = True

                        else: # required_gpus > 1.0
                            # --- Strategy 2: Multiple Full GPUs ---
                            num_needed = math.ceil(required_gpus)
                            logger.debug(f"Job {job_id} requires {num_needed} fully free GPUs.")
                            free_gpu_ids = [
                                i for i in candidate_indices
                                if abs(self.gpu_status[i]) < GPU_ALLOCATION_PRECISION
                            ]

                            if len(free_gpu_ids) >= num_needed:
                                suitable_free_gpus = []
                                # Perform real-time check only if snapshot is available
                                if gpus_util_snapshot:
                                    for gpu_id in free_gpu_ids:
                                        passes_realtime_check = True
                                        if gpu_id < len(gpus_util_snapshot):
                                            gpu = gpus_util_snapshot[gpu_id]
                                            passes_mem = gpu.memoryUtil < self.gpu_memory_threshold
                                            passes_load = gpu.load < self.gpu_load_threshold
                                            passes_realtime_check = passes_mem and passes_load
                                            logger.debug(f"GPU Check [{gpu_id}] Realtime (Multi-GPU): Mem={gpu.memoryUtil:.2f}<{self.gpu_memory_threshold}({passes_mem}), Load={gpu.load:.2f}<{self.gpu_load_threshold}({passes_load})")
                                        else:
                                            passes_realtime_check = False
                                            logger.warning(f"GPU Check [{gpu_id}] Realtime (Multi-GPU): Mismatch between configured GPUs ({self.num_gpus}) and GPUtil snapshot ({len(gpus_util_snapshot)}).")

                                        if passes_realtime_check:
                                            suitable_free_gpus.append(gpu_id)
                                        if len(suitable_free_gpus) == num_needed:
                                            break # Found enough suitable free GPUs
                                else: # No snapshot, assume free GPUs are okay if allocation is 0
                                    logger.warning(f"Multi-GPU Check: Skipping real-time check as GPUtil snapshot failed. Relying on allocation status only.")
                                    suitable_free_gpus = free_gpu_ids[:num_needed] # Take the first N free ones

                                # Check if enough suitable GPUs were found
                                if len(suitable_free_gpus) >= num_needed:
                                    assigned_gpu_ids = suitable_free_gpus[:num_needed] # Ensure we only take needed amount
                                    logger.info(f"Found suitable free GPUs {assigned_gpu_ids} for multi-GPU job ID {job_id} (Req: {required_gpus:.2f})")
                                    for gpu_id in assigned_gpu_ids:
                                        self.gpu_status[gpu_id] = 1.0 # Mark as fully busy
                                    assignment_successful_this_attempt = True
                                else:
                                    logger.debug(f"Found {len(free_gpu_ids)} free GPUs, but only {len(suitable_free_gpus)} passed checks for multi-GPU job {job_id} (needed {num_needed}).")
                            else:
                                logger.debug(f"Insufficient fully free GPUs for multi-GPU job ID {job_id} (Need: {num_needed}, Free & Allowed: {len(free_gpu_ids)})")

                        # --- After checking all strategies (still inside lock) ---
                        if assignment_successful_this_attempt:
                            try:
                                # Save state immediately after successful allocation modification
                                self.save_state()
                                assigned = True # Set the main flag for outer loop check
                                # Break is handled outside the lock based on 'assigned' flag
                            except Exception as save_err:
                                # save_state logs critical error and raises exception
                                # Revert allocation changes made in this attempt
                                logger.error(f"Reverting allocation for job {job_id} due to state save failure.")
                                allocation_to_revert = 0.0
                                if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION): allocation_to_revert = required_gpus
                                else: allocation_to_revert = 1.0

                                for gpu_id in assigned_gpu_ids: # Use the IDs we tried to assign
                                    if 0 <= gpu_id < len(self.gpu_status):
                                        # Subtract the allocation we added
                                        self.gpu_status[gpu_id] = max(0.0, self.gpu_status[gpu_id] - allocation_to_revert)
                                        logger.debug(f"Reverted allocation for GPU {gpu_id}. New status: {self.gpu_status[gpu_id]:.2f}")

                                assigned_gpu_ids = [] # Clear assigned GPUs
                                assignment_successful_this_attempt = False # Mark as failed
                                assigned = False # Ensure outer loop continues or fails

                    # <<< Lock Released >>>

                    # --- Post-Attempt Check ---
                    if assigned: # Check if assignment succeeded *and* state was saved
                         break # Exit the assignment attempt loop (while not assigned...)
                    else:
                        # No suitable GPU(s) found OR state save failed in this attempt
                        attempts += 1
                        if attempts < self.max_assignment_attempts:
                            logger.debug(f"Assignment attempt {attempts}/{self.max_assignment_attempts} failed for job ID {job_id}. Will wait and retry.")
                            # Wait *outside* the lock before next attempt
                            wait_time = min(self.assignment_retry_wait * (attempts), 30) # Exponential backoff capped at 30s
                            if self.stop_event.wait(timeout=wait_time):
                                logger.info(f"Stop event received while waiting to assign job {job_id}. Re-queueing.")
                                try:
                                    self.job_queue.put(job_tuple)
                                except Exception as e:
                                    logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} after max attempts: {e}. Job may be lost.")
                            # If stop_event caused the loop break, the job was already re-queued inside the loop

                # --- End of assignment attempt loop ---

                # --- Launch Job or Re-queue ---
                if assigned: # Check if ANY attempt succeeded
                    # Successfully found/reserved GPU(s) and saved state
                    _, run_job_id, _, _, run_args, _, _, run_req_gpus, run_llm_name, _ = job_tuple
                    mode_tag = _extract_and_sanitize_key_arg(run_args)
                    # **** Log the assignment ****
                    logger.info(f"Assigning job ID {run_job_id} (LLM: {run_llm_name}, Req: {run_req_gpus:.2f}) to GPU(s) {assigned_gpu_ids}")

                    # Reset circuit breaker on successful assignment
                    with self.lock:
                        self.circuit_breaker_failures = 0
                        self.circuit_breaker_timeout_until = 0

                    # Start a new thread to run the job
                    job_runner_thread = threading.Thread(
                        target=self._run_job,
                        args=(job_tuple, assigned_gpu_ids),
                        name=f"JobRunner-GPU{','.join(map(str, assigned_gpu_ids))}-{mode_tag}-{run_job_id[:4]}",
                        daemon=True
                    )
                    job_runner_thread.start()
                else:
                    # This block is reached if stop_event interrupted the wait OR max_attempts was reached without success
                    if not self.stop_event.is_set(): # Only log re-queue if not shutting down
                        logger.warning(f"Exceeded max attempts ({self.max_assignment_attempts}) or failed to assign GPU(s) for job ID {job_id} (Req: {required_gpus:.2f}). Re-queueing job.")
                        
                        # Circuit breaker logic
                        with self.lock:
                            self.circuit_breaker_failures += 1
                            if self.circuit_breaker_failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                                self.circuit_breaker_timeout_until = time.time() + CIRCUIT_BREAKER_TIMEOUT
                                logger.error(f"Circuit breaker triggered after {self.circuit_breaker_failures} consecutive failures. Backing off for {CIRCUIT_BREAKER_TIMEOUT}s")
                        
                        try:
                            self.job_queue.put(job_tuple)
                        except Exception as e:
                            logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} after max attempts: {e}. Job may be lost.")
                    # If stop_event caused the loop break, the job was already re-queued inside the loop

                # Check stop event again before trying to get the next job
                if self.stop_event.is_set():
                    logger.info("Stop event detected after assignment processing. Exiting worker.")
                    break

            except queue.Empty:
                # Queue was empty during the timeout, normal operation
                got_task = False
                continue # Go back to waiting for a job
            except Exception as e:
                # Catch unexpected errors in the main worker loop
                logger.error(f"Error in worker main loop processing job ID {job_id or 'N/A'} (Hash: {job_hash or 'N/A'}): {e}", exc_info=True)
                # Ensure task_done is called even if there was an error processing the job
                got_task = True # We did get a task, even if it caused an error here
            finally:
                # Crucial: Mark the task as done *if* we successfully got one from the queue.
                if got_task:
                    try:
                        self.job_queue.task_done()
                        logger.debug(f"task_done() called for job ID {job_id} (Hash: {job_hash})")
                    except ValueError:
                        # This should not happen with PriorityQueue unless task_done() is called elsewhere
                        logger.error(f"CRITICAL: task_done() called too many times for job ID {job_id} (Hash: {job_hash})!")
                    except Exception as e:
                        logger.error(f"Error calling task_done() for job ID {job_id} (Hash: {job_hash}): {e}")

        logger.info(f"Worker finished.")

    # --- Job Execution ---
    def _run_job(self, job_tuple: Tuple, assigned_gpu_ids: List[int]):
            """
            Internal method called by a dedicated thread to prepare environment
            and execute a single job on its assigned GPU(s).
            Modifies '--cuda' argument to pass assigned IDs.
            Ensures the GPU allocation is released afterwards. Logs original line.
            Logs job start to running_jobs.log. # <<< MODIFIED >>>

            Args:
                job_tuple: The full job tuple including requirements and original_line.
                assigned_gpu_ids: The list of GPU IDs assigned by the worker.
            """
            # Unpack the full tuple including original_line
            (priority, job_id, script_path, conda_env, original_args, _,
             job_hash, required_gpus, llm_name, original_line) = job_tuple
            job_name = Path(script_path).name
            start_time = datetime.now(timezone.utc) # <<< MODIFIED >>> Use timezone-aware UTC time
            mode_tag = _extract_and_sanitize_key_arg(original_args) # Extract mode from original args

            if not assigned_gpu_ids:
                 logger.error(f"CRITICAL: _run_job called for job ID {job_id} with no assigned GPUs. Aborting execution.")
                 # Allocation should not have happened, but double-check release logic might be needed if this occurs
                 return

            # --- Prepare arguments for the actual execution ---
            args_no_cuda = [arg for i, arg in enumerate(original_args or [])
                            if not (arg == '--cuda' or (i > 0 and original_args[i-1] == '--cuda'))]
            cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
            args_for_exec = args_no_cuda + ["--cuda", cuda_arg_value]
            # -------------------------------------------------

            job_completed_successfully = False
            release_reason = "unknown"
            screen_session_name = None # <<< ADDED >>> Store screen session name if used

            # --- Log Job Start with Original Line ---
            start_log_msg = f"GPU(s) {cuda_arg_value}: Preparing job ID {job_id} (LLM: {llm_name}, Req: {required_gpus:.2f})"
            if original_line:
                start_log_msg += f" [Original Line: {original_line.strip()}]"
            logger.info(start_log_msg)
            # ----------------------------------------
            logger.debug(f"GPU(s) {cuda_arg_value}: Job ID {job_id} arguments for execution: {args_for_exec}")

            # Set CUDA_VISIBLE_DEVICES environment variable for the job
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = cuda_arg_value

            try:
                # Create exec tuple with modified args for passing to run methods
                # Pass the full original tuple to execution methods so they have original_line
                exec_job_tuple = job_tuple[:4] + (args_for_exec,) + job_tuple[5:]

                # <<< ADDED >>> Log job start to running_jobs.log *before* execution starts
                # This happens regardless of screen or direct mode
                self._log_job_start(job_id, job_hash, assigned_gpu_ids, start_time, original_line, screen_session_name=None) # Screen name logged later if used

                if self.use_screen:
                    # --- Screen Mode ---
                    # _run_with_screen now returns tuple: (launched_ok, session_name)
                    launched_ok, screen_session_name = self._run_with_screen(exec_job_tuple, assigned_gpu_ids, env, start_time, mode_tag)
                    if not launched_ok:
                        logger.error(f"GPU(s) {cuda_arg_value}: Screen setup failed for job ID {job_id} (Hash: {job_hash}).")
                        job_completed_successfully = False
                        release_reason = "screen setup failure"
                        # Release GPU immediately if screen launch failed
                        self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, original_line, success=job_completed_successfully, reason=release_reason)
                    # If launch OK, _monitor_screen will call _release_gpu later
                    # <<< ADDED >>> Log start again with screen name if successful
                    elif screen_session_name:
                         self._log_job_start(job_id, job_hash, assigned_gpu_ids, start_time, original_line, screen_session_name)
                    return # Exit _run_job thread, monitor thread takes over

                else:
                    # --- Direct Mode ---
                    job_completed_successfully = self._run_directly(exec_job_tuple, assigned_gpu_ids, env, start_time, mode_tag)
                    release_reason = f"direct execution completion (Success: {job_completed_successfully})"
                    # Release GPU after direct execution finishes
                    self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, original_line, success=job_completed_successfully, reason=release_reason)
                    return

            except Exception as e:
                logger.error(f"GPU(s) {cuda_arg_value}: CRITICAL Unexpected error in _run_job setup/dispatch for job ID {job_id} (Hash: {job_hash}): {e}", exc_info=True)
                job_completed_successfully = False
                release_reason = "critical error in _run_job setup"
                allocation_present = False
                # Check if allocation still exists before attempting release
                with self.lock:
                    for gpu_id in assigned_gpu_ids:
                         if 0 <= gpu_id < len(self.gpu_status) and self.gpu_status[gpu_id] > GPU_ALLOCATION_PRECISION:
                             allocation_present = True
                             break
                if allocation_present:
                    logger.warning(f"GPU(s) {cuda_arg_value}: Releasing allocation due to error during _run_job setup for job ID {job_id} (Hash: {job_hash}).")
                    self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, original_line, success=job_completed_successfully, reason=release_reason)
                else:
                    logger.warning(f"GPU(s) {cuda_arg_value}: Error during _run_job setup for job ID {job_id} (Hash: {job_hash}), but allocation seemed already released.")

    def _run_directly(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: dict, start_time: datetime, mode_tag: str) -> bool:
        """
        Executes a job directly (non-screen mode) and returns whether it completed successfully.
        
        Args:
            job_tuple: Job tuple containing all job information
            assigned_gpu_ids: List of assigned GPU IDs
            env: Environment variables including CUDA_VISIBLE_DEVICES
            start_time: Job start time
            mode_tag: Sanitized mode tag for logging
            
        Returns:
            bool: True if job completed successfully (exit code 0), False otherwise
        """
        (priority, job_id, script_path, conda_env, args_for_exec, _,
         job_hash, required_gpus, llm_name, original_line) = job_tuple
        
        cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
        
        try:
            # Build the command for direct execution
            if conda_env:
                # Use conda run for environment activation
                cmd = ['conda', 'run', '-n', conda_env, 'python3', script_path] + args_for_exec
            else:
                # Direct python execution
                cmd = ['python3', script_path] + args_for_exec
            
            logger.info(f"GPU(s) {cuda_arg_value}: Starting direct execution for job ID {job_id}")
            logger.debug(f"GPU(s) {cuda_arg_value}: Command: {' '.join(cmd)}")
            
            # Execute the job
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=False,  # Let output go to terminal/logs
                text=True,
                cwd=os.getcwd()  # Run in current working directory
            )
            
            # Check the exit code
            if result.returncode == 0:
                logger.info(f"GPU(s) {cuda_arg_value}: Job ID {job_id} completed successfully (exit code: 0)")
                return True
            else:
                logger.error(f"GPU(s) {cuda_arg_value}: Job ID {job_id} failed with exit code: {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"GPU(s) {cuda_arg_value}: Job ID {job_id} failed with CalledProcessError: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"GPU(s) {cuda_arg_value}: Job ID {job_id} failed - script or interpreter not found: {e}")
            return False
        except Exception as e:
            logger.error(f"GPU(s) {cuda_arg_value}: Job ID {job_id} failed with unexpected error: {e}", exc_info=True)
            return False

    def _run_with_screen(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: dict, start_time: datetime, mode_tag: str) -> Tuple[bool, Optional[str]]:
        """
        Runs a job using GNU Screen and returns launch success and session name.
        Creates organized output directories for experiments.
        """
        priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash, required_gpus, llm_name, original_line = job_tuple

        try:
            # Extract hyperparameters and generate experiment ID
            hyperparams = self._extract_hyperparameters(args)
            experiment_id = self._generate_experiment_id(script_path, hyperparams)
            
            # Create experiment-specific output directory
            experiment_dir = self.experiment_results_dir / experiment_id
            experiment_dir.mkdir(exist_ok=True)
            
            # Create job-specific output directory within experiment
            job_output_dir = experiment_dir / f"job_{job_id}_{mode_tag}"
            job_output_dir.mkdir(exist_ok=True)
            
            # Log experiment start
            self._log_experiment_start(experiment_id, job_id, hyperparams)

            gpu_id_str = ','.join(map(str, assigned_gpu_ids))
            session_name = f"gpujob_{gpu_id_str}_{job_id}"

            if not args:
                args = []

            args_copy = args.copy()

            # Replace or add --cuda argument
            cuda_arg_found = False
            for i, arg in enumerate(args_copy):
                if arg == '--cuda':
                    if i + 1 < len(args_copy):
                        args_copy[i + 1] = gpu_id_str
                    else:
                        args_copy.append(gpu_id_str)
                    cuda_arg_found = True
                    break

            if not cuda_arg_found:
                args_copy.extend(['--cuda', gpu_id_str])

            # Add output directory to arguments if the script supports it
            if '--output_dir' not in args_copy and '--output-dir' not in args_copy:
                args_copy.extend(['--output_dir', str(job_output_dir)])

            # Construct the full command
            if conda_env:
                cmd_parts = ['conda', 'run', '-n', conda_env, 'python', script_path] + args_copy
            else:
                cmd_parts = ['python', script_path] + args_copy

            cmd_str = shlex.join(cmd_parts)

            # Create comprehensive log file path
            log_file = job_output_dir / f"execution_{job_id}.log"

            # Use script command for proper TTY handling and comprehensive logging
            screen_cmd = [
                'screen', '-dmS', session_name, '-L', '-Logfile', str(log_file),
                'script', '-f', str(log_file.with_suffix('.script')), '-c', cmd_str
            ]

            logger.info(f"Starting job {job_id} in screen session '{session_name}' with command: {cmd_str}")
            logger.info(f"Experiment: {experiment_id}, Output directory: {job_output_dir}")
            logger.info(f"Log file: {log_file}")

            # Start the screen session
            result = subprocess.run(screen_cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Job {job_id} launched successfully in screen session '{session_name}'")
                
                # Start monitoring thread
                monitor_thread = threading.Thread(
                    target=self._monitor_screen,
                    args=(job_tuple, assigned_gpu_ids, session_name, start_time, mode_tag),
                    name=f"ScreenMonitor-{job_id}",
                    daemon=True
                )
                monitor_thread.start()
                
                return True, session_name
            else:
                logger.error(f"Failed to launch job {job_id} in screen: {result.stderr}")
                # Log experiment failure
                self._log_experiment_completion(experiment_id, job_id, False)
                return False, None

        except Exception as e:
            logger.error(f"Error running job {job_id} with screen: {e}")
            # Try to log experiment failure if we have the experiment_id
            try:
                if 'experiment_id' in locals():
                    self._log_experiment_completion(experiment_id, job_id, False)
            except:
                pass
            return False, None

    def _monitor_screen(self, job_tuple: Tuple, assigned_gpu_ids: List[int], session_name: str, start_time: datetime, mode_tag: str):
        """
        Monitors a screen session until completion and analyzes results.
        Detects output files and tracks experiment completion.
        """
        priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash, required_gpus, llm_name, original_line = job_tuple
        
        try:
            # Extract experiment information
            hyperparams = self._extract_hyperparameters(args)
            experiment_id = self._generate_experiment_id(script_path, hyperparams)
            job_output_dir = self.experiment_results_dir / experiment_id / f"job_{job_id}_{mode_tag}"
            
            logger.info(f"Monitoring screen session '{session_name}' for job {job_id}")
            
            # Monitor until session ends
            while True:
                time.sleep(10)  # Check every 10 seconds
                
                if self.stop_event.is_set():
                    logger.info(f"Stop event set, ending monitoring for job {job_id}")
                    break
                
                # Check if screen session still exists
                active_sessions = list_screens()
                if session_name not in active_sessions:
                    logger.info(f"Screen session '{session_name}' has ended for job {job_id}")
                    break
            
            # Session has ended, analyze results
            success = False
            reason = "Unknown"
            output_files = []
            
            try:
                # Look for log files in the job output directory
                log_files = []
                if job_output_dir.exists():
                    log_files.extend(list(job_output_dir.glob("*.log")))
                    log_files.extend(list(job_output_dir.glob("*.script")))
                    
                    # Also look for common output files
                    output_patterns = ["*.json", "*.csv", "*.pkl", "*.pt", "*.pth", "*.txt", "*.xlsx"]
                    for pattern in output_patterns:
                        output_files.extend([str(f) for f in job_output_dir.glob(pattern)])
                
                # Analyze the main execution log for success/failure
                execution_log = job_output_dir / f"execution_{job_id}.log"
                if execution_log.exists():
                    log_content = execution_log.read_text()
                    
                    # Check for Python errors (same logic as before but more comprehensive)
                    error_patterns = [
                        r"Traceback \(most recent call last\):",
                        r"^\s*\w*Error:",
                        r"^\s*Exception:",
                        r"CUDA out of memory",
                        r"RuntimeError:",
                        r"ValueError:",
                        r"KeyError:",
                        r"ImportError:",
                        r"ModuleNotFoundError:",
                        r"Process finished with exit code [1-9]",
                        r"FAILED",
                        r"ERROR:"
                    ]
                    
                    has_errors = any(re.search(pattern, log_content, re.MULTILINE | re.IGNORECASE) 
                                   for pattern in error_patterns)
                    
                    # Check for success indicators
                    success_patterns = [
                        r"Process finished with exit code 0",
                        r"COMPLETED SUCCESSFULLY",
                        r"Evaluation completed",
                        r"Training completed",
                        r"All experiments completed",
                        r"Results saved to",
                        r"SUCCESS"
                    ]
                    
                    has_success = any(re.search(pattern, log_content, re.MULTILINE | re.IGNORECASE) 
                                    for pattern in success_patterns)
                    
                    if has_errors:
                        success = False
                        reason = "Python error detected in logs"
                    elif has_success:
                        success = True
                        reason = "Completed successfully"
                    elif output_files:
                        success = True
                        reason = f"Generated {len(output_files)} output files"
                    else:
                        success = False
                        reason = "No success indicators or output files found"
                else:
                    # Fallback to legacy log location
                    legacy_log = SCREEN_LOG_DIR / f"{session_name}.log"
                    if legacy_log.exists():
                        log_content = legacy_log.read_text()
                        has_errors = any(re.search(pattern, log_content, re.MULTILINE | re.IGNORECASE) 
                                       for pattern in error_patterns)
                        success = not has_errors
                        reason = "Error detected in legacy log" if has_errors else "No errors in legacy log"
                    else:
                        success = False
                        reason = "No log files found"
                        
            except Exception as e:
                logger.error(f"Error analyzing results for job {job_id}: {e}")
                success = False
                reason = f"Error analyzing results: {e}"
            
            # Release GPU and update metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            job_name = Path(script_path).stem
            
            # Log experiment completion with output files
            self._log_experiment_completion(experiment_id, job_id, success, output_files)
            
            self._release_gpu(
                assigned_gpu_ids, required_gpus, job_id, job_name, 
                start_time, mode_tag, job_hash, original_line, success, reason
            )
            
            # Log completion
            self._log_job_completion(
                job_id, job_hash, assigned_gpu_ids, start_time, end_time, 
                duration, success, reason, original_line
            )
            
            logger.info(f"Job {job_id} monitoring completed. Success: {success}, Reason: {reason}")
            if output_files:
                logger.info(f"Output files generated: {output_files}")
                
        except Exception as e:
            logger.error(f"Error in screen monitoring for job {job_id}: {e}", exc_info=True)
            # Ensure GPU is released even on error
            try:
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()
                job_name = Path(script_path).stem
                
                self._release_gpu(
                    assigned_gpu_ids, required_gpus, job_id, job_name,
                    start_time, mode_tag, job_hash, original_line, False, f"Monitoring error: {e}"
                )
                
                # Try to log experiment failure
                try:
                    hyperparams = self._extract_hyperparameters(args)
                    experiment_id = self._generate_experiment_id(script_path, hyperparams)
                    self._log_experiment_completion(experiment_id, job_id, False)
                except:
                    pass
                    
            except Exception as cleanup_error:
                logger.error(f"Error in cleanup for job {job_id}: {cleanup_error}")

    def _log_job_start(self, job_id: str, job_hash: str, assigned_gpu_ids: List[int], 
                      start_time: datetime, original_line: Optional[str], screen_session_name: Optional[str] = None):
        """
        Logs job start information to the running_jobs.log file.
        
        Args:
            job_id: Unique job identifier
            job_hash: Job hash for duplicate detection
            assigned_gpu_ids: List of assigned GPU IDs
            start_time: Job start timestamp
            original_line: Original job line from jobs file (if applicable)
            screen_session_name: Screen session name (if using screen mode)
        """
        try:
            cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
            
            # Create log entry
            log_entry = {
                "timestamp": start_time.isoformat(),
                "job_id": job_id,
                "job_hash": job_hash,
                "assigned_gpus": assigned_gpu_ids,
                "cuda_visible_devices": cuda_arg_value,
                "screen_session": screen_session_name,
                "original_line": original_line.strip() if original_line else None,
                "status": "STARTED"
            }
            
            # Write to running jobs log file
            with self.job_log_lock:
                with open(RUNNING_JOBS_LOG_FILE, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    
            logger.debug(f"Logged job start for ID {job_id} to {RUNNING_JOBS_LOG_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to log job start for ID {job_id}: {e}", exc_info=True)

    def _release_gpu(self, assigned_gpu_ids: List[int], required_gpus: float, job_id: str, 
                    job_name: str, start_time: datetime, mode_tag: str, job_hash: str, 
                    original_line: Optional[str], success: bool, reason: str):
        """
        Releases GPU allocation and logs job completion.
        
        Args:
            assigned_gpu_ids: List of GPU IDs to release
            required_gpus: GPU requirement that was allocated
            job_id: Unique job identifier
            job_name: Name of the job script
            start_time: Job start timestamp
            mode_tag: Sanitized mode tag
            job_hash: Job hash for tracking
            original_line: Original job line from jobs file
            success: Whether the job completed successfully
            reason: Reason for release (completion, failure, etc.)
        """
        cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        try:
            # Release GPU allocation
            with self.lock:
                allocation_to_release = 0.0
                if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION):
                    # Fractional or single GPU release
                    allocation_to_release = required_gpus
                    for gpu_id in assigned_gpu_ids:
                        if 0 <= gpu_id < len(self.gpu_status):
                            self.gpu_status[gpu_id] = max(0.0, self.gpu_status[gpu_id] - allocation_to_release)
                else:
                    # Multi-GPU release - set to 0.0 (free)
                    allocation_to_release = 1.0
                    for gpu_id in assigned_gpu_ids:
                        if 0 <= gpu_id < len(self.gpu_status):
                            self.gpu_status[gpu_id] = 0.0
                
                # Save state after releasing allocation
                try:
                    self.save_state()
                except Exception as save_error:
                    logger.error(f"CRITICAL: Failed to save state after releasing GPU(s) {cuda_arg_value} for job {job_id}: {save_error}")
            
            # Log job completion
            self._log_job_completion(job_id, job_hash, assigned_gpu_ids, start_time, end_time, 
                                   duration, success, reason, original_line)
            
            # Update performance metrics
            self._update_performance_metrics('job_completed', success=success, 
                                           duration_seconds=duration, gpu_requirement=required_gpus)
            
            status_msg = "COMPLETED SUCCESSFULLY" if success else "FAILED"
            logger.info(f"GPU(s) {cuda_arg_value}: Released allocation for job ID {job_id} - {status_msg} "
                       f"(Duration: {duration:.1f}s, Reason: {reason})")
            
        except Exception as e:
            logger.error(f"Error releasing GPU allocation for job ID {job_id}: {e}", exc_info=True)

    def _log_job_completion(self, job_id: str, job_hash: str, assigned_gpu_ids: List[int], 
                           start_time: datetime, end_time: datetime, duration: float, 
                           success: bool, reason: str, original_line: Optional[str]):
        """
        Logs job completion information to the done_jobs.log file.
        
        Args:
            job_id: Unique job identifier
            job_hash: Job hash for duplicate detection
            assigned_gpu_ids: List of assigned GPU IDs
            start_time: Job start timestamp
            end_time: Job end timestamp
            duration: Job duration in seconds
            success: Whether the job completed successfully
            reason: Reason for completion/failure
            original_line: Original job line from jobs file (if applicable)
        """
        try:
            cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
            
            # Create completion log entry
            log_entry = {
                "timestamp": end_time.isoformat(),
                "job_id": job_id,
                "job_hash": job_hash,
                "assigned_gpus": assigned_gpu_ids,
                "cuda_visible_devices": cuda_arg_value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "success": success,
                "reason": reason,
                "original_line": original_line.strip() if original_line else None,
                "status": "COMPLETED"
            }
            
            # Write to done jobs log file
            with self.job_log_lock:
                with open(DONE_JOBS_LOG_FILE, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Logged job completion for ID {job_id} to {DONE_JOBS_LOG_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to log job completion for ID {job_id}: {e}", exc_info=True)

    def add_job(self, script_path: str, conda_env: Optional[str] = None, 
               args: Optional[List[str]] = None, priority: int = 0, 
               allowed_gpus: Optional[List[int]] = None, original_line: Optional[str] = None) -> bool:
        """
        Adds a single job to the queue with LLM requirement calculation and duplicate detection.
        
        Args:
            script_path: Path to the Python script to execute
            conda_env: Optional conda environment name
            args: Optional list of arguments for the script
            priority: Job priority (lower number = higher priority)
            allowed_gpus: Optional list of allowed GPU IDs
            original_line: Original job line from file (for logging/tracking)
            
        Returns:
            bool: True if job was added, False if it was a duplicate or failed validation
        """
        try:
            # Validate script path
            if not script_path or not Path(script_path).exists():
                logger.error(f"Script path does not exist: {script_path}")
                return False
            
            # Extract LLM name and calculate GPU requirement
            llm_name = _extract_arg_value(args, '--llm') if args else None
            required_gpus = self.get_llm_gpu_requirement(llm_name)
            
            # Validate allowed GPUs
            if allowed_gpus:
                validated_gpus = []
                for gpu_id in allowed_gpus:
                    if 0 <= gpu_id < self.num_gpus:
                        validated_gpus.append(gpu_id)
                    else:
                        logger.warning(f"Invalid GPU ID {gpu_id} for job {script_path}. Must be 0-{self.num_gpus-1}")
                allowed_gpus = validated_gpus if validated_gpus else None
            
            # Calculate job hash for duplicate detection
            job_hash = self._calculate_job_hash(priority, script_path, conda_env, args, allowed_gpus, required_gpus)
            
            # Check for duplicates
            with self.lock:
                if job_hash in self.managed_job_hashes:
                    logger.debug(f"Duplicate job detected (Hash: {job_hash}): {script_path}")
                    return False
                
                # Cleanup hashes if needed
                if len(self.managed_job_hashes) >= MAX_MANAGED_HASHES * 0.9:
                    self._cleanup_managed_hashes()
                
                # Generate unique job ID
                job_id = str(uuid.uuid4())
                
                # Create job tuple
                job_tuple = (
                    priority,           # 0: Priority for queue ordering
                    job_id,            # 1: Unique job identifier
                    script_path,       # 2: Script path
                    conda_env,         # 3: Conda environment
                    args,              # 4: Script arguments
                    allowed_gpus,      # 5: Allowed GPU list
                    job_hash,          # 6: Job hash for duplicate detection
                    required_gpus,     # 7: GPU requirement (float)
                    llm_name,          # 8: LLM name (if specified)
                    original_line      # 9: Original line from file
                )
                
                # Add to queue
                self.job_queue.put(job_tuple)
                
                # Track as managed
                self.managed_job_hashes.add(job_hash)
                self.hash_to_job_id[job_hash] = job_id
                
                # Update queue size metrics
                self._update_performance_metrics('queue_size_update', size=self.job_queue.qsize())
            
            logger.info(f"Added job ID {job_id} to queue: {script_path} "
                       f"(Priority: {priority}, LLM: {llm_name}, GPU Req: {required_gpus:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add job {script_path}: {e}", exc_info=True)
            return False

    def add_jobs_from_file(self, file_path: str, initial_load: bool = False):
        """Adds multiple jobs from a file, parsing LLM requirements."""
        log_prefix = "[File Monitor]" if not initial_load else "[Initial Load]"
        logger.info(f"{log_prefix} Attempting to process jobs from file: {file_path}")
        jobs_processed = 0
        jobs_added = 0 # Track newly added jobs this scan
        file_p = Path(file_path)

        if not file_p.is_file():
            if initial_load: logger.error(f"{log_prefix} Job file not found: {file_path}.")
            else: logger.debug(f"{log_prefix} Job file not found: {file_path}.")
            return

        try:
            with open(file_p, 'r') as f: current_lines = f.readlines()

            hashes_in_current_scan = set()
            for i, line in enumerate(current_lines):
                line_num = i + 1
                original_line_content = line # Keep original for logging/hashing context
                line_strip = line.strip() # Use stripped version for checks
                if not line_strip or line_strip.startswith('#'): continue

                jobs_processed += 1
                try:
                    # Parse line parts
                    # Format: priority,script_path[,conda_env[,arguments[,allowed_gpus]]]
                    # Split only up to 4 commas to keep allowed_gpus together if it exists
                    parts = [p.strip() for p in line_strip.split(',', maxsplit=4)]
                    if len(parts) < 2 or not parts[0] or not parts[1]:
                        logger.error(f"{log_prefix} Invalid format line {line_num}: '{line_strip}'. Expected 'priority,script,...'.")
                        continue

                    priority = int(parts[0])
                    script = parts[1]
                    conda_env = parts[2] if len(parts) > 2 and parts[2] else None
                    args_str = parts[3] if len(parts) > 3 and parts[3] else None
                    allowed_gpus_str = parts[4] if len(parts) > 4 and parts[4] else None

                    # Parse args and get LLM requirement
                    args_list = shlex.split(args_str) if args_str else None
                    llm_name = _extract_arg_value(args_list, '--llm')
                    required_gpus = self.get_llm_gpu_requirement(llm_name)
                    if args_list and '--cuda' in args_list:
                         logger.warning(f"{log_prefix} Found '--cuda' in args line {line_num}. It will be ignored/overwritten by the scheduler.")

                    # Parse allowed GPUs (supports ranges like 0-3,5)
                    allowed_gpus_list = None
                    if allowed_gpus_str:
                        allowed_gpus_list = []
                        for part in allowed_gpus_str.replace(" ", "").split(','):
                             part = part.strip()
                             if not part: continue
                             if '-' in part:
                                 try:
                                     start_gpu, end_gpu = map(int, part.split('-'))
                                     if start_gpu <= end_gpu: allowed_gpus_list.extend(range(start_gpu, end_gpu + 1))
                                     else: logger.warning(f"{log_prefix} Invalid range '{part}' line {line_num}, start > end.")
                                 except ValueError: logger.warning(f"{log_prefix} Invalid range format '{part}' line {line_num}")
                             else:
                                 try: allowed_gpus_list.append(int(part.strip()))
                                 except ValueError: logger.warning(f"{log_prefix} Invalid GPU ID '{part}' line {line_num}")
                    # Validation of allowed_gpus_list against self.num_gpus happens inside add_job

                    # Calculate hash including requirement
                    current_job_hash = self._calculate_job_hash(priority, script, conda_env, args_list, allowed_gpus_list, required_gpus)
                    hashes_in_current_scan.add(current_job_hash)

                    # Check if managed and add if not
                    is_managed = False
                    with self.lock: is_managed = current_job_hash in self.managed_job_hashes

                    if not is_managed:
                        # Pass the original_line_content to add_job
                        self.add_job(script, conda_env, args_list, priority, allowed_gpus_list, original_line=original_line_content)
                        with self.lock: # Check if add_job succeeded in adding the hash
                            if current_job_hash in self.managed_job_hashes: jobs_added += 1
                    else:
                        logger.debug(f"{log_prefix} Job line {line_num} (Hash: {current_job_hash}) already managed. Skipping.")

                except ValueError as ve: logger.error(f"{log_prefix} Invalid number format line {line_num}: '{line_strip}'. {ve}")
                except Exception as e: logger.error(f"{log_prefix} Error parsing line {line_num}: '{line_strip}'. {e}", exc_info=True)

            log_message = f"{log_prefix} Finished processing '{file_path}'. Processed {jobs_processed} lines."
            log_message += f" Added {jobs_added} new jobs."
            logger.info(log_message)

        except FileNotFoundError: logger.error(f"{log_prefix} File disappeared: {file_path}")
        except Exception as e: logger.error(f"{log_prefix} Failed processing {file_path}: {e}", exc_info=True)


    # --- Combined File/State Monitoring ---
    def _monitor_files(self):
        """
        Periodically scans the jobs file (if configured) and adds new/unmanaged jobs.
        Also periodically checks the state file for external changes to paused GPUs.
        """
        monitor_log_prefix = "[File/State Monitor]"
        jobs_file_log_msg = f"'{self.jobs_file_path}' (Interval: {self.file_monitor_interval}s)" if self.jobs_file_path else "disabled"
        state_file_log_msg = f"'{self.state_file}' (Interval: {self.state_check_interval}s)"

        logger.info(f"{monitor_log_prefix} Starting. Job file monitoring: {jobs_file_log_msg}. State file monitoring: {state_file_log_msg}")

        # Perform initial load from the jobs file if configured
        if self.jobs_file_path:
            logger.info(f"{monitor_log_prefix} Performing initial load from job file: {self.jobs_file_path}")
            self.add_jobs_from_file(str(self.jobs_file_path), initial_load=True)

        last_job_check_time = time.monotonic()
        last_state_check_time = time.monotonic()

        while not self.stop_event.is_set():
            current_time = time.monotonic()

            # Check jobs file
            if self.jobs_file_path and (current_time - last_job_check_time >= self.file_monitor_interval):
                try:
                    logger.debug(f"{monitor_log_prefix} Checking jobs file: {self.jobs_file_path}")
                    self.add_jobs_from_file(str(self.jobs_file_path), initial_load=False)
                except Exception as e:
                    logger.error(f"{monitor_log_prefix} Error during check of {self.jobs_file_path}: {e}", exc_info=True)
                finally:
                    last_job_check_time = current_time # Update time even on error

            # Check state file for paused GPU changes
            if current_time - last_state_check_time >= self.state_check_interval:
                try:
                    logger.debug(f"{monitor_log_prefix} Checking state file for paused GPU changes: {self.state_file}")
                    self._check_and_apply_external_state_changes()
                except Exception as e:
                    logger.error(f"{monitor_log_prefix} Error during check of {self.state_file}: {e}", exc_info=True)
                finally:
                    last_state_check_time = current_time # Update time even on error

            # Determine next wakeup time
            next_job_wakeup = last_job_check_time + self.file_monitor_interval if self.jobs_file_path else float('inf')
            next_state_wakeup = last_state_check_time + self.state_check_interval
            next_wakeup = min(next_job_wakeup, next_state_wakeup)

            wait_time = max(0, next_wakeup - time.monotonic())

            # Wait until next check or stop event
            if self.stop_event.wait(timeout=wait_time):
                break # Exit loop if stop event is set

        logger.info(f"{monitor_log_prefix} Stopped.")


    # --- Start/Stop Modifications ---
    def enable_screen(self):
       """Checks if GNU Screen and the `script` command are available and enables screen use."""
       screen_ok = False
       script_ok = False
       try:
           subprocess.run(['screen', '-v'], check=True, capture_output=True, text=True)
           screen_ok = True
       except Exception as e:
           logger.error(f"Screen check failed: {e}. Screen functionality disabled.")

       try:
           subprocess.run(['script', '--version'], check=True, capture_output=True, text=True)
           script_ok = True
       except Exception as e:
           logger.error(f"`script` command check failed: {e}. Screen functionality disabled (required for TTY preservation).")

       if screen_ok and script_ok:
           self.use_screen = True
           logger.info("Screen functionality enabled (using `script` for logging).")
       else:
           self.use_screen = False
           logger.error("Screen functionality disabled due to missing `screen` or `script` command.")

    def _cleanup_managed_hashes(self):
        """Cleanup old job hashes to prevent memory leaks. Keeps only the most recent hashes."""
        with self.lock:
            if len(self.managed_job_hashes) > MAX_MANAGED_HASHES:
                # Convert to list to get a deterministic order, then keep only the most recent ones
                # Since we can't easily determine recency without additional metadata,
                # we'll use a simple approach: clear older hashes periodically
                excess_count = len(self.managed_job_hashes) - (MAX_MANAGED_HASHES // 2)
                hashes_list = list(self.managed_job_hashes)
                
                # Remove first N hashes (assuming older ones are processed first)
                for i in range(min(excess_count, len(hashes_list) // 2)):
                    hash_to_remove = hashes_list[i]
                    self.managed_job_hashes.discard(hash_to_remove)
                    # Also clean up the hash_to_job_id mapping
                    self.hash_to_job_id.pop(hash_to_remove, None)
                
                logger.info(f"Cleaned up {excess_count} old job hashes. Current count: {len(self.managed_job_hashes)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns a health status dictionary for monitoring purposes."""
        try:
            with self.lock:
                queue_size = self.job_queue.qsize()
                managed_hashes_count = len(self.managed_job_hashes)
                paused_gpus_count = len(self.paused_gpus)
                
            active_workers = sum(1 for t in self.worker_threads if t.is_alive())
            monitor_alive = self.monitor_thread and self.monitor_thread.is_alive()
            
            # Check if any GPUs are available
            available_gpus = 0
            with self.lock:
                for i, allocation in enumerate(self.gpu_status):
                    if i not in self.paused_gpus and allocation < (1.0 - GPU_ALLOCATION_PRECISION):
                        available_gpus += 1
            
            # Calculate basic health score
            health_issues = []
            if active_workers == 0:
                health_issues.append("No active worker threads")
            if not monitor_alive:
                health_issues.append("Monitor thread not running")
            if available_gpus == 0:
                health_issues.append("No available GPUs")
            if managed_hashes_count >= MAX_MANAGED_HASHES * 0.9:
                health_issues.append("Hash table nearly full")
                
            status = "healthy" if not health_issues else "degraded" if len(health_issues) <= 2 else "unhealthy"
            
            return {
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queue_size": queue_size,
                "active_workers": active_workers,
                "monitor_thread_alive": monitor_alive,
                "available_gpus": available_gpus,
                "paused_gpus": paused_gpus_count,
                "managed_hashes_count": managed_hashes_count,
                "health_issues": health_issues,
                "stop_event_set": self.stop_event.is_set()
            }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "health_issues": [f"Error getting health status: {e}"]
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Returns current performance metrics with additional experiment statistics.
        """
        with self.metrics_lock:
            metrics = self.performance_metrics.copy()
            
        # Calculate additional metrics
        current_time = time.time()
        uptime = current_time - metrics['start_time']
        metrics['uptime_hours'] = uptime / 3600
        
        if metrics['jobs_processed'] > 0:
            metrics['success_rate'] = metrics['jobs_successful'] / metrics['jobs_processed']
            metrics['failure_rate'] = metrics['jobs_failed'] / metrics['jobs_processed']
        else:
            metrics['success_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            
        # Add experiment statistics
        with self.experiment_lock:
            metrics['active_experiments'] = len(self.experiment_groups)
            total_experiment_jobs = sum(len(jobs) for jobs in self.experiment_groups.values())
            metrics['total_experiment_jobs'] = total_experiment_jobs
            
        return metrics

    def generate_parameter_sweep_jobs(self, script_path: str, conda_env: str,
                                    parameter_grid: Dict[str, List[str]], 
                                    base_args: Optional[List[str]] = None,
                                    priority: int = 5) -> List[str]:
        """
        Generates job lines for a parameter sweep experiment.
        
        Args:
            script_path: Path to the experiment script
            conda_env: Conda environment name
            parameter_grid: Dictionary of parameter names to lists of values
            base_args: Base arguments to include in all jobs
            priority: Job priority
            
        Returns:
            List of job line strings ready for the jobs file
        """
        import itertools
        
        if base_args is None:
            base_args = []
            
        job_lines = []
        
        # Generate all combinations of parameters
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for combination in itertools.product(*param_values):
            # Build arguments for this combination
            args = base_args.copy()
            
            # Add parameter values
            for param_name, param_value in zip(param_names, combination):
                args.extend([param_name, str(param_value)])
            
            # Create job line
            args_str = ' '.join(args)
            job_line = f"{priority},{script_path},{conda_env},{args_str}"
            job_lines.append(job_line)
            
        logger.info(f"Generated {len(job_lines)} jobs for parameter sweep")
        return job_lines

    def generate_model_comparison_jobs(self, script_path: str, conda_env: str,
                                     models: List[str], datasets: List[str],
                                     modes: List[str], base_args: Optional[List[str]] = None,
                                     priorities: Optional[Dict[str, int]] = None) -> List[str]:
        """
        Generates jobs for comparing multiple models across datasets and modes.
        Common pattern in ML research.
        
        Args:
            script_path: Path to the experiment script
            conda_env: Conda environment name
            models: List of model names
            datasets: List of dataset names  
            modes: List of evaluation modes
            base_args: Base arguments to include in all jobs
            priorities: Optional priority mapping for models
            
        Returns:
            List of job line strings
        """
        if base_args is None:
            base_args = []
        if priorities is None:
            priorities = {}
            
        job_lines = []
        
        for model in models:
            for dataset in datasets:
                for mode in modes:
                    args = base_args.copy()
                    args.extend(['--llm', model])
                    args.extend(['--dataset', dataset])
                    args.extend(['--mode', mode])
                    args.extend(['--cuda', '0'])  # Will be replaced by scheduler
                    
                    # Get priority for this model
                    priority = priorities.get(model, 5)
                    
                    args_str = ' '.join(args)
                    job_line = f"{priority},{script_path},{conda_env},{args_str}"
                    job_lines.append(job_line)
                    
        logger.info(f"Generated {len(job_lines)} jobs for model comparison")
        return job_lines

    def save_jobs_to_file(self, job_lines: List[str], filename: str, 
                         append: bool = False, add_comments: bool = True) -> bool:
        """
        Saves generated job lines to a file with optional organization.
        
        Args:
            job_lines: List of job line strings
            filename: Output filename
            append: Whether to append to existing file
            add_comments: Whether to add organizational comments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mode = 'a' if append else 'w'
            with open(filename, mode) as f:
                if add_comments:
                    f.write(f"\n# Generated jobs - {datetime.now().isoformat()}\n")
                    f.write(f"# Total jobs: {len(job_lines)}\n\n")
                
                for job_line in job_lines:
                    f.write(job_line + '\n')
                    
                if add_comments:
                    f.write("\n# End generated jobs\n\n")
                    
            logger.info(f"Saved {len(job_lines)} jobs to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving jobs to {filename}: {e}")
            return False

    def estimate_job_duration(self, script_path: str, hyperparams: Dict[str, str]) -> Optional[float]:
        """
        Estimates job duration based on historical data and hyperparameters.
        Useful for scheduling optimization.
        
        Args:
            script_path: Path to the script
            hyperparams: Job hyperparameters
            
        Returns:
            Estimated duration in seconds, or None if no estimate available
        """
        try:
            # Load historical experiment data
            if not self.experiment_log_file.exists():
                return None
                
            script_name = Path(script_path).stem
            durations = []
            
            with open(self.experiment_log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if (entry.get('event') == 'experiment_completion' and 
                            entry.get('success') and
                            script_name in entry.get('experiment_id', '')):
                            
                            # Try to find matching completion entry with duration
                            # This is a simplified approach - in practice you'd want
                            # more sophisticated matching based on hyperparameters
                            durations.append(3600)  # Default 1 hour estimate
                            
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
            if durations:
                # Return median duration as estimate
                durations.sort()
                return durations[len(durations) // 2]
                
        except Exception as e:
            logger.warning(f"Error estimating duration for {script_path}: {e}")
            
        return None

    def _update_performance_metrics(self, metric_type: str, **kwargs):
        """Internal method to update performance metrics safely."""
        try:
            with self.metrics_lock:
                if metric_type == 'job_completed':
                    self.performance_metrics['jobs_processed'] += 1
                    if kwargs.get('success', False):
                        self.performance_metrics['jobs_successful'] += 1
                    else:
                        self.performance_metrics['jobs_failed'] += 1
                    
                    # Update GPU hours (duration in hours * GPU requirement)
                    duration_hours = kwargs.get('duration_seconds', 0) / 3600.0
                    gpu_requirement = kwargs.get('gpu_requirement', 0.0)
                    self.performance_metrics['total_gpu_hours_allocated'] += duration_hours * gpu_requirement
                    
                elif metric_type == 'assignment_failure':
                    self.performance_metrics['assignment_failures'] += 1
                    
                elif metric_type == 'queue_size_update':
                    current_size = kwargs.get('size', 0)
                    if current_size > self.performance_metrics['peak_queue_size']:
                        self.performance_metrics['peak_queue_size'] = current_size
                        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")


# --- Command Line Interface ---
# Helper function for CLI pause/resume to interact with the state file directly
def _modify_state_file_pause_status(state_file_path: str, num_gpus_expected: int, gpu_id: int, pause: bool):
    """Loads state file, modifies paused_gpus list, and saves it back."""
    state_file = Path(state_file_path)
    action = "pause" if pause else "resume"
    logger_cli = logging.getLogger("GPUJobSchedulerCLI") # Use separate logger for CLI actions

    # 1. Load current state from file
    current_paused_gpus = set()
    current_gpu_status = [0.0] * num_gpus_expected # Default if file missing
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            # Load status, handling size mismatches like in main scheduler
            loaded_status = state.get('gpu_status', [])
            if len(loaded_status) == num_gpus_expected:
                current_gpu_status = [max(0.0, min(1.0, float(s))) for s in loaded_status]
            elif len(loaded_status) < num_gpus_expected:
                valid_loaded = [max(0.0, min(1.0, float(s))) for s in loaded_status]
                current_gpu_status = valid_loaded + [0.0] * (num_gpus_expected - len(loaded_status))
            else: # len(loaded_status) > num_gpus_expected
                current_gpu_status = [max(0.0, min(1.0, float(s))) for s in loaded_status[:num_gpus_expected]]

            # Load paused list
            loaded_paused = state.get('paused_gpus', [])
            current_paused_gpus = set(gid for gid in loaded_paused if 0 <= gid < num_gpus_expected)
            logger_cli.debug(f"Loaded state for {action}: Status Length={len(current_gpu_status)}, Paused={current_paused_gpus}")
        except Exception as e:
            logger_cli.error(f"Failed to load state from '{state_file}' for {action}: {e}. Cannot proceed.")
            return False
    else:
        logger_cli.info(f"State file '{state_file}' not found. Assuming all GPUs available/unpaused.")

    # 2. Modify the paused set
    original_paused_count = len(current_paused_gpus)
    if pause:
        if gpu_id in current_paused_gpus:
            logger_cli.warning(f"GPU {gpu_id} is already marked as paused in '{state_file}'.")
            return True # No change needed
        current_paused_gpus.add(gpu_id)
    else: # resume
        if gpu_id not in current_paused_gpus:
            logger_cli.warning(f"GPU {gpu_id} was not marked as paused in '{state_file}'.")
            return True # No change needed
        current_paused_gpus.remove(gpu_id)

    # 3. Save the modified state back to the file
    try:
        new_state = {
            "gpu_status": [round(s, 4) for s in current_gpu_status], # Keep existing status
            "paused_gpus": sorted(list(current_paused_gpus)) # Save updated paused list
        }
        # Atomic write
        temp_state_file = state_file.with_suffix(f".tmp_cli_{os.getpid()}")
        with open(temp_state_file, 'w') as f:
            json.dump(new_state, f, indent=4)
        os.replace(temp_state_file, state_file)
        logger_cli.info(f"State file '{state_file}' updated successfully to {action} GPU {gpu_id}.")
        return True
    except Exception as e:
        logger_cli.error(f"Failed to save updated state to '{state_file}' for {action}: {e}")
        # Clean up temp file on error
        if temp_state_file.exists():
            try: temp_state_file.unlink()
            except OSError: pass
        return False


def main():
    # Detect GPUs early for defaults
    detected_gpus = 0
    try:
        gpus_list = GPUtil.getGPUs(); detected_gpus = len(gpus_list) if gpus_list else 0
    except Exception as e:
        logger.warning(f"GPUtil detection failed: {e}. Defaulting --gpus may be inaccurate.")
        detected_gpus = 1 # Fallback, maybe 8 is better?

    parser = argparse.ArgumentParser(description='GPU Job Scheduler v2.13 (Job Logging)', formatter_class=argparse.RawTextHelpFormatter) # <<< MODIFIED >>> Version bump
    parser.add_argument('--state-file', type=str, default=DEFAULT_STATE_FILE, help=f'Path to scheduler state file (default: {DEFAULT_STATE_FILE}).')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- Start Command ---
    start_parser = subparsers.add_parser('start', help='Start the scheduler daemon process.')
    start_parser.add_argument('--gpus', type=int, default=detected_gpus, help=f'Number of GPUs to manage (default: {detected_gpus} detected)')
    start_parser.add_argument('--jobs-file', type=str, default=DEFAULT_JOBS_FILE, help=f'Path to jobs file for initial load/monitoring (default: {DEFAULT_JOBS_FILE}). Pass empty string "" to disable.')
    start_parser.add_argument('--llm-config', type=str, default=DEFAULT_LLM_CONFIG_FILE, help=f'Path to LLM requirements JSON file (default: {DEFAULT_LLM_CONFIG_FILE}).')
    # start_parser.add_argument('--no-monitor', action='store_true', help='DEPRECATED: Pass --jobs-file "" to disable monitoring.') # Kept for compatibility but hidden
    start_parser.add_argument('--screen', action='store_true', help='Enable GNU Screen sessions for job execution (requires `script` command).')
    start_parser.add_argument('--mem-threshold', type=float, default=0.8, help='GPU memory utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--load-threshold', type=float, default=0.8, help='GPU load utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--monitor-interval', type=int, default=FILE_MONITOR_INTERVAL_S, help=f'Interval (s) to check jobs file (default: {FILE_MONITOR_INTERVAL_S})')
    start_parser.add_argument('--state-interval', type=int, default=STATE_CHECK_INTERVAL_S, help=f'Interval (s) to check state file for pause changes (default: {STATE_CHECK_INTERVAL_S})')
    start_parser.add_argument('--max-assign-attempts', type=int, default=MAX_ASSIGNMENT_ATTEMPTS, help=f'Max attempts to assign job (default: {MAX_ASSIGNMENT_ATTEMPTS})')
    start_parser.add_argument('--assign-retry-wait', type=int, default=ASSIGNMENT_RETRY_WAIT_S, help=f'Base wait (s) between assignment attempts (default: {ASSIGNMENT_RETRY_WAIT_S})')

    # --- Add Command (Manual addition - requires running scheduler to pick up via IPC/file monitor) ---
    # Note: This command doesn't directly interact with a running scheduler.
    # It's mainly useful for adding lines to the jobs file if monitoring is enabled.
    add_parser = subparsers.add_parser('add', help='Helper to format a job line for the jobs file.')
    add_parser.add_argument('script', type=str, help='Path to the Python script.')
    add_parser.add_argument('--conda', type=str, help='Conda environment name (optional).')
    add_parser.add_argument('--args', type=str, help='Script arguments (quote if needed). Include --llm <n> if applicable. DO NOT include --cuda.')
    add_parser.add_argument('--priority', type=int, default=0, help='Job priority (lower=higher, default: 0).')
    add_parser.add_argument('--gpus', type=str, help='Allowed GPU IDs (e.g., "0,1", "0-3,5", optional).')
    add_parser.add_argument('--output-file', type=str, default=DEFAULT_JOBS_FILE, help=f'Append the formatted job line to this file (default: {DEFAULT_JOBS_FILE}). Set to "-" to print to stdout.')

    # --- Other Commands ---
    status_parser = subparsers.add_parser('status', help='Show status based on the state file (may lag behind running scheduler).')
    # status_parser inherits --state-file from main parser

    pause_parser = subparsers.add_parser('pause', help='Pause a GPU by modifying the state file.')
    pause_parser.add_argument('gpu_id', type=int, help='ID of the GPU to pause.')
    # pause_parser inherits --state-file

    resume_parser = subparsers.add_parser('resume', help='Resume a paused GPU by modifying the state file.')
    resume_parser.add_argument('gpu_id', type=int, help='ID of the GPU to resume.')
    # resume_parser inherits --state-file

    screens_parser = subparsers.add_parser('screens', help='List active scheduler GNU Screen sessions.')
    # screens_parser doesn't need state file

    args = parser.parse_args()

    # --- Determine Number of GPUs for non-start commands based on state file or detection ---
    num_gpus_for_control = detected_gpus # Start with detected default
    state_file_path_control = Path(args.state_file)
    try:
        if state_file_path_control.exists():
             with open(state_file_path_control, 'r') as f: state = json.load(f)
             if 'gpu_status' in state and isinstance(state['gpu_status'], list):
                 num_gpus_for_control = len(state['gpu_status'])
                 logger.debug(f"Determined GPU count ({num_gpus_for_control}) from state file '{state_file_path_control}' for control command.")
             else:
                 logger.warning(f"State file '{state_file_path_control}' lacks valid 'gpu_status'. Using detected count: {num_gpus_for_control}")
        else:
             logger.debug(f"State file '{state_file_path_control}' not found. Using detected GPU count: {num_gpus_for_control}")
    except Exception as e:
        logger.warning(f"Error reading GPU count from state file '{state_file_path_control}': {e}. Using detected count: {num_gpus_for_control}")


    # --- Execute Commands ---
    if args.command == 'start':
        jobs_file_to_monitor_path = args.jobs_file if args.jobs_file else None # Handle empty string case

        if jobs_file_to_monitor_path:
            logger.info(f"Job file monitoring enabled for: '{jobs_file_to_monitor_path}'")
        else:
            logger.info("Job file monitoring disabled.")

        logger.info(f"Starting scheduler: GPUs={args.gpus}, State File='{args.state_file}', LLM Config='{args.llm_config}'")
        logger.info(f"Running jobs will be logged to: {RUNNING_JOBS_LOG_FILE}") # <<< ADDED >>>
        logger.info(f"Completed jobs will be logged to: {DONE_JOBS_LOG_FILE}") # <<< ADDED >>>

        scheduler = GPUJobScheduler(
            num_gpus=args.gpus,
            gpu_memory_threshold=args.mem_threshold,
            gpu_load_threshold=args.load_threshold,
            jobs_file_path=jobs_file_to_monitor_path, # Path for monitor thread
            llm_config_path=args.llm_config,
            state_file_path=args.state_file, # Pass state file path
            monitor_interval=args.monitor_interval,
            state_check_interval=args.state_interval, # Pass state check interval
            max_assignment_attempts=args.max_assign_attempts,
            assignment_retry_wait=args.assign_retry_wait
        )
        if args.screen: scheduler.enable_screen() # Check for screen AND script command here

        # Start workers and monitor (monitor also does initial load if job file provided)
        scheduler.start()
        try:
            while True: time.sleep(60) # Keep main thread alive
        except KeyboardInterrupt: logger.info("Ctrl+C received. Initiating shutdown...")
        finally: scheduler.stop(); logger.info("Scheduler shutdown complete.")

    elif args.command == 'add':
        # Format the job line
        parts = [
            str(args.priority),
            args.script,
            args.conda or "", # Use empty string if None
            args.args or "", # Use empty string if None
            args.gpus or ""   # Use empty string if None
        ]
        # Only include trailing parts if they are specified
        while len(parts) > 2 and parts[-1] == "":
            parts.pop()
        job_line = ",".join(parts)

        if args.output_file == "-":
            print(job_line)
        else:
            output_p = Path(args.output_file)
            try:
                with open(output_p, 'a') as f:
                    f.write(job_line + "\n")
                print(f"Job line appended to: {output_p}")
                print(f"Line: {job_line}")
                print("NOTE: If the scheduler is running with monitoring enabled, it should pick this up.")
            except Exception as e:
                print(f"Error writing to file {output_p}: {e}")

    elif args.command == 'status':
        # Create a temporary instance JUST to read state and get status
        # Use the determined number of GPUs
        scheduler_control = GPUJobScheduler(num_gpus=num_gpus_for_control, state_file_path=args.state_file)
        status = scheduler_control.get_gpu_status()
        print(f"\n--- GPU Status (Based on State File: {args.state_file}) ---")
        print(f"{'GPU ID':<8} {'State':<15} {'Allocation':<12} {'Memory Util':<15} {'Load Util':<15}")
        print("-" * 70)
        for gpu in status:
             print(f"{gpu['gpu_id']:<8} {gpu['state']:<15} {gpu['allocation']:<12} {gpu['memory_util']:<15} {gpu['load']:<15}")
        print("\nNote: Real-time utilization requires GPUtil. Status reflects state file content.")
        print("Job queue information is only available within the running scheduler process.")
        # <<< ADDED >>> Also mention the log files
        print(f"\nCheck '{RUNNING_JOBS_LOG_FILE}' and '{DONE_JOBS_LOG_FILE}' for job-specific logs.")


    elif args.command == 'pause':
        if not 0 <= args.gpu_id < num_gpus_for_control:
             print(f"Error: Invalid GPU ID {args.gpu_id}. Expected 0 to {num_gpus_for_control - 1}.")
        elif _modify_state_file_pause_status(args.state_file, num_gpus_for_control, args.gpu_id, pause=True):
             print(f"GPU {args.gpu_id} pause request written to state file '{args.state_file}'.")
             print("The running scheduler should detect this change shortly.")
        else: print(f"Failed to update state file '{args.state_file}' to pause GPU {args.gpu_id}.")

    elif args.command == 'resume':
        if not 0 <= args.gpu_id < num_gpus_for_control:
             print(f"Error: Invalid GPU ID {args.gpu_id}. Expected 0 to {num_gpus_for_control - 1}.")
        elif _modify_state_file_pause_status(args.state_file, num_gpus_for_control, args.gpu_id, pause=False):
             print(f"GPU {args.gpu_id} resume request written to state file '{args.state_file}'.")
             print("The running scheduler should detect this change shortly.")
        else: print(f"Failed to update state file '{args.state_file}' to resume GPU {args.gpu_id}.")

    elif args.command == 'screens':
        print("Listing active scheduler GNU Screen sessions...")
        active_sessions = list_screens()
        if not active_sessions: print("No matching screen sessions found.")
        else:
            print("\n--- Active GPU Job Screen Sessions ---")
            for i, session in enumerate(active_sessions): print(f"{i+1}. {session} (Attach: screen -r {session})")


if __name__ == "__main__":
    main()
# ```
#