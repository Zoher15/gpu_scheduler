import os
import time
import subprocess
import threading
import queue # Use PriorityQueue
import GPUtil
import logging
import shlex
import json
from datetime import datetime
import re
import argparse
import uuid # Import UUID library
from pathlib import Path # Use pathlib
from typing import List, Dict, Tuple, Any, Optional, Set # Add type hints and Set
import hashlib # For hashing job lines
import math # For ceil

# --- Constants ---
DEFAULT_JOBS_FILE = "jobs.txt"
DEFAULT_LLM_CONFIG_FILE = "llm_config.json" # Default LLM config file name
FILE_MONITOR_INTERVAL_S = 30 # Check jobs file every 30 seconds
MARKER_FILE_DIR = Path("/tmp/gpu_scheduler_markers") # Directory for screen success markers
MAX_ASSIGNMENT_ATTEMPTS = 5 # Max times a worker tries to assign a job before requeuing
ASSIGNMENT_RETRY_WAIT_S = 5 # Base wait time between assignment attempts
GPU_ALLOCATION_PRECISION = 0.01 # For float comparisons

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

# Ensure marker directory exists
try:
    MARKER_FILE_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create marker directory {MARKER_FILE_DIR}: {e}. Screen success detection may fail.")


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
        screen_pattern = re.compile(r'^\s*(\d+\.gpujob_\d+_\S+)\s+\(.*\)', re.MULTILINE)
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
    """
    def __init__(self,
                 num_gpus: int = 8,
                 gpu_memory_threshold: float = 0.8,
                 gpu_load_threshold: float = 0.8,
                 jobs_file_path: Optional[str] = None, # Path to jobs file
                 llm_config_path: str = DEFAULT_LLM_CONFIG_FILE, # Path to LLM config
                 monitor_interval: int = FILE_MONITOR_INTERVAL_S, # Interval for file check
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
            monitor_interval: Interval in seconds to check the jobs file.
            max_assignment_attempts: Max times a worker tries to assign a job before requeuing.
            assignment_retry_wait: Base wait time (s) between assignment attempts.
        """
        self.use_screen: bool = False # Whether to use GNU Screen for jobs
        self.num_gpus: int = num_gpus
        # --- Job Tuple Format ---
        # (priority, job_id, script_path, conda_env, args, allowed_gpus,
        #  job_hash, required_gpus, llm_name)
        # required_gpus: float, derived from llm_config based on --llm arg
        # llm_name: str or None
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        # --- GPU Status Tracking ---
        # List of floats representing allocated capacity (0.0 = free, 1.0 = fully busy)
        self.gpu_status: List[float] = [0.0] * num_gpus
        # -------------------------
        self.gpu_memory_threshold: float = gpu_memory_threshold
        self.gpu_load_threshold: float = gpu_load_threshold
        self.lock: threading.Lock = threading.Lock() # Lock for shared resources (queue, status, hashes, paused_gpus)
        self.stop_event: threading.Event = threading.Event() # Event to signal threads to stop
        self.worker_threads: List[threading.Thread] = [] # List of worker threads
        self.paused_gpus: set[int] = set() # Set of GPU IDs that are manually paused
        self.state_file: Path = Path("gpu_scheduler_state.json") # File to persist GPU status/paused state
        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_retry_wait = assignment_retry_wait

        # --- LLM Configuration ---
        self.llm_config_path = Path(llm_config_path)
        self.llm_requirements: Dict[str, float] = {}
        self.default_llm_requirement: float = 1.0
        self._load_llm_config()
        # -------------------------

        # --- Attributes for dynamic job file handling ---
        self.jobs_file_path: Optional[Path] = Path(jobs_file_path) if jobs_file_path else None
        self.file_monitor_interval: int = monitor_interval
        self.file_monitor_thread: Optional[threading.Thread] = None
        # Stores hashes of jobs from the file that have been added to the queue
        # during this scheduler run. Prevents re-adding completed jobs.
        self.managed_job_hashes: Set[str] = set()
        # Maps the hash (from file) to the unique job_id assigned when queued.
        self.hash_to_job_id: Dict[str, str] = {}
        # ----------------------------------------------------

        self.load_state() # Load previous state (paused/busy GPUs)
        self._apply_paused_state() # Apply the loaded paused state (log info)

    def _load_llm_config(self):
        """Loads LLM GPU requirements from the specified JSON file."""
        logger.info(f"Loading LLM configuration from: {self.llm_config_path}")
        if not self.llm_config_path.is_file():
            logger.warning(f"LLM config file not found: {self.llm_config_path}. Using default requirement ({self.default_llm_requirement}) for all LLMs.")
            return

        try:
            with open(self.llm_config_path, 'r') as f:
                config = json.load(f)
            self.default_llm_requirement = float(config.get("default_requirement", 1.0))
            self.llm_requirements = {k: float(v) for k, v in config.get("models", {}).items()}
            logger.info(f"LLM config loaded. Default: {self.default_llm_requirement}, Models: {self.llm_requirements}")
            if not self.llm_requirements:
                 logger.warning("LLM config file loaded, but 'models' section is empty or missing.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM config file {self.llm_config_path}: {e}. Using defaults.")
        except (ValueError, TypeError) as e:
             logger.error(f"Error parsing values in LLM config file {self.llm_config_path} (requirements must be numbers): {e}. Using defaults.")
        except Exception as e:
            logger.error(f"Failed to load or parse LLM config file {self.llm_config_path}: {e}. Using defaults.", exc_info=True)

    def get_llm_gpu_requirement(self, llm_name: Optional[str]) -> float:
        """Gets the GPU requirement for a given LLM name from the loaded config."""
        if llm_name and llm_name in self.llm_requirements:
            return self.llm_requirements[llm_name]
        else:
            if llm_name:
                 logger.debug(f"LLM '{llm_name}' not found in config. Using default requirement: {self.default_llm_requirement}")
            return self.default_llm_requirement

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
    def _apply_paused_state(self):
        """Logs the currently paused GPUs after loading state."""
        logger.info(f"Applying paused state for GPUs: {self.paused_gpus}")

    def load_state(self):
        """Loads GPU allocation/paused status from the state file."""
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
                loaded_paused = state.get('paused_gpus', [])
                # Ensure paused GPUs are valid within the current range
                self.paused_gpus = set(gpu_id for gpu_id in loaded_paused if 0 <= gpu_id < self.num_gpus)

                logger.info(f"State loaded successfully. Status (Allocation): {self.gpu_status}, Paused: {self.paused_gpus}")
            except Exception as e:
                logger.error(f"Failed to load state from '{self.state_file}': {e}. Starting fresh.", exc_info=True)
                self.gpu_status = [0.0] * self.num_gpus # Initialize as free
                self.paused_gpus = set()
        else:
            logger.info("No state file found. Starting with all GPUs free.")
            self.gpu_status = [0.0] * self.num_gpus
            self.paused_gpus = set()
        # Note: We don't persist managed_job_hashes. They are rebuilt on restart by scanning the file.

    def save_state(self):
        """Saves the current GPU allocation/paused status to the state file."""
        with self.lock: # Ensure thread safety when accessing shared state
            try:
                state = {
                    # Save allocation status
                    "gpu_status": [round(s, 4) for s in self.gpu_status], # Round for cleaner JSON
                    "paused_gpus": list(self.paused_gpus)
                    # Do not save managed_job_hashes here; they are transient
                }
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4)
                logger.debug("Scheduler state saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save state to '{self.state_file}': {e}")

    # --- GPU Control ---
    def pause_gpu(self, gpu_id: int) -> bool:
        """Marks a GPU as paused, preventing new jobs from being assigned to it."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} is already paused.")
                return True
            self.paused_gpus.add(gpu_id)
            logger.info(f"GPU {gpu_id} paused. It will not accept new jobs.")
        self.save_state() # Persist the change
        return True

    def resume_gpu(self, gpu_id: int) -> bool:
        """Resumes a paused GPU, allowing new jobs to be assigned to it."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id not in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} was not paused.")
                return True
            self.paused_gpus.remove(gpu_id)
            logger.info(f"GPU {gpu_id} resumed. It can now accept new jobs.")
        self.save_state() # Persist the change
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
            current_paused_gpus = set(self.paused_gpus)

        for gpu_id in range(self.num_gpus):
            # Determine state based on scheduler's knowledge
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
                "state": state, # Shows PAUSED, AVAILABLE, or BUSY(% allocation)
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
            #             job_hash, required_gpus, llm_name)
            return sorted(list(self.job_queue.queue), key=lambda x: (x[0], x[1]))

    # --- Scheduler Lifecycle ---
    def start(self):
        """Start the GPU job scheduler worker threads and file monitor."""
        if self.worker_threads or self.file_monitor_thread:
            logger.warning("Scheduler appears to be already started.")
            return

        logger.info(f"Starting GPU Job Scheduler with {self.num_gpus} GPUs...")
        self.stop_event.clear() # Ensure stop event is not set

        # --- Start File Monitor Thread (if file path provided) ---
        if self.jobs_file_path:
            # Perform initial load from the jobs file before starting monitor
            logger.info(f"Performing initial load from job file: {self.jobs_file_path}")
            self.add_jobs_from_file(str(self.jobs_file_path), initial_load=True)

            self.file_monitor_thread = threading.Thread(
                target=self._monitor_jobs_file,
                name="FileMonitor",
                daemon=True # Allow program exit even if this thread is running
            )
            self.file_monitor_thread.start()
        else:
            logger.info("No jobs file specified (--jobs-file), file monitoring disabled.")
        # ---------------------------------------------------------

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

        # Stop the file monitor thread first
        if self.file_monitor_thread and self.file_monitor_thread.is_alive():
            logger.info("Waiting for file monitor thread to finish...")
            self.file_monitor_thread.join(timeout=5.0) # Wait with timeout
            if self.file_monitor_thread.is_alive():
                logger.warning("File monitor thread did not finish within timeout.")

        # Stop worker threads by putting sentinel values in the queue
        logger.info("Signaling worker threads to stop...")
        num_sentinels = len(self.worker_threads) # Send one sentinel per worker
        for i in range(num_sentinels):
            try:
                # Sentinel: (priority, job_id=None, ..., required_gpus=0.0, llm_name=None)
                self.job_queue.put((float('inf'), None, None, None, None, None, None, 0.0, None))
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
            self.job_queue.join()
            logger.info("Job queue successfully joined (all tasks processed).")
        except Exception as e:
            logger.error(f"Error joining job queue during shutdown: {e}")
            if not self.job_queue.empty():
                logger.warning(f"Job queue is not empty after workers stopped ({self.job_queue.qsize()} items remaining).")


        # --- Clean up managed hashes ---
        logger.info("Clearing internal managed job tracking for restart.")
        with self.lock:
            self.managed_job_hashes.clear()
            self.hash_to_job_id.clear()
        # --------------------------------------------------------------

        logger.info("Saving final scheduler state (GPU allocation/paused)...")
        self.save_state() # Save allocation and paused status
        logger.info("Scheduler stopped.")

    # --- Worker Logic (REVISED for Fractional/Multi-GPU) ---
    def worker(self):
        """Worker thread logic: Get job, find suitable GPU(s), launch job runner."""
        thread_name = threading.current_thread().name
        logger.info(f"Worker started.")

        while not self.stop_event.is_set():
            job_tuple = None
            job_id = None
            job_hash = None
            script_path = None
            required_gpus = 0.0
            got_task = False # Track if we successfully got a task (job or sentinel)

            try:
                # Get job: (priority, job_id, script_path, conda_env, args, allowed_gpus,
                #           job_hash, required_gpus, llm_name)
                job_tuple = self.job_queue.get(block=True, timeout=1.0)
                got_task = True # We got something from the queue
                (priority, job_id, script_path, _, original_args,
                 allowed_gpus_list, job_hash, required_gpus, llm_name) = job_tuple

                # Check for sentinel value (job_id is None) used for stopping
                if job_id is None:
                    logger.info(f"Received sentinel, exiting.")
                    break # Exit the worker loop

                logger.debug(f"Retrieved job ID {job_id} (Hash: {job_hash}, LLM: {llm_name}, GPUs Req: {required_gpus:.2f})")

                assigned = False
                attempts = 0
                assigned_gpu_ids: List[int] = [] # Store the ID(s) of the assigned GPU(s)

                # Loop to find suitable GPU(s) for the *current* job
                while not assigned and not self.stop_event.is_set() and attempts < self.max_assignment_attempts:
                    assigned_gpu_ids = [] # Reset for each attempt
                    with self.lock: # Lock needed to check/update shared self.gpu_status and self.paused_gpus
                        
                        # --- Determine candidate GPUs (respecting allowed_gpus_list) ---
                        candidate_indices = [
                            i for i in range(self.num_gpus)
                            if i not in self.paused_gpus and \
                               (allowed_gpus_list is None or i in allowed_gpus_list)
                        ]
                        logger.debug(f"Job {job_id}: Candidate GPUs (not paused, allowed): {candidate_indices}")

                        # --- Allocation Strategy ---
                        if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION):
                            # --- Strategy 1: Fractional or Single Full GPU ---
                            found_gpu = False
                            for gpu_id in candidate_indices:
                                current_allocation = self.gpu_status[gpu_id]
                                # Check if adding the job exceeds 1.0 (within precision)
                                if (current_allocation + required_gpus) <= (1.0 + GPU_ALLOCATION_PRECISION):
                                    # Check real-time load/memory thresholds if GPU is partially busy
                                    passes_realtime_check = True
                                    if current_allocation > GPU_ALLOCATION_PRECISION: # Only check if already busy
                                         try:
                                             gpus_util = GPUtil.getGPUs()
                                             if gpu_id < len(gpus_util):
                                                 gpu = gpus_util[gpu_id]
                                                 passes_mem = gpu.memoryUtil < self.gpu_memory_threshold
                                                 passes_load = gpu.load < self.gpu_load_threshold
                                                 passes_realtime_check = passes_mem and passes_load
                                                 logger.debug(f"GPU Check [{gpu_id}] Realtime: Mem={gpu.memoryUtil:.2f}<{self.gpu_memory_threshold}({passes_mem}), Load={gpu.load:.2f}<{self.gpu_load_threshold}({passes_load})")
                                             else: passes_realtime_check = False # Should not happen
                                         except Exception as e:
                                             logger.error(f"GPUtil Check [{gpu_id}] failed during fractional allocation: {e}")
                                             passes_realtime_check = False # Don't allocate if check fails

                                    if passes_realtime_check:
                                        logger.info(f"Found suitable fractional/single GPU {gpu_id} for job ID {job_id} (Req: {required_gpus:.2f}, Avail: {1.0-current_allocation:.2f})")
                                        self.gpu_status[gpu_id] += required_gpus
                                        assigned_gpu_ids = [gpu_id]
                                        found_gpu = True
                                        break # Found a suitable GPU
                                    else:
                                         logger.debug(f"GPU Check [{gpu_id}] failed real-time check for fractional job {job_id}.")
                                else:
                                     logger.debug(f"GPU Check [{gpu_id}] insufficient capacity for job ID {job_id} (Req: {required_gpus:.2f}, Avail: {1.0-current_allocation:.2f})")
                            if found_gpu:
                                assigned = True

                        else:
                            # --- Strategy 2: Multiple Full GPUs ---
                            # Simplification: Requires *completely free* GPUs (allocation == 0.0)
                            num_needed = math.ceil(required_gpus) # How many physical GPUs
                            logger.debug(f"Job {job_id} requires {num_needed} fully free GPUs.")
                            free_gpu_ids = [
                                i for i in candidate_indices
                                if abs(self.gpu_status[i]) < GPU_ALLOCATION_PRECISION # Check if essentially 0.0
                            ]

                            if len(free_gpu_ids) >= num_needed:
                                # Check real-time stats for these free GPUs (optional but good practice)
                                suitable_free_gpus = []
                                try:
                                    gpus_util = GPUtil.getGPUs()
                                    for gpu_id in free_gpu_ids:
                                        passes_realtime_check = True
                                        if gpu_id < len(gpus_util):
                                            gpu = gpus_util[gpu_id]
                                            passes_mem = gpu.memoryUtil < self.gpu_memory_threshold
                                            passes_load = gpu.load < self.gpu_load_threshold
                                            passes_realtime_check = passes_mem and passes_load
                                            logger.debug(f"GPU Check [{gpu_id}] Realtime (Multi-GPU): Mem={gpu.memoryUtil:.2f}<{self.gpu_memory_threshold}({passes_mem}), Load={gpu.load:.2f}<{self.gpu_load_threshold}({passes_load})")
                                        else: passes_realtime_check = False

                                        if passes_realtime_check:
                                            suitable_free_gpus.append(gpu_id)
                                        if len(suitable_free_gpus) == num_needed:
                                            break # Found enough suitable free GPUs
                                except Exception as e:
                                     logger.error(f"GPUtil Check failed during multi-GPU allocation: {e}")
                                     suitable_free_gpus = [] # Reset on error


                                if len(suitable_free_gpus) >= num_needed:
                                    assigned_gpu_ids = suitable_free_gpus[:num_needed] # Take the first N found
                                    logger.info(f"Found suitable free GPUs {assigned_gpu_ids} for multi-GPU job ID {job_id} (Req: {required_gpus:.2f})")
                                    # Mark each assigned GPU as fully busy (1.0)
                                    for gpu_id in assigned_gpu_ids:
                                        self.gpu_status[gpu_id] = 1.0
                                    assigned = True
                                else:
                                     logger.debug(f"Found {len(free_gpu_ids)} free GPUs, but only {len(suitable_free_gpus)} passed real-time checks for multi-GPU job {job_id} (needed {num_needed}).")
                            else:
                                logger.debug(f"Insufficient fully free GPUs for multi-GPU job ID {job_id} (Need: {num_needed}, Free & Allowed: {len(free_gpu_ids)})")
                    # --- End Allocation Strategy ---

                    # --- After checking all GPUs (inside lock) ---
                    if assigned:
                        break # Exit the attempt loop (while not assigned...)

                # --- After attempt loop (outside lock) ---
                if assigned:
                    # Successfully found and reserved GPU(s)
                    self.save_state() # Save state reflecting the new allocation
                    # Unpack details for the job runner thread
                    _, run_job_id, run_script_path, run_conda_env, run_args, _, run_job_hash, run_req_gpus, run_llm_name = job_tuple
                    mode_tag = _extract_and_sanitize_key_arg(run_args)
                    logger.info(f"Assigning job ID {run_job_id} (LLM: {run_llm_name}, Req: {run_req_gpus:.2f}) to GPU(s) {assigned_gpu_ids}")

                    # Start a new thread to run the job, passing the assigned GPU IDs and requirement
                    job_runner_thread = threading.Thread(
                        target=self._run_job,
                        # Pass job_tuple, assigned_gpu_ids
                        args=(job_tuple, assigned_gpu_ids),
                        name=f"JobRunner-GPU{','.join(map(str, assigned_gpu_ids))}-{mode_tag}-{run_job_id[:4]}",
                        daemon=True
                    )
                    job_runner_thread.start()
                    # assigned = True # Already set
                else:
                    # No suitable GPU(s) found in this attempt
                    attempts += 1
                    logger.debug(f"Found no suitable GPU(s) for job ID {job_id} (Req: {required_gpus:.2f}) (Attempt {attempts}/{self.max_assignment_attempts}). Will wait and retry.")

                    # Wait *outside* the lock before next attempt
                    wait_time = min(self.assignment_retry_wait * (attempts), 30)
                    if self.stop_event.wait(timeout=wait_time):
                        logger.info(f"Stop event received while waiting to assign job {job_id}. Re-queueing.")
                        try:
                            self.job_queue.put(job_tuple)
                            logger.info(f"Job ID {job_id} re-queued due to worker stop.")
                        except Exception as e:
                            logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} during stop: {e}. Job may be lost.")
                        break # Exit assignment attempts loop

                # --- End of assignment attempt loop --- (while not assigned...)
                if not assigned:
                    # This block is reached if stop_event interrupted the wait OR max_attempts was reached
                    if attempts >= self.max_assignment_attempts:
                        logger.warning(f"Exceeded max attempts ({self.max_assignment_attempts}) to find GPU(s) for job ID {job_id} (Req: {required_gpus:.2f}). Re-queueing job.")
                        try:
                            self.job_queue.put(job_tuple)
                        except Exception as e:
                            logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} after max attempts: {e}. Job may be lost.")
                    # If stop_event caused the loop break, the job was already re-queued inside the loop

                # Check stop event again before trying to get the next job
                if self.stop_event.is_set():
                    logger.info("Stop event detected after assignment loop. Exiting worker.")
                    break # Exit the main worker loop (while not self.stop_event.is_set():)

            except queue.Empty:
                # Queue was empty during the timeout, just loop again and wait
                got_task = False # Didn't get a task this time
                continue
            except Exception as e:
                # Catch unexpected errors in the main worker loop after getting a job
                logger.error(f"Error in worker main loop processing job ID {job_id or 'N/A'} (Hash: {job_hash or 'N/A'}): {e}", exc_info=True)
                got_task = True # We did get a task, even if it caused an error here
            finally:
                # Crucial: Mark the task as done *if* we successfully got one from the queue.
                if got_task:
                    try:
                        self.job_queue.task_done()
                        logger.debug(f"task_done() called for job ID {job_id} (Hash: {job_hash})")
                    except ValueError:
                        logger.error(f"CRITICAL: task_done() called too many times for job ID {job_id} (Hash: {job_hash})!")
                    except Exception as e:
                        logger.error(f"Error calling task_done() for job ID {job_id} (Hash: {job_hash}): {e}")

        logger.info(f"Worker finished.")


    # --- Job Execution (REVISED) ---
    def _run_job(self, job_tuple: Tuple, assigned_gpu_ids: List[int]):
            """
            Internal method called by a dedicated thread to prepare environment
            and execute a single job on its assigned GPU(s).
            Modifies '--cuda' argument to pass assigned IDs.
            Ensures the GPU allocation is released afterwards.

            Args:
                job_tuple: The full job tuple including requirements.
                assigned_gpu_ids: The list of GPU IDs assigned by the worker.
            """
            (priority, job_id, script_path, conda_env, original_args, _,
             job_hash, required_gpus, llm_name) = job_tuple
            job_name = Path(script_path).name
            start_time = datetime.now()
            mode_tag = _extract_and_sanitize_key_arg(original_args) # Extract mode from original args

            if not assigned_gpu_ids:
                 logger.error(f"CRITICAL: _run_job called for job ID {job_id} with no assigned GPUs. Aborting execution.")
                 # Attempt to release based on requirement? Risky. Log critical error.
                 # No GPU to release state for. Hash remains managed.
                 return

            # --- Prepare arguments for the actual execution ---
            # Remove any existing --cuda argument first
            args_no_cuda = [arg for i, arg in enumerate(original_args or [])
                            if not (arg == '--cuda' or (i > 0 and original_args[i-1] == '--cuda'))]
            # Add the new --cuda argument with the assigned ID(s)
            cuda_arg_value = ",".join(map(str, sorted(assigned_gpu_ids)))
            args_for_exec = args_no_cuda + ["--cuda", cuda_arg_value]
            # -------------------------------------------------

            job_completed_successfully = False
            release_reason = "unknown"

            logger.info(f"GPU(s) {cuda_arg_value}: Preparing job ID {job_id} (LLM: {llm_name}, Req: {required_gpus:.2f})")
            logger.debug(f"GPU(s) {cuda_arg_value}: Job ID {job_id} arguments for execution: {args_for_exec}")

            # Set CUDA_VISIBLE_DEVICES environment variable for the job
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = cuda_arg_value

            try:
                # Create exec tuple with modified args for passing to run methods
                exec_job_tuple = (priority, job_id, script_path, conda_env, args_for_exec,
                                  job_tuple[5], job_hash, required_gpus, llm_name) # Pass full tuple

                if self.use_screen:
                    # --- Screen Mode ---
                    # Pass assigned_gpu_ids and required_gpus for monitoring/release
                    launched_ok = self._run_with_screen(exec_job_tuple, assigned_gpu_ids, env, start_time, mode_tag)
                    if not launched_ok:
                        logger.error(f"GPU(s) {cuda_arg_value}: Screen setup failed for job ID {job_id} (Hash: {job_hash}).")
                        job_completed_successfully = False
                        release_reason = "screen setup failure"
                        # Release allocation immediately if screen launch failed
                        self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                    # If launched_ok is True, the monitor thread will handle release
                    return # Exit _run_job thread; monitor takes over

                else:
                    # --- Direct Mode ---
                    job_completed_successfully = self._run_directly(exec_job_tuple, assigned_gpu_ids, env, start_time, mode_tag)
                    release_reason = f"direct execution completion (Success: {job_completed_successfully})"
                    # Release allocation after direct execution finishes
                    self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                    return

            except Exception as e:
                logger.error(f"GPU(s) {cuda_arg_value}: CRITICAL Unexpected error in _run_job setup/dispatch for job ID {job_id} (Hash: {job_hash}): {e}", exc_info=True)
                job_completed_successfully = False
                release_reason = "critical error in _run_job setup"
                # Ensure GPU allocation is released if error occurred before launch/monitor took over
                # Check if allocation still seems present
                allocation_present = False
                with self.lock:
                    for gpu_id in assigned_gpu_ids:
                         if 0 <= gpu_id < len(self.gpu_status) and self.gpu_status[gpu_id] > GPU_ALLOCATION_PRECISION:
                              allocation_present = True
                              break
                if allocation_present:
                    logger.warning(f"GPU(s) {cuda_arg_value}: Releasing allocation due to error during _run_job setup for job ID {job_id} (Hash: {job_hash}).")
                    self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                else:
                    logger.warning(f"GPU(s) {cuda_arg_value}: Error during _run_job setup for job ID {job_id} (Hash: {job_hash}), but allocation seemed already released.")


    # --- Job Release (REVISED) ---
    def _release_gpu(self, assigned_gpu_ids: List[int], required_gpus: float,
                     job_id: str, job_name: str, start_time: datetime, mode_tag: str,
                     job_hash: Optional[str], success: bool, reason: str="completion"):
        """
        Helper method to mark GPU allocation as released and save state.
        Subtracts the required_gpus fraction (or 1.0 for multi-GPU) from assigned GPUs.
        Hash is NOT removed here.

        Args:
            assigned_gpu_ids: List of IDs of the GPUs the job ran on.
            required_gpus: The fractional GPU requirement of the job.
            job_id, job_name, start_time, mode_tag, job_hash: Job metadata.
            success: Boolean indicating if the job succeeded.
            reason: String describing why the release is happening.
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
        log_prefix = f"GPU(s) {gpu_ids_str}: Job ID {job_id} (Hash: {job_hash}, Name: '{job_name}', Mode: {mode_tag})"
        logger.info(f"{log_prefix} finished due to {reason} in {duration:.2f} seconds. Success: {success}. Releasing allocation (Req: {required_gpus:.2f}).")

        with self.lock: # Ensure thread safety
            # --- Release Allocation ---
            allocation_to_release = 0.0
            if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION):
                # Fractional or single full GPU job
                allocation_to_release = required_gpus
            else:
                # Multi-GPU job (currently assumes 1.0 allocation per GPU)
                allocation_to_release = 1.0

            for gpu_id in assigned_gpu_ids:
                if 0 <= gpu_id < len(self.gpu_status):
                    # Subtract the allocation, ensuring it doesn't go below 0
                    new_allocation = max(0.0, self.gpu_status[gpu_id] - allocation_to_release)
                    logger.debug(f"Releasing GPU {gpu_id}: Old Alloc={self.gpu_status[gpu_id]:.2f}, Release={allocation_to_release:.2f}, New Alloc={new_allocation:.2f}")
                    self.gpu_status[gpu_id] = new_allocation
                else:
                    logger.error(f"Attempted to release allocation for invalid GPU ID {gpu_id}. State not changed for this ID.")
            # -------------------------

            # --- Manage Job Hash ---
            # The job_hash is intentionally NOT removed from managed_job_hashes here.
            if job_hash:
                 logger.debug(f"{log_prefix}: Hash '{job_hash}' remains in managed set after job completion.")
            # --- End Hash Management ---

        self.save_state() # Save updated GPU allocation status


    # --- Run with Screen (REVISED) ---
    def _run_with_screen(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: Dict, start_time: datetime, mode_tag: str) -> bool:
            """
            Sets up and launches a job inside a GNU Screen session on assigned GPUs.
            Uses `tee` to attempt showing progress bars while logging.
            Args include modified '--cuda' and full job details.

            Returns:
                bool: True if the screen session was launched successfully, False otherwise.
            """
            (priority, job_id, script_path, conda_env, args_with_cuda, _,
             job_hash, required_gpus, llm_name) = job_tuple # Unpack full tuple
            script_p = Path(script_path)
            script_basename = script_p.name
            gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))

            session_name = f"gpujob_{gpu_ids_str}_{mode_tag}_{job_id[:6]}_{start_time.strftime('%H%M%S')}"
            session_name = session_name[:60] # Limit length

            # --- Files for communication ---
            marker_file = MARKER_FILE_DIR / f"success_{session_name}.marker"
            exit_code_file = MARKER_FILE_DIR / f"exitcode_{session_name}.txt"
            job_stdout_log = MARKER_FILE_DIR / f"output_{session_name}.log"
            marker_file.unlink(missing_ok=True)
            exit_code_file.unlink(missing_ok=True)
            job_stdout_log.unlink(missing_ok=True)
            # -----------------------------

            temp_script_path = None
            try:
                # --- Create temporary wrapper script ---
                script_content = [
                    "#!/bin/bash",
                    "# --- Auto-generated by GPUJobScheduler ---",
                    f"# Job ID: {job_id}", f"# Hash: {job_hash}", f"# LLM: {llm_name}", f"# GPUs Req: {required_gpus:.2f}",
                    f"# Screen session: {session_name}",
                    f"# Script: {script_path}", f"# Assigned GPU(s): {gpu_ids_str}", f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# Marker File: {marker_file}",
                    f"# Exit Code File: {exit_code_file}",
                    f"# Job Stdout/Stderr Log: {job_stdout_log}",
                    "echo \"--- Starting Job Execution Script ---\"",
                    "echo \"Timestamp: $(date)\"",
                    "echo \"Running as user: $(whoami)\"",
                    "echo \"Current directory: $(pwd)\"",
                    f"echo \"Assigned CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not Set')}\"", # Log assigned devices
                    "set -e",
                    "set -o pipefail"
                ]

                # --- Conda activation logic (Robust path finding - same as before) ---
                if conda_env:
                    # ... [Copy robust conda path finding and activation logic here] ...
                    conda_base_cmd = "..." # Placeholder
                    conda_path_found = False # Placeholder
                    # ...
                    if not conda_path_found: logger.warning(f"GPU(s) {gpu_ids_str}: Job ID {job_id}: Could not reliably determine conda base path.")
                    script_content.extend([
                         f"echo 'Attempting to initialize conda using: {conda_base_cmd}'", conda_base_cmd, "...",
                         f"echo 'Activating conda environment: {conda_env}'", f"conda activate {shlex.quote(conda_env)}", "...",
                         "echo 'Conda environment activated.'", "..."
                    ])
                else:
                    script_content.append("echo 'No conda environment specified.'")

                # Prepare Python command execution using args_with_cuda (which has modified --cuda)
                cmd_list = ["python", "-u", script_path]
                if args_with_cuda:
                    try: str_args = [str(arg) for arg in args_with_cuda]; cmd_list.extend(str_args)
                    except Exception as e_args: raise ValueError(f"Invalid screen arguments for job {job_id}") from e_args

                cmd_str = shlex.join(cmd_list)

                # --- Use tee for execution ---
                script_content.extend([
                    f"echo 'Executing Python command: {cmd_str}'",
                    f"echo 'Output piped via tee to log: {job_stdout_log}'",
                    f"echo '--- Python Script Output Start ---' >> {shlex.quote(str(job_stdout_log))}",
                    f"{cmd_str} 2>&1 | tee -a {shlex.quote(str(job_stdout_log))}",
                    "exit_code=${PIPESTATUS[0]}",
                    f"echo '--- Python Script Output End (Exit Code: $exit_code) ---' >> {shlex.quote(str(job_stdout_log))}",
                    f"echo \"Python command finished with exit code: $exit_code\"",
                ])
                # --- End tee execution ---

                # --- Record results ---
                script_content.extend([
                    f"echo $exit_code > {shlex.quote(str(exit_code_file))}",
                    f"if [ $exit_code -eq 0 ]; then",
                    f"  echo 'Command succeeded. Creating marker file.'",
                    f"  touch {shlex.quote(str(marker_file))}",
                    f"else",
                    f"  echo 'Command failed (Exit Code: $exit_code). NOT creating marker file.'",
                    f"fi",
                    f"echo \"--- Ending Job Execution Script --- Timestamp: $(date)\"",
                    f"exit $exit_code"
                ])

                # Write and execute the wrapper script
                temp_script_dir = Path("/tmp")
                temp_script_path = temp_script_dir / f"run_{session_name}_{job_id[:4]}.sh"
                with open(temp_script_path, 'w') as f: f.write("\n".join(script_content))
                os.chmod(temp_script_path, 0o755)
                logger.debug(f"GPU(s) {gpu_ids_str}: Temp script for job ID {job_id}: {temp_script_path}")

                # --- Start screen session ---
                screen_cmd = ['screen', '-dmS', session_name, str(temp_script_path)]
                logger.info(f"GPU(s) {gpu_ids_str}: Starting screen session '{session_name}' for job ID {job_id}")
                process = subprocess.run(screen_cmd, env=env, check=True, capture_output=True, text=True)
                logger.info(f"GPU(s) {gpu_ids_str}: Screen session launched. To view progress: screen -r {session_name}")
                logger.info(f"GPU(s) {gpu_ids_str}: Job stdout/stderr log: {job_stdout_log}")

                # --- Start monitoring thread ---
                # Pass required_gpus for release calculation
                monitoring_thread = threading.Thread(
                    target=self._monitor_screen,
                    args=(session_name, job_id, script_path, assigned_gpu_ids, required_gpus, temp_script_path, start_time, mode_tag, job_hash, marker_file, exit_code_file),
                    name=f"ScreenMonitor-GPU{gpu_ids_str}-{mode_tag}-{job_id[:6]}",
                    daemon=True
                )
                monitoring_thread.start()

                return True # Launch success

            except subprocess.CalledProcessError as e:
                 gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
                 logger.error(f"GPU(s) {gpu_ids_str}: Failed to launch screen session '{session_name}'. Return code: {e.returncode}")
                 logger.error(f"Stdout: {e.stdout}")
                 logger.error(f"Stderr: {e.stderr}")
            except Exception as e:
                gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
                logger.error(f"GPU(s) {gpu_ids_str}: Error setting up or launching screen session for job ID {job_id} (Hash: {job_hash}): {e}", exc_info=True)
                # Cleanup temp files if created
                if temp_script_path and temp_script_path.exists():
                    try: temp_script_path.unlink(missing_ok=True)
                    except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")
                marker_file.unlink(missing_ok=True)
                exit_code_file.unlink(missing_ok=True)
                job_stdout_log.unlink(missing_ok=True)
                return False # Launch failure


    # --- Monitor Screen (REVISED) ---
    def _monitor_screen(self, session_name: str, job_id: str, script_path: str,
                        assigned_gpu_ids: List[int], required_gpus: float, # Added required_gpus
                        temp_script_path: Optional[Path], start_time: datetime, mode_tag: str,
                        job_hash: Optional[str], marker_file: Path, exit_code_file: Path):
        """
        Monitors a specific screen session until it terminates or the scheduler stops.
        Determines job success based on the marker file.
        Cleans up temporary files and calls _release_gpu.
        """
        job_name = Path(script_path).name
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
        log_prefix = f"GPU(s) {gpu_ids_str}: Job ID {job_id} (Hash: {job_hash}, Screen: {session_name}, Mode: {mode_tag})"
        logger.info(f"{log_prefix} Monitoring screen session...")
        active = True
        final_success = False # Assume failure unless marker file proves otherwise
        exit_code = -1 # Default exit code if not found or script fails early
        check_interval = 15 # Seconds between checks

        # Loop while the screen session appears active and the scheduler hasn't stopped
        while active and not self.stop_event.is_set():
            try:
                cmd = f"screen -ls | grep -qE '^\\s*[0-9]+\\.{re.escape(session_name)}\\s+\\(' "
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                active = True
                logger.debug(f"{log_prefix} Screen session is active.")
            except subprocess.CalledProcessError:
                active = False
                logger.info(f"{log_prefix} Screen session finished or not found.")
            except FileNotFoundError:
                 logger.error(f"{log_prefix} 'screen' command not found during monitoring.")
                 active = False
            except Exception as e:
                logger.error(f"{log_prefix} Error checking screen session: {e}. Assuming finished.")
                active = False

            if active:
                if self.stop_event.wait(timeout=check_interval):
                    logger.warning(f"{log_prefix} Stop event received during monitoring.")
                    active = False
                    final_success = False
                    # Proceed to cleanup

        # --- Screen session ended or scheduler stopped ---
        logger.info(f"{log_prefix} Monitoring finished.")

        # Determine final success status based on the marker file
        if marker_file.exists():
            final_success = True
            logger.info(f"{log_prefix} Success marker file found ({marker_file}).")
            try: marker_file.unlink(missing_ok=True)
            except OSError as e: logger.warning(f"{log_prefix} Error removing marker file {marker_file}: {e}")
        else:
            final_success = False
            if not self.stop_event.is_set():
                 logger.warning(f"{log_prefix} Success marker file NOT found. Assuming failure.")
            else:
                 logger.info(f"{log_prefix} Success marker file not found (scheduler stopping).")

        # Read the exit code from the file
        if exit_code_file.exists():
            try:
                exit_code_str = exit_code_file.read_text().strip()
                exit_code = int(exit_code_str)
                logger.info(f"{log_prefix} Read exit code {exit_code} from {exit_code_file}.")
            except Exception as e_read:
                logger.warning(f"{log_prefix} Could not read or parse exit code from {exit_code_file}: {e_read}")
            finally:
                try: exit_code_file.unlink(missing_ok=True)
                except OSError as e: logger.warning(f"{log_prefix} Error removing exit code file {exit_code_file}: {e}")
        elif not self.stop_event.is_set():
             logger.warning(f"{log_prefix} Exit code file ({exit_code_file}) not found.")

        # Clean up the temporary wrapper script
        if temp_script_path and temp_script_path.exists():
            logger.debug(f"{log_prefix} Removing temporary script: {temp_script_path}")
            try: temp_script_path.unlink()
            except OSError as e: logger.warning(f"{log_prefix} Error removing temp script {temp_script_path}: {e}")

        # Determine the reason string for the release log message
        reason = f"screen session ended (Exit Code: {exit_code}, Success: {final_success})"
        if self.stop_event.is_set() and not active:
            reason = f"scheduler shutdown during screen session (Final Status - Exit Code: {exit_code}, Success: {final_success})"

        # --- Call _release_gpu from the monitor thread ---
        # Pass required_gpus needed for correct release calculation
        self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, final_success, reason)


    # --- Run Directly (REVISED) ---
    def _run_directly(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: Dict, start_time: datetime, mode_tag: str) -> bool:
        """
        Runs a job directly as a subprocess on its assigned GPU(s).
        Args include modified '--cuda'.

        Returns:
            bool: True if the job script exited with code 0, False otherwise.
        """
        (priority, job_id, script_path, conda_env, args_with_cuda, _,
         job_hash, required_gpus, llm_name) = job_tuple # Unpack full tuple
        script_p = Path(script_path)
        job_name = script_p.name
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))

        log_filename = Path(f"job_{job_name}_{mode_tag}_{job_hash[:8] if job_hash else 'nohash'}_gpu{gpu_ids_str}_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
        log_prefix = f"GPU(s) {gpu_ids_str}: Job ID {job_id} (LLM: {llm_name}, Req: {required_gpus:.2f})"
        logger.info(f"{log_prefix} Running directly. Log: {log_filename}")

        process = None
        log_file = None
        temp_script_path = None # For conda activation wrapper
        return_code = -1 # Default to failure

        try:
            # Open log file and write header information
            log_file = open(log_filename, 'w', buffering=1) # Line buffered
            log_file.write(f"Job ID: {job_id}\n")
            log_file.write(f"Job Hash: {job_hash or 'N/A'}\n")
            log_file.write(f"LLM Name: {llm_name or 'N/A'}\n")
            log_file.write(f"GPUs Required: {required_gpus:.2f}\n")
            log_file.write(f"Mode Tag: {mode_tag}\n")
            log_file.write(f"Script: {script_path}\n")
            log_file.write(f"Conda Env: {conda_env or 'None'}\n")
            # Log arguments *including* the modified --cuda
            log_file.write(f"Arguments (with modified --cuda): {args_with_cuda or 'None'}\n")
            log_file.write(f"GPU ID(s) Assigned: {gpu_ids_str}\n")
            log_file.write(f"Start Time: {start_time.isoformat()}\n")
            log_file.write("-" * 60 + "\n")
            log_file.flush()

            command_list = []
            shell_mode = False

            # --- Build command (with conda wrapper if needed) ---
            if conda_env:
                # Use a temporary bash script
                shell_script_lines = ["#!/bin/bash", f"# Job ID: {job_id}", f"# Hash: {job_hash}", "set -e", "set -o pipefail"]
                # --- Conda activation logic (Copy from _run_with_screen) ---
                # ... [Copy robust conda path finding and activation logic here] ...
                conda_base_cmd = "..." # Placeholder
                conda_path_found = False # Placeholder
                # ...
                if not conda_path_found: logger.warning(f"{log_prefix}: Could not reliably determine conda base path.")
                shell_script_lines.extend([conda_base_cmd, "...", f"conda activate {shlex.quote(conda_env)}", "..."])
                # --- End Conda Logic ---

                # Prepare the python command part using args_with_cuda
                python_cmd_list = ["python", "-u", script_path]
                if args_with_cuda:
                    try: str_args = [str(arg) for arg in args_with_cuda] ; python_cmd_list.extend(str_args)
                    except Exception as e_args: raise ValueError(f"Invalid direct arguments for job {job_id}") from e_args
                shell_script_lines.append(shlex.join(python_cmd_list))

                # Write and prepare temp script execution
                script_id = f"{start_time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_{job_id[:4]}"
                temp_script_dir = Path("/tmp")
                temp_script_path = temp_script_dir / f"direct_job_{script_id}.sh"
                with open(temp_script_path, 'w') as f: f.write("\n".join(shell_script_lines))
                os.chmod(temp_script_path, 0o755)
                command_list = [str(temp_script_path)]
                shell_mode = False
            else:
                # No conda env, run python directly with args_with_cuda
                command_list = ['python', '-u', script_path]
                if args_with_cuda:
                    try: str_args = [str(arg) for arg in args_with_cuda] ; command_list.extend(str_args)
                    except Exception as e_args: raise ValueError(f"Invalid direct arguments for job {job_id}") from e_args
                shell_mode = False

            logger.info(f"{log_prefix}: Executing command: {' '.join(map(shlex.quote, command_list))}")
            # Start the subprocess
            process = subprocess.Popen(
                command_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, shell=shell_mode
            )

            # --- Real-time output handling ---
            def stream_output(pipe, prefix, log_f, logger_func):
                # (Same implementation as before)
                try:
                    for line in iter(pipe.readline, ''):
                        line_stripped = line.rstrip()
                        logger_func(f"[{log_prefix} {prefix}] {line_stripped}")
                        if log_f:
                            try: log_f.write(f"{prefix}: {line}"); log_f.flush()
                            except Exception as write_e: logger.error(f"Error writing to log file {log_filename}: {write_e}")
                except Exception as e: logger.error(f"Stream reading error ({log_prefix} {prefix}): {e}")
                finally:
                    try: pipe.close()
                    except Exception: pass

            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "OUT", log_file, logger.info), daemon=True)
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERR", log_file, logger.warning), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process completion
            return_code = process.wait()

            # Wait for output threads
            stdout_thread.join(timeout=5.0)
            stderr_thread.join(timeout=5.0)

            # Log final status
            end_iso = datetime.now().isoformat()
            log_file.write("-" * 60 + "\n")
            log_file.write(f"End Time: {end_iso}\n")
            log_file.write(f"Process finished with exit code: {return_code}\n")
            log_file.flush()

            if return_code == 0:
                logger.info(f"{log_prefix} completed successfully (Exit Code: {return_code}).")
            else:
                logger.error(f"{log_prefix} failed (Exit Code: {return_code}). See log: {log_filename}")

        except Exception as e:
            logger.error(f"{log_prefix}: Error executing job directly: {e}", exc_info=True)
            if log_file:
                try: log_file.write(f"\nCRITICAL ERROR during execution setup: {e}\n")
                except Exception: pass
            return_code = -1 # Ensure failure
        finally:
            # --- Cleanup ---
            if log_file:
                try: log_file.close()
                except Exception: pass
            if temp_script_path and temp_script_path.exists():
                try: temp_script_path.unlink()
                except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")

            # Return True only if the process exit code was explicitly 0
            return return_code == 0


    # --- Job Addition Methods (REVISED) ---
    def add_job(self, script: str, conda_env: Optional[str] = None, args: Optional[List[str]] = None,
                priority: int = 0, allowed_gpus: Optional[List[int]] = None,
                original_line: Optional[str] = None):
        """
        Adds a single job with a unique ID, LLM requirement, and hash to the queue.
        Parses --llm argument to determine GPU requirements from config.
        """
        script_p = Path(script)
        if not script_p.is_file():
            logger.error(f"Script path not found or is not a file: {script}. Job not added.")
            return

        job_id = str(uuid.uuid4())
        log_source = "Manual" if original_line is None else "File"
        args_list = args or [] # Ensure args is a list

        # --- Parse --llm argument and get requirement ---
        llm_name = _extract_arg_value(args_list, '--llm')
        required_gpus = self.get_llm_gpu_requirement(llm_name)
        logger.debug(f"Job Add: Script={script}, LLM='{llm_name}', Determined GPU Req={required_gpus:.2f}")
        # ---------------------------------------------

        # --- Determine final allowed GPUs based ONLY on input parameter ---
        final_allowed_gpus: Optional[List[int]] = None
        if allowed_gpus is not None:
            # Filter out invalid GPU IDs
            valid_gpus_from_param = [gpu_id for gpu_id in allowed_gpus if 0 <= gpu_id < self.num_gpus]
            if not valid_gpus_from_param:
                logger.error(f"No valid GPUs in allowed_gpus list {allowed_gpus} for script {script}. Max GPU ID is {self.num_gpus - 1}. Job not added.")
                return
            # Warn if some were removed
            if len(valid_gpus_from_param) < len(allowed_gpus):
                 logger.warning(f"Invalid GPU IDs removed from allowed_gpus for script {script}. Original: {allowed_gpus}, Valid: {valid_gpus_from_param}")
            final_allowed_gpus = sorted(list(set(valid_gpus_from_param)))
        # -----------------------------------------------------------------

        # --- Hash calculation and tracking for jobs from file ---
        job_hash = None
        if original_line is not None:
            log_source = "File"
            try:
                # Calculate hash including the GPU requirement
                job_hash = self._calculate_job_hash(priority, script, conda_env, args_list, final_allowed_gpus, required_gpus)
                with self.lock:
                    if job_hash in self.managed_job_hashes:
                        existing_job_id = self.hash_to_job_id.get(job_hash, "Unknown")
                        logger.debug(f"Job from line '{original_line.strip()}' (Hash: {job_hash}, ExistingID: {existing_job_id}) is already managed. Skipping add.")
                        return # Don't add duplicates
                    else:
                        self.managed_job_hashes.add(job_hash)
                        self.hash_to_job_id[job_hash] = job_id
                        logger.info(f"Adding job from file (Hash: {job_hash}, JobID: {job_id}, LLM: {llm_name}, Req: {required_gpus:.2f}). Marked as managed.")
            except Exception as e:
                logger.error(f"Error calculating hash or managing state for job from line '{original_line.strip()}': {e}. Job not added.")
                if job_hash:
                     with self.lock:
                         self.managed_job_hashes.discard(job_hash)
                         self.hash_to_job_id.pop(job_hash, None)
                return
        # -------------------------------------------------------

        # --- Add job to queue ---
        # Job tuple: (priority, job_id, script_path, conda_env, args, allowed_gpus,
        #             job_hash, required_gpus, llm_name)
        job_tuple = (priority, job_id, str(script_p), conda_env, args_list, final_allowed_gpus,
                     job_hash, required_gpus, llm_name)
        try:
            self.job_queue.put(job_tuple)
            logger.info(f"Job ID {job_id} ({log_source}, Hash: {job_hash or 'N/A'}, LLM: {llm_name}, Req: {required_gpus:.2f}) added to queue: '{script_p.name}' (Prio: {priority}, GPUs Allowed: {final_allowed_gpus or 'Any'})")
        except Exception as e:
            logger.error(f"Failed to add job ID {job_id} ('{script_p.name}') to queue: {e}")
            if job_hash: # Clean up managed state if queue add failed
                with self.lock:
                    if job_hash in self.managed_job_hashes:
                        self.managed_job_hashes.remove(job_hash)
                        self.hash_to_job_id.pop(job_hash, None)
                        logger.warning(f"Removed hash {job_hash} from managed set due to queue insertion failure.")


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
                original_line_content = line
                line = line.strip()
                if not line or line.startswith('#'): continue

                jobs_processed += 1
                try:
                    # Parse line parts
                    parts = [p.strip() for p in line.split(',', maxsplit=4)]
                    if len(parts) < 2 or not parts[0] or not parts[1]:
                         logger.error(f"{log_prefix} Invalid format line {line_num}: '{line}'.")
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
                        logger.warning(f"{log_prefix} Found '--cuda' in args line {line_num}. It will be ignored/overwritten.")

                    # Parse allowed GPUs
                    allowed_gpus_list = None
                    if allowed_gpus_str:
                        # ... [Copy allowed GPU parsing logic here] ...
                        allowed_gpus_list = [] # Placeholder
                        for part in allowed_gpus_str.replace(" ", "").split(','):
                             # ... (range and single ID parsing)
                             if '-' in part:
                                 try:
                                     start_gpu, end_gpu = map(int, part.split('-'))
                                     if start_gpu <= end_gpu: allowed_gpus_list.extend(range(start_gpu, end_gpu + 1))
                                     else: logger.warning(f"{log_prefix} Invalid range '{part}' line {line_num}")
                                 except ValueError: logger.warning(f"{log_prefix} Invalid range format '{part}' line {line_num}")
                             else:
                                 try: allowed_gpus_list.append(int(part.strip()))
                                 except ValueError: logger.warning(f"{log_prefix} Invalid GPU ID '{part}' line {line_num}")
                    # Validate allowed_gpus_list inside add_job

                    # Calculate hash including requirement
                    current_job_hash = self._calculate_job_hash(priority, script, conda_env, args_list, allowed_gpus_list, required_gpus)
                    hashes_in_current_scan.add(current_job_hash)

                    # Check if managed and add if not
                    is_managed = False
                    with self.lock: is_managed = current_job_hash in self.managed_job_hashes

                    if not is_managed:
                        self.add_job(script, conda_env, args_list, priority, allowed_gpus_list, original_line=original_line_content)
                        with self.lock: # Check if add_job succeeded in adding the hash
                            if current_job_hash in self.managed_job_hashes: jobs_added += 1
                    else:
                        logger.debug(f"{log_prefix} Job line {line_num} (Hash: {current_job_hash}) already managed.")

                except ValueError as ve: logger.error(f"{log_prefix} Invalid number format line {line_num}: '{line}'. {ve}")
                except Exception as e: logger.error(f"{log_prefix} Error parsing line {line_num}: '{line}'. {e}", exc_info=True)

            log_message = f"{log_prefix} Finished processing '{file_path}'. Processed {jobs_processed} lines."
            log_message += f" Added {jobs_added} new jobs."
            logger.info(log_message)

        except FileNotFoundError: logger.error(f"{log_prefix} File disappeared: {file_path}")
        except Exception as e: logger.error(f"{log_prefix} Failed processing {file_path}: {e}", exc_info=True)


    # --- File Monitoring ---
    def _monitor_jobs_file(self):
        """Periodically scans the jobs file and adds new/unmanaged jobs."""
        if not self.jobs_file_path: return
        logger.info(f"Starting job file monitor for '{self.jobs_file_path}' (Interval: {self.file_monitor_interval}s)")
        while not self.stop_event.is_set():
            try:
                logger.debug(f"Checking jobs file: {self.jobs_file_path}")
                self.add_jobs_from_file(str(self.jobs_file_path), initial_load=False)
            except Exception as e:
                logger.error(f"[File Monitor] Error during check of {self.jobs_file_path}: {e}", exc_info=True)

            if self.stop_event.wait(timeout=self.file_monitor_interval): break
        logger.info("Job file monitor stopped.")

    # --- Start/Stop Modifications ---
    def enable_screen(self):
       """Checks if GNU Screen is available and enables its use."""
       try:
           subprocess.run(['screen', '-v'], check=True, capture_output=True, text=True)
           self.use_screen = True
           logger.info("Screen functionality enabled.")
       except Exception as e:
           logger.error(f"Screen check failed: {e}. Screen functionality disabled.")
           self.use_screen = False


# --- Command Line Interface (REVISED) ---
def main():
    # Detect GPUs early
    detected_gpus = 0
    try:
        gpus_list = GPUtil.getGPUs(); detected_gpus = len(gpus_list) if gpus_list else 0
    except Exception as e:
        logger.warning(f"GPUtil detection failed: {e}. Defaulting --gpus may be inaccurate.")
        detected_gpus = 1 # Fallback

    parser = argparse.ArgumentParser(description='GPU Job Scheduler v2.5 (Fractional/Multi-GPU LLM Support)', formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- Start Command ---
    start_parser = subparsers.add_parser('start', help='Start the scheduler daemon process.')
    start_parser.add_argument('--gpus', type=int, default=detected_gpus, help=f'Number of GPUs to manage (default: {detected_gpus} detected)')
    start_parser.add_argument('--jobs-file', type=str, default=DEFAULT_JOBS_FILE, help=f'Path to jobs file for initial load/monitoring (default: {DEFAULT_JOBS_FILE}).')
    # Added LLM config path argument
    start_parser.add_argument('--llm-config', type=str, default=DEFAULT_LLM_CONFIG_FILE, help=f'Path to LLM requirements JSON file (default: {DEFAULT_LLM_CONFIG_FILE}).')
    start_parser.add_argument('--no-monitor', action='store_true', help='Disable dynamic monitoring of the jobs file.')
    start_parser.add_argument('--screen', action='store_true', help='Enable GNU Screen sessions for job execution.')
    start_parser.add_argument('--mem-threshold', type=float, default=0.8, help='GPU memory utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--load-threshold', type=float, default=0.8, help='GPU load utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--monitor-interval', type=int, default=FILE_MONITOR_INTERVAL_S, help=f'Interval (s) to check jobs file (default: {FILE_MONITOR_INTERVAL_S})')
    start_parser.add_argument('--max-assign-attempts', type=int, default=MAX_ASSIGNMENT_ATTEMPTS, help=f'Max attempts to assign job (default: {MAX_ASSIGNMENT_ATTEMPTS})')
    start_parser.add_argument('--assign-retry-wait', type=int, default=ASSIGNMENT_RETRY_WAIT_S, help=f'Base wait (s) between assignment attempts (default: {ASSIGNMENT_RETRY_WAIT_S})')

    # --- Add Command (Manual addition) ---
    add_parser = subparsers.add_parser('add', help='Manually add a new job (requires IPC/restart).')
    add_parser.add_argument('script', type=str, help='Path to the Python script.')
    add_parser.add_argument('--conda', type=str, help='Conda environment name (optional).')
    # Arguments here should include --llm if needed, but NOT --cuda
    add_parser.add_argument('--args', type=str, help='Script arguments (quote if needed). Include --llm <name> if applicable. DO NOT include --cuda.')
    add_parser.add_argument('--priority', type=int, default=0, help='Job priority (lower=higher, default: 0).')
    add_parser.add_argument('--gpus', type=str, help='Allowed GPU IDs (e.g., "0,1", optional).')

    # --- Add File Command ---
    add_file_parser = subparsers.add_parser('add-file', help='Manually add jobs from a file (requires IPC/restart).')
    add_file_parser.add_argument('file_path', type=str, help='Path to the job definition file.')

    # --- Other Commands ---
    subparsers.add_parser('status', help='Show status of GPUs and job queue (limited without IPC).')
    pause_parser = subparsers.add_parser('pause', help='Pause a GPU (modifies state file).')
    pause_parser.add_argument('gpu_id', type=int, help='ID of the GPU to pause.')
    resume_parser = subparsers.add_parser('resume', help='Resume a paused GPU (modifies state file).')
    resume_parser.add_argument('gpu_id', type=int, help='ID of the GPU to resume.')
    subparsers.add_parser('screens', help='List active scheduler GNU Screen sessions.')

    args = parser.parse_args()

    # --- Determine Number of GPUs for non-start commands ---
    default_num_gpus = 8 # Fallback
    try:
        # (Same logic as before to get default_num_gpus from state or detection)
        state_file = Path("gpu_scheduler_state.json")
        if state_file.exists():
             with open(state_file, 'r') as f: state = json.load(f)
             if 'gpu_status' in state and isinstance(state['gpu_status'], list): default_num_gpus = len(state['gpu_status'])
             else: raise ValueError("State file lacks valid 'gpu_status'.")
        else:
             gpus_list = GPUtil.getGPUs()
             if gpus_list: default_num_gpus = len(gpus_list)
             else: logger.warning("State file not found and GPUtil failed. Using fallback GPU count: %d", default_num_gpus)
    except Exception as e: logger.warning("Error determining GPU count for control command: %s. Using fallback: %d", e, default_num_gpus)


    # --- Execute Commands ---
    if args.command == 'start':
        # Determine jobs file path for monitoring vs initial load
        jobs_file_path_arg = args.jobs_file
        monitor_enabled = not args.no_monitor
        jobs_file_to_monitor_path = None
        jobs_file_for_initial_load_path = None

        if jobs_file_path_arg:
             jobs_file_path_obj = Path(jobs_file_path_arg)
             if monitor_enabled:
                 jobs_file_to_monitor_path = jobs_file_path_arg
                 if not jobs_file_path_obj.exists(): logger.warning(f"Jobs file '{jobs_file_path_arg}' not found. Monitoring enabled.")
                 else: logger.info(f"Jobs file '{jobs_file_path_arg}' found. Initial load and monitoring enabled.")
             else: # Monitoring disabled
                 if jobs_file_path_obj.exists():
                     logger.info(f"Jobs file '{jobs_file_path_arg}' found. Monitoring disabled. Performing initial load only.")
                     jobs_file_for_initial_load_path = jobs_file_path_arg
                 else: logger.warning(f"Jobs file '{jobs_file_path_arg}' not found. Monitoring disabled. No initial load.")
        else: logger.info("No jobs file specified.")

        logger.info(f"Starting scheduler: GPUs={args.gpus}, Monitor={monitor_enabled}, LLM Config='{args.llm_config}'")

        scheduler = GPUJobScheduler(
            num_gpus=args.gpus,
            gpu_memory_threshold=args.mem_threshold,
            gpu_load_threshold=args.load_threshold,
            jobs_file_path=jobs_file_to_monitor_path, # Path for monitor thread
            llm_config_path=args.llm_config,         # Path for LLM config
            monitor_interval=args.monitor_interval,
            max_assignment_attempts=args.max_assign_attempts,
            assignment_retry_wait=args.assign_retry_wait
        )
        if args.screen: scheduler.enable_screen()

        # Explicit initial load if monitoring disabled but file exists
        if jobs_file_for_initial_load_path:
             logger.info(f"Performing one-time initial load from: {jobs_file_for_initial_load_path}")
             scheduler.add_jobs_from_file(jobs_file_for_initial_load_path, initial_load=True)

        # Start workers and monitor (start also does initial load if monitor enabled)
        scheduler.start()
        try:
            while True: time.sleep(60) # Keep main thread alive
        except KeyboardInterrupt: logger.info("Ctrl+C received. Initiating shutdown...")
        finally: scheduler.stop(); logger.info("Scheduler shutdown complete.")

    else:
        # --- Control Commands ---
        # Need LLM config path for status display potentially
        llm_config_for_control = args.llm_config if hasattr(args, 'llm_config') else DEFAULT_LLM_CONFIG_FILE
        scheduler_control = GPUJobScheduler(num_gpus=default_num_gpus, llm_config_path=llm_config_for_control)

        if args.command == 'add':
            print("WARNING: 'add' command requires IPC.")
            print("Simulating manual job addition:")
            # Parse args to show potential requirement
            parsed_args = shlex.split(args.args or "")
            llm_name = _extract_arg_value(parsed_args, '--llm')
            req = scheduler_control.get_llm_gpu_requirement(llm_name)
            print(f"  Script: {args.script}, Conda: {args.conda or 'N/A'}, Priority: {args.priority}")
            print(f"  Args: {args.args or 'N/A'} (LLM: {llm_name or 'N/A'}, Estimated Req: {req:.2f})")
            print(f"  Allowed GPUs: {args.gpus or 'Any'}")

        elif args.command == 'add-file':
            print("WARNING: 'add-file' command requires IPC.")
            print(f"Use '--jobs-file' with 'start', or add lines to monitored file.")

        elif args.command == 'status':
            status = scheduler_control.get_gpu_status()
            print("\n--- GPU Status (Allocation/State & Real-time Util) ---")
            print(f"{'GPU ID':<8} {'State':<15} {'Allocation':<12} {'Memory Util':<15} {'Load Util':<15}")
            print("-" * 70)
            for gpu in status:
                 print(f"{gpu['gpu_id']:<8} {gpu['state']:<15} {gpu['allocation']:<12} {gpu['memory_util']:<15} {gpu['load']:<15}")

            print("\n--- Job Queue (Info potentially limited without IPC) ---")
            queued_jobs = scheduler_control.get_job_queue_info()
            if not queued_jobs: print("Local queue instance is empty.")
            else:
                print(f"{'Prio':<6} {'Job ID':<10} {'Hash':<8} {'LLM':<15} {'Req':<5} {'Script':<25} {'Allowed GPUs':<12} {'Arguments'}")
                print("-" * 130)
                for job in queued_jobs:
                    (priority, job_id, script, conda, job_args, allowed_gpus,
                     job_hash, required_gpus, llm_name) = job # Unpack full tuple
                    script_name = Path(script).name if script else "N/A"
                    args_str = shlex.join(job_args) if job_args else ""
                    allowed_gpus_str = ','.join(map(str, allowed_gpus)) if allowed_gpus is not None else 'Any'
                    print(f"{priority:<6} {job_id[:8]:<10} {job_hash[:6] if job_hash else 'N/A':<8} {llm_name or 'N/A':<15} {required_gpus:<5.2f} {script_name:<25} {allowed_gpus_str:<12} {args_str[:35]}")

        elif args.command == 'pause':
            if scheduler_control.pause_gpu(args.gpu_id): print(f"GPU {args.gpu_id} paused.")
            else: print(f"Failed to pause GPU {args.gpu_id}.")

        elif args.command == 'resume':
            if scheduler_control.resume_gpu(args.gpu_id): print(f"GPU {args.gpu_id} resumed.")
            else: print(f"Failed to resume GPU {args.gpu_id}.")

        elif args.command == 'screens':
            print("Listing active scheduler GNU Screen sessions...")
            active_sessions = list_screens()
            if not active_sessions: print("No matching screen sessions found.")
            else:
                print("\n--- Active GPU Job Screen Sessions ---")
                for i, session in enumerate(active_sessions): print(f"{i+1}. {session} (Attach: screen -r {session})")


if __name__ == "__main__":
    main()
