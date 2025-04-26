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
DEFAULT_STATE_FILE = "gpu_scheduler_state.json" # Default state file name
FILE_MONITOR_INTERVAL_S = 30 # Check jobs file every 30 seconds
STATE_CHECK_INTERVAL_S = 20 # Check state file for external changes every 20 seconds
# Directory for screen output logs (raw output including escape codes)
SCREEN_LOG_DIR = Path("/tmp/gpu_scheduler_logs")
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
        self.stop_event: threading.Event = threading.Event() # Event to signal threads to stop
        self.worker_threads: List[threading.Thread] = [] # List of worker threads
        self.paused_gpus: set[int] = set() # Set of GPU IDs that are manually paused (IN MEMORY)
        self.state_file: Path = Path(state_file_path) # File to persist/read GPU status/paused state
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
            os.replace(temp_state_file, self.state_file) # Atomic rename/replace
            logger.debug(f"Scheduler state saved successfully to {self.state_file}.")
        except Exception as e:
            # Critical error if state cannot be saved after allocation
            logger.critical(f"CRITICAL FAILURE: Failed to save state to '{self.state_file}': {e}", exc_info=True)
            # Clean up temp file on error
            if temp_state_file.exists():
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
                                    logger.info(f"Job ID {job_id} re-queued due to worker stop.")
                                except Exception as e:
                                    logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} during stop: {e}. Job may be lost.")
                                break # Exit assignment attempts loop
                        # else: Loop will terminate naturally after max attempts failed

                # --- End of assignment attempt loop ---

                # --- Launch Job or Re-queue ---
                if assigned: # Check if ANY attempt succeeded
                    # Successfully found/reserved GPU(s) and saved state
                    _, run_job_id, _, _, run_args, _, _, run_req_gpus, run_llm_name, _ = job_tuple
                    mode_tag = _extract_and_sanitize_key_arg(run_args)
                    # **** Log the assignment ****
                    logger.info(f"Assigning job ID {run_job_id} (LLM: {run_llm_name}, Req: {run_req_gpus:.2f}) to GPU(s) {assigned_gpu_ids}")

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

            Args:
                job_tuple: The full job tuple including requirements and original_line.
                assigned_gpu_ids: The list of GPU IDs assigned by the worker.
            """
            # Unpack the full tuple including original_line
            (priority, job_id, script_path, conda_env, original_args, _,
             job_hash, required_gpus, llm_name, original_line) = job_tuple
            job_name = Path(script_path).name
            start_time = datetime.now()
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

                if self.use_screen:
                    # --- Screen Mode ---
                    launched_ok = self._run_with_screen(exec_job_tuple, assigned_gpu_ids, env, start_time, mode_tag)
                    if not launched_ok:
                        logger.error(f"GPU(s) {cuda_arg_value}: Screen setup failed for job ID {job_id} (Hash: {job_hash}).")
                        job_completed_successfully = False
                        release_reason = "screen setup failure"
                        # Release GPU immediately if screen launch failed
                        self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, original_line, success=job_completed_successfully, reason=release_reason)
                    # If launch OK, _monitor_screen will call _release_gpu later
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


    # --- Job Release (REVISED for Retry Logic) ---
    def _release_gpu(self, assigned_gpu_ids: List[int], required_gpus: float,
                     job_id: str, job_name: str, start_time: datetime, mode_tag: str,
                     job_hash: Optional[str], original_line: Optional[str], # Added original_line
                     success: bool, reason: str="completion"):
        """
        Helper method to mark GPU allocation as released IN MEMORY and save state.
        Logs original line on completion/failure.
        If job failed and originated from file (has hash), removes hash to allow retry.
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
        log_prefix = f"GPU(s) {gpu_ids_str}: Job ID {job_id} (Hash: {job_hash}, Name: '{job_name}', Mode: {mode_tag})"

        # --- Log Completion with Original Line ---
        completion_log_msg = f"{log_prefix} finished due to {reason} in {duration:.2f} seconds. Success: {success}. Releasing allocation (Req: {required_gpus:.2f})."
        if original_line:
            completion_log_msg += f" [Original Line: {original_line.strip()}]"
        if success:
             logger.info(completion_log_msg)
        else:
             logger.warning(completion_log_msg) # Log failures as warning
        # -----------------------------------------

        save_needed = False
        with self.lock: # Ensure thread safety for modifying status and hashes
            # --- Release Allocation ---
            allocation_to_release = 0.0
            if required_gpus <= (1.0 + GPU_ALLOCATION_PRECISION):
                allocation_to_release = required_gpus
            else:
                allocation_to_release = 1.0 # Release 1.0 from each assigned GPU for multi-GPU jobs

            for gpu_id in assigned_gpu_ids:
                if 0 <= gpu_id < len(self.gpu_status):
                    if self.gpu_status[gpu_id] > GPU_ALLOCATION_PRECISION: # Only release if allocated
                        old_allocation = self.gpu_status[gpu_id]
                        new_allocation = max(0.0, old_allocation - allocation_to_release)
                        self.gpu_status[gpu_id] = new_allocation
                        logger.debug(f"Releasing GPU {gpu_id}: Old Alloc={old_allocation:.2f}, Release={allocation_to_release:.2f}, New Alloc={new_allocation:.2f}")
                        save_needed = True # Mark state save as needed
                    else:
                        logger.debug(f"Skipping release for GPU {gpu_id}: Already has allocation {self.gpu_status[gpu_id]:.2f}")
                else:
                    logger.error(f"Attempted to release allocation for invalid GPU ID {gpu_id}.")
            # -------------------------

            # --- Manage Job Hash for Retries ---
            if job_hash:
                if not success:
                    # If job failed AND it came from the file (has a hash), remove the hash
                    # to allow the file monitor to re-add it for a retry.
                    if job_hash in self.managed_job_hashes:
                         self.managed_job_hashes.remove(job_hash)
                         self.hash_to_job_id.pop(job_hash, None) # Remove mapping too
                         logger.info(f"{log_prefix}: Removed hash '{job_hash}' from managed set due to failure. Job eligible for retry on next monitor scan.")
                         # No state save needed just for hash removal
                    else:
                         # This might happen if the job was manually added or state cleared unexpectedly
                         logger.warning(f"{log_prefix}: Failed job hash '{job_hash}' not found in managed set during release.")
                else:
                    # Job succeeded, keep hash in managed set to prevent re-run
                    logger.debug(f"{log_prefix}: Hash '{job_hash}' remains in managed set after successful job completion.")
            # --- End Hash Management ---

            # --- Save State if Allocation Changed ---
            if save_needed:
                try:
                    self.save_state() # Save updated GPU allocation status
                except Exception as e:
                    # Log critical error, but continue - state might be inconsistent
                    logger.critical(f"CRITICAL FAILURE: Failed to save state during GPU release for job {job_id}: {e}", exc_info=True)
            # --------------------------------------


    # --- Run with Screen (MODIFIED TO USE `script` command) ---
    def _run_with_screen(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: Dict, start_time: datetime, mode_tag: str) -> bool:
        """
        Sets up and launches a job inside a GNU Screen session on assigned GPUs.
        Uses the `script` command to capture output to a log file while preserving
        pseudo-terminal behavior for the executed command (helps with progress bars).
        Args include modified '--cuda' and full job details including original_line.

        Returns:
            bool: True if the screen session was launched successfully, False otherwise.
        """
        # Unpack the full tuple including original_line
        (priority, job_id, script_path, conda_env, args_with_cuda, _,
         job_hash, required_gpus, llm_name, original_line) = job_tuple
        script_p = Path(script_path)
        script_basename = script_p.name
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))

        session_name = f"gpujob_{gpu_ids_str}_{mode_tag}_{job_id[:6]}_{start_time.strftime('%H%M%S')}"
        session_name = session_name[:60] # Limit length

        # --- File for `script` command output ---
        # Note: This log will contain raw terminal escape codes.
        job_output_log = SCREEN_LOG_DIR / f"output_{session_name}.typescript" # Use .typescript extension convention
        job_output_log.unlink(missing_ok=True) # Clear previous log if exists
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
                f"# Original Line: {original_line.strip() if original_line else 'N/A'}", # Add original line comment
                f"# Job Output Log (typescript): {job_output_log}",
                "echo \"--- Starting Job Execution Script ---\"",
                "echo \"Timestamp: $(date)\"",
                "echo \"Running as user: $(whoami)\"",
                "echo \"Current directory: $(pwd)\"",
                f"echo \"Assigned CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not Set')}\"", # Log assigned devices
                "set -e", # Exit on error (mostly for setup part)
                # pipefail might not be needed if not piping the main command anymore
                # "set -o pipefail"
            ]

            # --- Conda activation logic (Robust path finding - same as before) ---
            if conda_env:
                conda_base_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh" # Default guess
                conda_path_found = False
                conda_prefix = os.environ.get('CONDA_PREFIX')
                potential_conda_sh = None
                if conda_prefix: # Check relative to current env
                    conda_base_dir = Path(conda_prefix).parent.parent
                    potential_conda_sh = conda_base_dir / "etc/profile.d/conda.sh"
                    if potential_conda_sh.exists():
                        conda_base_cmd = f"source {shlex.quote(str(potential_conda_sh))}"
                        conda_path_found = True
                if not conda_path_found: # Check common default locations
                    home_dir = os.environ.get('HOME', '/root') # Get home directory
                    for default_path in [f"{home_dir}/anaconda3/etc/profile.d/conda.sh",
                                         f"{home_dir}/miniconda3/etc/profile.d/conda.sh",
                                         "/opt/conda/etc/profile.d/conda.sh"]:
                        potential_conda_sh = Path(default_path)
                        if potential_conda_sh.exists():
                            conda_base_cmd = f"source {shlex.quote(str(potential_conda_sh))}"
                            conda_path_found = True
                            break
                # if not conda_path_found: # Logged by scheduler already
                    # logger.warning(...)

                script_content.extend([
                    f"echo 'Attempting to initialize conda using: {conda_base_cmd}'",
                    conda_base_cmd,
                    "conda_init_exit_code=$?",
                    "if [ $conda_init_exit_code -ne 0 ]; then echo \"WARNING: Conda init command exited with code $conda_init_exit_code (may be benign)\" >&2; fi",
                    f"echo 'Activating conda environment: {conda_env}'",
                    f"conda activate {shlex.quote(conda_env)}",
                    "conda_activate_exit_code=$?",
                    "if [ $conda_activate_exit_code -ne 0 ]; then echo \"ERROR: Conda activate '{conda_env}' failed with code $conda_activate_exit_code\" >&2; exit $conda_activate_exit_code; fi",
                    "echo 'Conda environment activated.'",
                    "echo \"PATH: $PATH\"", "echo \"CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV\"",
                    "echo \"Which Python: $(which python)\"", "echo \"Python Version: $(python --version)\"",
                ])
            else:
                script_content.append("echo 'No conda environment specified.'")

            # Prepare Python command execution using args_with_cuda (which has modified --cuda)
            python_cmd_list = ["python", "-u", script_path] # -u for unbuffered python output
            if args_with_cuda:
                try: str_args = [str(arg) for arg in args_with_cuda]; python_cmd_list.extend(str_args)
                except Exception as e_args: raise ValueError(f"Invalid screen arguments for job {job_id}") from e_args
            python_cmd_str = shlex.join(python_cmd_list)

            # --- Use `script` command for execution and logging ---
            # script -q: quiet (no start/done messages from script itself)
            # script -e: return exit code of the command
            # script -c "command": execute the command string
            # script logfile: the file to write the typescript to
            script_command_str = f"script -q -e -c {shlex.quote(python_cmd_str)} {shlex.quote(str(job_output_log))}"

            script_content.extend([
                f"echo 'Executing Python command via script: {script_command_str}'",
                f"echo 'Output being captured to: {job_output_log}'",
                script_command_str, # Execute the script command
                "exit_code=$?", # Capture the exit code returned by `script -e`
                f"echo \"Script command finished with exit code: $exit_code\"",
            ])
            # --- End `script` execution ---

            # --- Record results (based on exit code from `script`) ---
            script_content.extend([
                f"if [ $exit_code -eq 0 ]; then",
                f"  echo 'Command reported success (Exit Code: 0).'",
                f"else",
                f"  echo 'Command reported failure (Exit Code: $exit_code).'",
                f"fi",
                f"echo \"--- Ending Job Execution Script --- Timestamp: $(date)\"",
                f"exit $exit_code" # Exit the wrapper script with the python script's exit code
            ])

            # Write and execute the wrapper script
            temp_script_dir = Path("/tmp")
            temp_script_path = temp_script_dir / f"run_{session_name}_{job_id[:4]}.sh"
            with open(temp_script_path, 'w') as f: f.write("\n".join(script_content))
            os.chmod(temp_script_path, 0o755)
            logger.debug(f"GPU(s) {gpu_ids_str}: Temp script for job ID {job_id}: {temp_script_path}")

            # --- Start screen session ---
            # The screen session now just runs the wrapper script
            screen_cmd = ['screen', '-dmS', session_name, str(temp_script_path)]
            logger.info(f"GPU(s) {gpu_ids_str}: Starting screen session '{session_name}' for job ID {job_id}")
            # Check if `script` command exists before trying to launch
            try:
                subprocess.run(['script', '--version'], check=True, capture_output=True, text=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as script_err:
                 logger.error(f"CRITICAL: `script` command not found or failed check. Cannot launch job {job_id} in screen with TTY preservation. Error: {script_err}")
                 # Clean up temp wrapper script if created
                 if temp_script_path and temp_script_path.exists():
                     try: temp_script_path.unlink(missing_ok=True)
                     except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")
                 return False # Launch failure

            process = subprocess.run(screen_cmd, env=env, check=True, capture_output=True, text=True)
            logger.info(f"GPU(s) {gpu_ids_str}: Screen session launched. To view progress: screen -r {session_name}")
            logger.info(f"GPU(s) {gpu_ids_str}: Job output log (typescript): {job_output_log}")
            logger.warning(f"GPU(s) {gpu_ids_str}: Note: Log file '{job_output_log}' contains raw terminal escape codes.")

            # --- Start monitoring thread ---
            # Pass the full original job_tuple so monitor has original_line for release log
            monitoring_thread = threading.Thread(
                target=self._monitor_screen,
                args=(session_name, job_id, script_path, assigned_gpu_ids, required_gpus, temp_script_path, start_time, mode_tag, job_hash, original_line, job_output_log), # Pass log path for potential cleanup info
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
             # Clean up temp files if created
             if temp_script_path and temp_script_path.exists():
                 try: temp_script_path.unlink(missing_ok=True)
                 except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")
             job_output_log.unlink(missing_ok=True) # Clean up log file
             return False # Launch failure
        except Exception as e:
            gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
            logger.error(f"GPU(s) {gpu_ids_str}: Error setting up or launching screen session for job ID {job_id} (Hash: {job_hash}): {e}", exc_info=True)
            # Cleanup temp files if created
            if temp_script_path and temp_script_path.exists():
                try: temp_script_path.unlink(missing_ok=True)
                except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")
            # Cleanup log file if created during failed setup
            job_output_log.unlink(missing_ok=True)
            return False # Launch failure


    # --- Monitor Screen (Takes log path for info) ---
    def _monitor_screen(self, session_name: str, job_id: str, script_path: str,
                        assigned_gpu_ids: List[int], required_gpus: float,
                        temp_script_path: Optional[Path], start_time: datetime, mode_tag: str,
                        job_hash: Optional[str], original_line: Optional[str], job_output_log: Path): # Added job_output_log
        """
        Monitors a specific screen session until it terminates or the scheduler stops.
        Determines job success based on whether the session ended before scheduler stop.
        Cleans up temporary files and calls _release_gpu.
        NOTE: Does not check the actual exit code of the job anymore directly,
              relies on the wrapper script's exit code captured by `script -e`.
              Success is primarily inferred from the session ending naturally.
        """
        job_name = Path(script_path).name
        gpu_ids_str = ",".join(map(str, sorted(assigned_gpu_ids)))
        log_prefix = f"GPU(s) {gpu_ids_str}: Job ID {job_id} (Hash: {job_hash}, Screen: {session_name}, Mode: {mode_tag})"
        logger.info(f"{log_prefix} Monitoring screen session... Log: {job_output_log}")
        active = True
        session_ended_naturally = False # Track if session ended before stop event
        check_interval = 15 # Seconds between checks

        # Loop while the screen session appears active and the scheduler hasn't stopped
        while active and not self.stop_event.is_set():
            try:
                # Use regex to check for exact session name match, avoiding pid prefix issues
                cmd = f"screen -ls | grep -qE '^\\s*[0-9]+\\.{re.escape(session_name)}\\s+\\(' "
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                active = True
                logger.debug(f"{log_prefix} Screen session is active.")
            except subprocess.CalledProcessError:
                active = False
                session_ended_naturally = True # Screen session ended on its own
                logger.info(f"{log_prefix} Screen session finished or not found.")
            except FileNotFoundError:
                 logger.error(f"{log_prefix} 'screen' command not found during monitoring.")
                 active = False
                 session_ended_naturally = False # Cannot determine outcome
            except Exception as e:
                logger.error(f"{log_prefix} Error checking screen session: {e}. Assuming finished.")
                active = False
                session_ended_naturally = False # Cannot determine outcome

            if active:
                if self.stop_event.wait(timeout=check_interval):
                    logger.warning(f"{log_prefix} Stop event received during monitoring.")
                    active = False # Stop monitoring
                    session_ended_naturally = False # Scheduler stopped it
                    # Proceed to cleanup

        # --- Screen session ended or scheduler stopped ---
        logger.info(f"{log_prefix} Monitoring finished.")

        # Determine final success status based only on whether the session ended before stop event
        # We rely on the wrapper script + `script -e` to handle the actual exit code,
        # but for the scheduler's perspective, ending naturally is the primary success indicator.
        final_success = session_ended_naturally
        if final_success:
             logger.info(f"{log_prefix} Screen session ended naturally. Assuming success based on session termination.")
        else:
             if not self.stop_event.is_set(): # If stop event isn't set, but session didn't end naturally (e.g., error checking screen)
                  logger.warning(f"{log_prefix} Screen session did not end naturally or status check failed. Assuming failure.")
             else: # Stop event was set
                  logger.info(f"{log_prefix} Scheduler stop event triggered during monitoring. Assuming failure.")

        # Clean up the temporary wrapper script
        if temp_script_path and temp_script_path.exists():
            logger.debug(f"{log_prefix} Removing temporary script: {temp_script_path}")
            try: temp_script_path.unlink()
            except OSError as e: logger.warning(f"{log_prefix} Error removing temp script {temp_script_path}: {e}")

        # Determine the reason string for the release log message
        reason = f"screen session ended (Success inferred: {final_success})"
        if not session_ended_naturally and self.stop_event.is_set(): # Check if stop event occurred *and* session didn't end naturally
            reason = f"scheduler shutdown during screen session (Success inferred: {final_success})"

        # --- Call _release_gpu from the monitor thread ---
        # Pass original_line for logging
        self._release_gpu(assigned_gpu_ids, required_gpus, job_id, job_name, start_time, mode_tag, job_hash, original_line, final_success, reason)


    # --- Run Directly ---
    def _run_directly(self, job_tuple: Tuple, assigned_gpu_ids: List[int], env: Dict, start_time: datetime, mode_tag: str) -> bool:
        """
        Runs a job directly as a subprocess on its assigned GPU(s).
        Args include modified '--cuda'. Logs original line in header.

        Returns:
            bool: True if the job script exited with code 0, False otherwise.
        """
        # Unpack the full tuple including original_line
        (priority, job_id, script_path, conda_env, args_with_cuda, _,
         job_hash, required_gpus, llm_name, original_line) = job_tuple
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
            log_file.write(f"Original Line: {original_line.strip() if original_line else 'N/A'}\n") # Add original line
            log_file.write(f"LLM Name: {llm_name or 'N/A'}\n")
            log_file.write(f"GPUs Required: {required_gpus:.2f}\n")
            log_file.write(f"Mode Tag: {mode_tag}\n")
            log_file.write(f"Script: {script_path}\n")
            log_file.write(f"Conda Env: {conda_env or 'None'}\n")
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
                conda_base_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh" # Default guess
                conda_path_found = False
                conda_prefix = os.environ.get('CONDA_PREFIX')
                potential_conda_sh = None
                if conda_prefix: # Check relative to current env
                    conda_base_dir = Path(conda_prefix).parent.parent
                    potential_conda_sh = conda_base_dir / "etc/profile.d/conda.sh"
                    if potential_conda_sh.exists():
                        conda_base_cmd = f"source {shlex.quote(str(potential_conda_sh))}"
                        conda_path_found = True
                if not conda_path_found: # Check common default locations
                    home_dir = os.environ.get('HOME', '/root') # Get home directory
                    for default_path in [f"{home_dir}/anaconda3/etc/profile.d/conda.sh",
                                         f"{home_dir}/miniconda3/etc/profile.d/conda.sh",
                                         "/opt/conda/etc/profile.d/conda.sh"]:
                        potential_conda_sh = Path(default_path)
                        if potential_conda_sh.exists():
                            conda_base_cmd = f"source {shlex.quote(str(potential_conda_sh))}"
                            conda_path_found = True
                            break
                # if not conda_path_found: logger.warning(...) # Logged by scheduler
                # --- End Conda Logic ---

                shell_script_lines.extend([
                    conda_base_cmd,
                    "conda_init_exit_code=$?",
                    "if [ $conda_init_exit_code -ne 0 ]; then echo \"WARNING: Conda init command exited with code $conda_init_exit_code\" >&2; fi", # Log warning only
                    f"conda activate {shlex.quote(conda_env)}",
                    "conda_activate_exit_code=$?",
                    "if [ $conda_activate_exit_code -ne 0 ]; then echo \"ERROR: Conda activate '{conda_env}' failed with code $conda_activate_exit_code\" >&2; exit $conda_activate_exit_code; fi", # Fail if activation fails
                ])

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
                shell_mode = False # Execute the script directly, not via shell
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
                text=True, bufsize=1, shell=shell_mode # Ensure shell=False unless using shell features explicitly
            )

            # --- Real-time output handling ---
            def stream_output(pipe, prefix, log_f, logger_func):
                # Reads lines from the process's stdout or stderr pipe
                # Logs them with a prefix and writes to the job-specific log file
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

            # Wait for output threads to finish logging remaining output
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


    # --- Job Addition Methods ---
    def add_job(self, script: str, conda_env: Optional[str] = None, args: Optional[List[str]] = None,
                priority: int = 0, allowed_gpus: Optional[List[int]] = None,
                original_line: Optional[str] = None): # Added original_line parameter
        """
        Adds a single job with a unique ID, LLM requirement, hash, and original line to the queue.
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
                        logger.debug(f"Job from line '{original_line.strip()}' (Hash: {job_hash}, ExistingID: {existing_job_id}) is already managed for this run. Skipping add.")
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
        #             job_hash, required_gpus, llm_name, original_line) # Added original_line
        job_tuple = (priority, job_id, str(script_p), conda_env, args_list, final_allowed_gpus,
                     job_hash, required_gpus, llm_name, original_line) # Pass original_line here
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

    parser = argparse.ArgumentParser(description='GPU Job Scheduler v2.12 (Screen list fix)', formatter_class=argparse.RawTextHelpFormatter) # Version bump
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
    add_parser.add_argument('--args', type=str, help='Script arguments (quote if needed). Include --llm <name> if applicable. DO NOT include --cuda.')
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
