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

# --- Constants ---
DEFAULT_JOBS_FILE = "jobs.txt"
FILE_MONITOR_INTERVAL_S = 30 # Check jobs file every 30 seconds
MARKER_FILE_DIR = Path("/tmp/gpu_scheduler_markers") # Directory for screen success markers
MAX_ASSIGNMENT_ATTEMPTS = 5 # Max times a worker tries to assign a job before requeuing
ASSIGNMENT_RETRY_WAIT_S = 5 # Base wait time between assignment attempts

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

def _extract_and_sanitize_key_arg(args: Optional[List[str]], key: str = '--mode') -> str:
    """
    Extracts a specific argument value (defaulting to '--mode') from a list of arguments
    and sanitizes it for use in filenames or tags.
    """
    value = "unknown" # Default if key not found or no args
    if args:
        try:
            key_index = args.index(key)
            if key_index + 1 < len(args):
                value = args[key_index + 1]
        except ValueError:
            # Key argument not found in the list
            pass # Keep default "unknown"

    # Sanitize the value: replace non-alphanumeric with underscores
    sanitized_value = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value))
    # Optional: Truncate if values can be very long
    max_len = 20
    return sanitized_value[:max_len]

# --- Main Scheduler Class ---
class GPUJobScheduler:
    """
    Manages a queue of GPU jobs, assigns them to available GPUs based on load
    and memory thresholds, supports dynamic job file monitoring, and can run
    jobs directly or within GNU Screen sessions. Automatically adds '--cuda <gpu_id>'
    to the executed script's arguments.
    """
    def __init__(self,
                 num_gpus: int = 8,
                 gpu_memory_threshold: float = 0.8,
                 gpu_load_threshold: float = 0.8,
                 jobs_file_path: Optional[str] = None, # Path to jobs file
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
            monitor_interval: Interval in seconds to check the jobs file.
            max_assignment_attempts: Max times a worker tries to assign a job before requeuing.
            assignment_retry_wait: Base wait time (s) between assignment attempts.
        """
        self.use_screen: bool = False # Whether to use GNU Screen for jobs
        self.num_gpus: int = num_gpus
        # Job tuple format: (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash)
        # Note: 'args' here are the *original* args from the job file/add command.
        # The '--cuda' flag is added *later* during execution.
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.gpu_status: List[bool] = [False] * num_gpus # False = available, True = busy
        self.gpu_memory_threshold: float = gpu_memory_threshold
        self.gpu_load_threshold: float = gpu_load_threshold
        self.lock: threading.Lock = threading.Lock() # Lock for shared resources (queue, status, hashes, paused_gpus)
        self.stop_event: threading.Event = threading.Event() # Event to signal threads to stop
        self.worker_threads: List[threading.Thread] = [] # List of worker threads
        self.paused_gpus: set[int] = set() # Set of GPU IDs that are manually paused
        self.state_file: Path = Path("gpu_scheduler_state.json") # File to persist GPU status/paused state
        self.max_assignment_attempts = max_assignment_attempts
        self.assignment_retry_wait = assignment_retry_wait

        # --- Attributes for dynamic job file handling ---
        self.jobs_file_path: Optional[Path] = Path(jobs_file_path) if jobs_file_path else None
        self.file_monitor_interval: int = monitor_interval
        self.file_monitor_thread: Optional[threading.Thread] = None
        # Stores hashes of jobs from the file that are queued, running, or failed (awaiting rescan)
        # This prevents re-adding the same job line multiple times while it's being processed.
        self.managed_job_hashes: Set[str] = set()
        # Optional: Track job_id associated with a hash for better logging/cleanup on stop
        self.hash_to_job_id: Dict[str, str] = {}
        # ----------------------------------------------------

        self.load_state() # Load previous state (paused/busy GPUs)
        self._apply_paused_state() # Apply the loaded paused state (log info)

    def _calculate_job_hash(self, priority: int, script: str, conda_env: Optional[str], args: Optional[List[str]], allowed_gpus: Optional[List[int]]) -> str:
        """
        Calculates a stable MD5 hash for a job definition based on its parameters
        as defined in the job source (file or add command). Does NOT include
        the runtime-added '--cuda' argument.
        """
        hasher = hashlib.md5()
        hasher.update(str(priority).encode())
        hasher.update(str(script).encode())
        hasher.update(str(conda_env or '').encode())
        # Hash based on the *original* arguments provided
        hasher.update(str(sorted(args) if args else []).encode())
        hasher.update(str(sorted(allowed_gpus) if allowed_gpus else []).encode()) # Sort gpus
        return hasher.hexdigest()

    # --- State Management ---
    def _apply_paused_state(self):
        """Logs the currently paused GPUs after loading state."""
        logger.info(f"Applying paused state for GPUs: {self.paused_gpus}")

    def load_state(self):
        """Loads GPU busy/paused status from the state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # --- Load gpu_status ---
                loaded_status = state.get('gpu_status', [])
                # Adjust loaded status to match current num_gpus configuration
                if len(loaded_status) == self.num_gpus:
                    self.gpu_status = loaded_status
                elif len(loaded_status) < self.num_gpus:
                    # Add newly detected GPUs as available
                    self.gpu_status = loaded_status + [False] * (self.num_gpus - len(loaded_status))
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Added new GPUs as AVAILABLE.")
                else:
                    # Ignore extra GPUs from old state if num_gpus decreased
                    self.gpu_status = loaded_status[:self.num_gpus]
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Ignoring extra GPUs from state.")

                # --- Load paused GPUs ---
                loaded_paused = state.get('paused_gpus', [])
                # Ensure paused GPUs are valid within the current range
                self.paused_gpus = set(gpu_id for gpu_id in loaded_paused if 0 <= gpu_id < self.num_gpus)

                logger.info(f"State loaded successfully. Status: {self.gpu_status}, Paused: {self.paused_gpus}")
            except Exception as e:
                logger.error(f"Failed to load state from '{self.state_file}': {e}. Starting fresh.")
                self.gpu_status = [False] * self.num_gpus
                self.paused_gpus = set()
        else:
            logger.info("No state file found. Starting with all GPUs available.")
            self.gpu_status = [False] * self.num_gpus
            self.paused_gpus = set()
        # Note: We don't persist managed_job_hashes. They are rebuilt on restart by scanning the file.

    def save_state(self):
        """Saves the current GPU busy/paused status to the state file."""
        with self.lock: # Ensure thread safety when accessing shared state
            try:
                state = {
                    "gpu_status": list(self.gpu_status),
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
        """Returns the current status (AVAILABLE, BUSY, PAUSED) and utilization of each GPU."""
        status_list = []
        gpus_util = []
        try:
            # Get real-time utilization stats
            gpus_util = GPUtil.getGPUs()
        except Exception as e:
            logger.error(f"Could not get GPU utilization via GPUtil: {e}")

        # Get scheduler's view of state under lock
        with self.lock:
            current_gpu_status = list(self.gpu_status)
            current_paused_gpus = set(self.paused_gpus)

        for gpu_id in range(self.num_gpus):
            # Determine state based on scheduler's knowledge
            state = "PAUSED" if gpu_id in current_paused_gpus else \
                    "BUSY" if current_gpu_status[gpu_id] else \
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
                "gpu_id": gpu_id, "state": state,
                "memory_util": memory_util_str, "load": load_str
            })
        return status_list

    def get_job_queue_info(self) -> List[Tuple]:
        """Returns a sorted list of jobs currently in the priority queue."""
        # Note: This shows the queue of the current instance. For CLI status, IPC would be needed.
        with self.job_queue.mutex: # Lock the queue while accessing its internal list
            # Sort by priority (lower first), then job_id (FIFO tie-breaker)
            # Job tuple: (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash)
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
        # Create one worker thread per GPU (or adjust as needed)
        num_workers = self.num_gpus
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
                # Sentinel: (priority, job_id=None, script_path=None, ..., job_hash=None)
                # Use lowest priority to ensure it's processed after real jobs if queue isn't empty
                self.job_queue.put((float('inf'), None, None, None, None, None, None))
            except Exception as e:
                logger.error(f"Error putting sentinel value {i+1}/{num_sentinels} in queue: {e}")

        logger.info("Waiting for worker threads to finish...")
        for thread in self.worker_threads:
            thread.join(timeout=10.0) # Wait with timeout
            if thread.is_alive():
                logger.warning(f"Worker thread {thread.name} did not finish within timeout.")

        # --- Clean up managed hashes for jobs interrupted mid-run ---
        # Simple approach: Clear all managed hashes on stop.
        # On restart, the monitor will re-add any jobs still present in jobs.txt.
        logger.info("Clearing internal managed job tracking for restart.")
        with self.lock:
            self.managed_job_hashes.clear()
            self.hash_to_job_id.clear()
        # --------------------------------------------------------------

        logger.info("Saving final scheduler state (GPU status only)...")
        self.save_state() # Save only GPU busy/paused status
        logger.info("Scheduler stopped.")

    # --- Worker Logic (REVISED) ---
    def worker(self):
        """Worker thread logic: Get job, find suitable GPU, launch job runner."""
        thread_name = threading.current_thread().name
        logger.info(f"Worker started.")

        while not self.stop_event.is_set():
            job_tuple = None
            job_id = None
            job_hash = None
            script_path = None
            got_task = False # Track if we successfully got a task (job or sentinel)

            try:
                # Get job: (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash)
                # Block with timeout to allow checking stop_event periodically
                job_tuple = self.job_queue.get(block=True, timeout=1.0)
                got_task = True # We got something from the queue
                priority, job_id, script_path, _, original_args, allowed_gpus, job_hash = job_tuple

                # Check for sentinel value (job_id is None) used for stopping
                if job_id is None:
                    logger.info(f"Received sentinel, exiting.")
                    break # Exit the worker loop

                logger.debug(f"Retrieved job ID {job_id} (Hash: {job_hash}): {script_path} (Priority: {priority})")

                assigned = False
                attempts = 0

                # Loop to find a suitable GPU for the *current* job
                while not assigned and not self.stop_event.is_set() and attempts < self.max_assignment_attempts:
                    gpu_id_to_run = -1
                    with self.lock: # Lock needed to check/update shared self.gpu_status and self.paused_gpus
                        gpu_indices = list(range(self.num_gpus)) # Check all GPUs
                        for gpu_id in gpu_indices:
                            # --- Start Detailed Check Logging ---
                            logger.debug(f"GPU Check [{gpu_id}] attempting for Job ID {job_id} (Hash: {job_hash})")
                            # Re-unpack current job details inside loop (already done above)
                            _, current_job_id, _, _, current_args, current_allowed_gpus, current_job_hash = job_tuple

                            # Check basic availability and constraints
                            is_busy = self.gpu_status[gpu_id]
                            is_paused = gpu_id in self.paused_gpus
                            is_allowed = (current_allowed_gpus is None or gpu_id in current_allowed_gpus)

                            if not is_busy and not is_paused and is_allowed:
                                # If basic checks pass, check resource utilization
                                try:
                                    # Get fresh GPU stats (can be slow, consider optimizing if contention is high)
                                    gpus_util = GPUtil.getGPUs()
                                    if gpu_id < len(gpus_util):
                                        gpu = gpus_util[gpu_id]
                                        mem_util = gpu.memoryUtil
                                        load_util = gpu.load
                                        passes_mem = mem_util < self.gpu_memory_threshold
                                        passes_load = load_util < self.gpu_load_threshold

                                        logger.debug(f"GPU Check [{gpu_id}]: Busy={is_busy}, Paused={is_paused}, Allowed={is_allowed}, Mem={mem_util:.2f}<{self.gpu_memory_threshold}({passes_mem}), Load={load_util:.2f}<{self.gpu_load_threshold}({passes_load})")

                                        # If resource thresholds are met, assign the GPU
                                        if passes_mem and passes_load:
                                            mode_tag = _extract_and_sanitize_key_arg(current_args) # Get mode for logging
                                            logger.info(f"Found suitable GPU {gpu_id} for job ID {current_job_id} (Hash: {current_job_hash}, Mode: {mode_tag}).")
                                            self.gpu_status[gpu_id] = True # Mark busy *within lock*
                                            gpu_id_to_run = gpu_id
                                            break # Exit GPU check loop (inner for loop) since we found one
                                        else:
                                            logger.debug(f"GPU Check [{gpu_id}]: Failed resource threshold.")
                                    else:
                                        # GPUtil didn't report on this GPU ID (shouldn't happen if num_gpus is correct)
                                        logger.warning(f"GPUtil Check [{gpu_id}]: Did not report on GPU. Skipping.")
                                except Exception as e:
                                    # Handle errors during GPUtil check
                                    logger.error(f"GPUtil Check [{gpu_id}]: Error during check for Job ID {current_job_id}: {e}. Skipping.", exc_info=True)
                                    continue # Skip this GPU check on error
                            else:
                                # Log why the GPU was skipped before resource check
                                reason_skip = []
                                if is_busy: reason_skip.append("Busy")
                                if is_paused: reason_skip.append("Paused")
                                if not is_allowed: reason_skip.append("NotAllowed")
                                logger.debug(f"GPU Check [{gpu_id}]: Skipped ({', '.join(reason_skip)})")
                            # --- End Detailed Check Logging ---

                    # --- After checking all GPUs (outside lock) ---
                    if gpu_id_to_run != -1:
                        # Successfully found and reserved a GPU
                        self.save_state() # Save state reflecting the busy GPU
                        # Unpack details for the job runner thread
                        _, run_job_id, run_script_path, _, run_args, _, run_job_hash = job_tuple
                        mode_tag = _extract_and_sanitize_key_arg(run_args)
                        logger.info(f"Assigning job ID {run_job_id} (Hash: {run_job_hash}, Script: {run_script_path}, Mode: {mode_tag}) to GPU {gpu_id_to_run}")

                        # Start a new thread to run the job, detaching it from the worker
                        job_runner_thread = threading.Thread(
                            target=self._run_job,
                            args=(job_tuple, gpu_id_to_run), # Pass the original job tuple and assigned GPU
                            name=f"JobRunner-GPU{gpu_id_to_run}-{mode_tag}-{run_job_id[:4]}", # Descriptive name
                            daemon=True # Allow scheduler exit even if job runners are stuck
                        )
                        job_runner_thread.start()
                        assigned = True # Mark job as assigned by this worker

                    else:
                        # No suitable GPU found in this attempt
                        attempts += 1
                        logger.debug(f"Found no suitable GPU for job ID {job_id} (Hash: {job_hash}) (Attempt {attempts}/{self.max_assignment_attempts}). Will wait and retry.")

                        # Wait *outside* the lock before next attempt
                        # Use exponential backoff for waiting time, capped
                        wait_time = min(self.assignment_retry_wait * (attempts), 30)
                        # Make the sleep interruptible by the stop event
                        if self.stop_event.wait(timeout=wait_time):
                            logger.info(f"Stop event received while waiting to assign job {job_id}. Re-queueing.")
                            # Put the job back because we are stopping this worker/scheduler
                            try:
                                self.job_queue.put(job_tuple)
                                logger.info(f"Job ID {job_id} re-queued due to worker stop.")
                                # Do NOT mark as assigned
                            except Exception as e:
                                logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} during stop: {e}. Job may be lost.")
                            # Break from the assignment loop, the outer loop will check stop_event again
                            break

                # --- End of assignment attempt loop ---
                if not assigned:
                    # This block is reached if stop_event interrupted the wait OR max_attempts was reached
                    if attempts >= self.max_assignment_attempts:
                        # Max attempts reached, put the job back
                        logger.warning(f"Exceeded max attempts ({self.max_assignment_attempts}) to find GPU for job ID {job_id} (Hash: {job_hash}). Re-queueing job.")
                        try:
                            self.job_queue.put(job_tuple)
                        except Exception as e:
                            logger.critical(f"CRITICAL: Failed to re-queue job ID {job_id} after max attempts: {e}. Job may be lost.")
                            got_task = False # If requeue fails, we didn't really process the task
                    # If stop_event caused the loop break, the job was already re-queued inside the loop

                # Check stop event again before trying to get the next job
                if self.stop_event.is_set():
                    logger.info("Stop event detected after assignment loop. Exiting worker.")
                    break # Exit the main worker loop

            except queue.Empty:
                # Queue was empty during the timeout, just loop again and wait
                got_task = False # Didn't get a task this time
                continue
            except Exception as e:
                # Catch unexpected errors in the main worker loop after getting a job
                logger.error(f"Error in worker main loop processing job ID {job_id or 'N/A'} (Hash: {job_hash or 'N/A'}): {e}", exc_info=True)
                # Assume the task we got caused an error, mark it done to avoid blocking join()
                got_task = True
            finally:
                # Crucial: Mark the task as done *if* we successfully got one from the queue.
                # This is essential for queue.join() to work correctly during shutdown.
                # Call task_done whether the job was assigned, re-queued, or caused an error after retrieval.
                if got_task:
                    try:
                        self.job_queue.task_done()
                        logger.debug(f"task_done() called for job ID {job_id} (Hash: {job_hash})")
                    except ValueError:
                        # This error means task_done() was called more times than get()
                        logger.error(f"CRITICAL: task_done() called too many times for job ID {job_id} (Hash: {job_hash})!")
                    except Exception as e:
                        logger.error(f"Error calling task_done() for job ID {job_id} (Hash: {job_hash}): {e}")

        logger.info(f"Worker finished.")


    # --- Job Execution ---
    def _run_job(self, job_tuple: Tuple, gpu_id: int):
            """
            Internal method called by a dedicated thread to prepare environment
            and execute a single job either directly or using screen.
            Adds '--cuda <gpu_id>' to the arguments passed to the script.
            Ensures the GPU is released afterwards.

            Args:
                job_tuple: The original job tuple (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash).
                gpu_id: The ID of the GPU assigned by the worker.
            """
            priority, job_id, script_path, conda_env, original_args, _, job_hash = job_tuple # Get original args
            job_name = Path(script_path).name
            start_time = datetime.now()
            mode_tag = _extract_and_sanitize_key_arg(original_args) # Extract mode from original args

            # Prepare arguments for the actual execution: original args + scheduler-added --cuda flag
            args_for_exec = (original_args or []) + ["--cuda", str(gpu_id)]

            # 'launched_ok' tracks if the process/screen started without immediate error.
            launched_ok = False
            # 'job_completed_successfully' tracks the final outcome of the job script itself.
            job_completed_successfully = False
            # Reason for releasing the GPU (completion, error, shutdown)
            release_reason = "unknown"

            logger.info(f"GPU {gpu_id}: Preparing job ID {job_id} (Hash: {job_hash}, Name: '{job_name}', Mode: {mode_tag}, Prio: {priority}, Conda: {conda_env or 'None'})")
            # Log the arguments *including* the added --cuda flag
            logger.debug(f"GPU {gpu_id}: Job ID {job_id} arguments for execution: {args_for_exec}")

            # Set CUDA_VISIBLE_DEVICES environment variable for the job
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            try:
                # Create a modified job tuple with the arguments including --cuda for passing to run methods
                # Note: allowed_gpus and hash remain the same as the original job definition
                exec_job_tuple = (priority, job_id, script_path, conda_env, args_for_exec, job_tuple[5], job_hash)

                if self.use_screen:
                    # --- Screen Mode ---
                    launched_ok = self._run_with_screen(exec_job_tuple, gpu_id, env, start_time, mode_tag)
                    if not launched_ok:
                        logger.error(f"GPU {gpu_id}: Screen setup failed for job ID {job_id} (Hash: {job_hash}).")
                        job_completed_successfully = False
                        release_reason = "screen setup failure"
                        # Release GPU immediately if screen launch failed
                        self._release_gpu(gpu_id, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                    # If launched_ok is True, the monitor thread will handle release
                    return # Exit _run_job thread; monitor takes over

                else:
                    # --- Direct Mode ---
                    job_completed_successfully = self._run_directly(exec_job_tuple, gpu_id, env, start_time, mode_tag)
                    release_reason = f"direct execution completion (Success: {job_completed_successfully})"
                    # Release GPU after direct execution finishes
                    self._release_gpu(gpu_id, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                    return

            except Exception as e:
                # Catch unexpected errors during the setup/dispatch phase of _run_job
                logger.error(f"GPU {gpu_id}: CRITICAL Unexpected error in _run_job setup/dispatch for job ID {job_id} (Hash: {job_hash}): {e}", exc_info=True)
                job_completed_successfully = False
                release_reason = "critical error in _run_job setup"
                # Ensure GPU is released if error occurred before launch/monitor took over
                gpu_still_busy = False
                with self.lock:
                    # Check if the GPU is still marked as busy by *this scheduler instance*
                    # It might have been released by another action if the error took time
                    if 0 <= gpu_id < len(self.gpu_status) and self.gpu_status[gpu_id]:
                        gpu_still_busy = True

                if gpu_still_busy:
                    logger.warning(f"GPU {gpu_id}: Releasing GPU due to error during _run_job setup for job ID {job_id} (Hash: {job_hash}).")
                    self._release_gpu(gpu_id, job_id, job_name, start_time, mode_tag, job_hash, success=job_completed_successfully, reason=release_reason)
                else:
                    logger.warning(f"GPU {gpu_id}: Error during _run_job setup for job ID {job_id} (Hash: {job_hash}), but GPU was not marked busy or already released.")


    def _release_gpu(self, gpu_id: int, job_id: str, job_name: str, start_time: datetime, mode_tag: str, job_hash: Optional[str], success: bool, reason: str="completion"):
        """Helper method to mark GPU as available, save state, and manage job hash."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        # Include mode_tag and hash in log for clarity
        log_prefix = f"GPU {gpu_id}: Job ID {job_id} (Hash: {job_hash}, Name: '{job_name}', Mode: {mode_tag})"
        logger.info(f"{log_prefix} finished due to {reason} in {duration:.2f} seconds. Success: {success}. Releasing GPU.")

        with self.lock: # Ensure thread safety
            # Mark GPU as available
            if 0 <= gpu_id < len(self.gpu_status):
                 self.gpu_status[gpu_id] = False
            else:
                 logger.error(f"Attempted to release invalid GPU ID {gpu_id}. State not changed.")
                 # Continue to manage hash if possible

            # --- Manage Job Hash (for jobs originating from the file) ---
            if job_hash:
                if job_hash in self.managed_job_hashes:
                    # Remove hash from the managed set regardless of success or failure.
                    # This allows the job to be re-added by the monitor if it's still in the file
                    # and failed previously.
                    self.managed_job_hashes.remove(job_hash)
                    logger.info(f"{log_prefix}: Removed hash '{job_hash}' from managed set (Success: {success}). Job can be re-added by monitor if still in file.")
                    # Clean up the job_id mapping as well
                    if job_hash in self.hash_to_job_id:
                        del self.hash_to_job_id[job_hash]
                else:
                    # This might happen if the job was added manually or if the scheduler restarted
                    # and the hash wasn't reloaded yet.
                    logger.debug(f"{log_prefix}: Hash '{job_hash}' not found in managed set during release (might be expected).")
            # -----------------------

        self.save_state() # Save updated GPU status (now available)


    def _run_with_screen(self, job_tuple: Tuple, gpu_id: int, env: Dict, start_time: datetime, mode_tag: str) -> bool:
            """
            Sets up and launches a job inside a GNU Screen session.
            The job_tuple's args should include the scheduler-added '--cuda <gpu_id>'.

            Returns:
                bool: True if the screen session was launched successfully, False otherwise.
            """
            # Unpack the job tuple which now includes the modified arguments
            priority, job_id, script_path, conda_env, args_with_cuda, _, job_hash = job_tuple
            script_p = Path(script_path)
            script_basename = script_p.name
            safe_basename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', script_basename)
            # Create a unique session name incorporating relevant info
            session_name = f"gpujob_{gpu_id}_{mode_tag}_{job_id[:6]}_{start_time.strftime('%H%M%S')}"
            session_name = session_name[:60] # Limit length for sanity

            # --- Files for communication ---
            marker_file = MARKER_FILE_DIR / f"success_{session_name}.marker"
            exit_code_file = MARKER_FILE_DIR / f"exitcode_{session_name}.txt"
            job_stdout_log = MARKER_FILE_DIR / f"output_{session_name}.log"
            # Ensure clean state before starting
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
                    f"# Job ID: {job_id}", f"# Hash: {job_hash}", f"# Mode Tag: {mode_tag}",
                    f"# Screen session: {session_name}",
                    f"# Script: {script_path}", f"# GPU: {gpu_id}", f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# Marker File: {marker_file}",
                    f"# Exit Code File: {exit_code_file}",
                    f"# Job Stdout/Stderr Log: {job_stdout_log}",
                    "echo \"--- Starting Job Execution Script ---\"",
                    "echo \"Timestamp: $(date)\"",
                    "echo \"Running as user: $(whoami)\"",
                    "echo \"Current directory: $(pwd)\"",
                    "set -e", # Exit immediately if a command exits with a non-zero status.
                    "set -o pipefail" # Causes a pipeline to return the exit status of the last command in the pipe that failed
                ]

                # --- Conda activation logic ---
                if conda_env:
                    # Try to find conda.sh robustly
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
                            logger.debug(f"Found conda.sh via CONDA_PREFIX: {potential_conda_sh}")
                    if not conda_path_found: # Check common default locations
                        home_dir = os.environ.get('HOME', '/root') # Get home directory
                        for default_path in [f"{home_dir}/anaconda3/etc/profile.d/conda.sh",
                                             f"{home_dir}/miniconda3/etc/profile.d/conda.sh",
                                             "/opt/conda/etc/profile.d/conda.sh"]: # Add more if needed
                            potential_conda_sh = Path(default_path)
                            if potential_conda_sh.exists():
                                conda_base_cmd = f"source {shlex.quote(str(potential_conda_sh))}"
                                conda_path_found = True
                                logger.debug(f"Found conda.sh via default path: {potential_conda_sh}")
                                break
                    if not conda_path_found:
                        logger.warning(f"GPU {gpu_id}: Job ID {job_id}: Could not reliably determine conda base path. Using default guess: '{conda_base_cmd}'. Activation might fail.")

                    script_content.extend([
                        f"echo 'Attempting to initialize conda using: {conda_base_cmd}'",
                        conda_base_cmd,
                        "conda_init_exit_code=$?",
                        # Allow non-zero exit code for init, sometimes happens harmlessly
                        "if [ $conda_init_exit_code -ne 0 ]; then echo \"WARNING: Conda init command exited with code $conda_init_exit_code (may be benign)\" >&2; fi",
                        f"echo 'Activating conda environment: {conda_env}'",
                        f"conda activate {shlex.quote(conda_env)}",
                        "conda_activate_exit_code=$?",
                        # Activation MUST succeed
                        "if [ $conda_activate_exit_code -ne 0 ]; then echo \"ERROR: Conda activate '{conda_env}' failed with code $conda_activate_exit_code\" >&2; exit $conda_activate_exit_code; fi",
                        "echo 'Conda environment activated.'",
                        "echo \"PATH: $PATH\"", "echo \"CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV\"",
                        "echo \"Which Python: $(which python)\"", "echo \"Python Version: $(python --version)\"",
                    ])
                else:
                    script_content.append("echo 'No conda environment specified.'")

                # Prepare Python command execution using args_with_cuda
                cmd_list = ["python", "-u", script_path] # -u for unbuffered output
                # Extend with the arguments that *already include* --cuda
                if args_with_cuda:
                    try:
                        # Ensure all args are strings for shlex.join
                        str_args = [str(arg) for arg in args_with_cuda]
                        cmd_list.extend(str_args)
                    except Exception as e_args:
                        # This should ideally not happen if args are well-formed lists/tuples
                        raise ValueError(f"Invalid screen arguments for job {job_id}: {e_args}") from e_args

                cmd_str = shlex.join(cmd_list) # Safely join command parts for execution and logging
                script_content.extend([
                    f"echo 'Executing Python command: {cmd_str}'", # Log the command with --cuda
                    # Redirect stdout and stderr of the python command to the job log file
                    f"echo '--- Python Script Output Start ---' >> {shlex.quote(str(job_stdout_log))}",
                    f"({cmd_str}) >> {shlex.quote(str(job_stdout_log))} 2>&1",
                    "exit_code=$?", # Capture the exit code of the python script
                    f"echo '--- Python Script Output End (Exit Code: $exit_code) ---' >> {shlex.quote(str(job_stdout_log))}",
                    f"echo \"Python command finished with exit code: $exit_code\"",
                    # --- Record results ---
                    f"echo $exit_code > {shlex.quote(str(exit_code_file))}", # Write exit code to file
                    f"if [ $exit_code -eq 0 ]; then",
                    f"  echo 'Command succeeded. Creating marker file.'",
                    f"  touch {shlex.quote(str(marker_file))}", # Create marker only on success (exit code 0)
                    f"else",
                    f"  echo 'Command failed (Exit Code: $exit_code). NOT creating marker file.'",
                    f"fi",
                    f"echo \"--- Ending Job Execution Script --- Timestamp: $(date)\"",
                    f"exit $exit_code" # Exit the wrapper script with the python script's exit code
                ])

                # Write and execute the wrapper script
                temp_script_dir = Path("/tmp") # Or a more specific temp directory
                temp_script_path = temp_script_dir / f"run_{session_name}_{job_id[:4]}.sh"
                with open(temp_script_path, 'w') as f: f.write("\n".join(script_content))
                os.chmod(temp_script_path, 0o755) # Make executable
                logger.debug(f"GPU {gpu_id}: Temp script for job ID {job_id} (Hash: {job_hash}, Mode: {mode_tag}): {temp_script_path}")

                # --- Start screen session ---
                # -d -m: Start detached
                # -S session_name: Set the session name
                screen_cmd = ['screen', '-dmS', session_name, str(temp_script_path)]
                logger.info(f"GPU {gpu_id}: Starting screen session '{session_name}' for job ID {job_id} (Hash: {job_hash}, Mode: {mode_tag})")
                # Use the environment with CUDA_VISIBLE_DEVICES set
                process = subprocess.run(screen_cmd, env=env, check=True, capture_output=True, text=True)
                logger.info(f"GPU {gpu_id}: Screen session launched. To view progress: screen -r {session_name}")
                logger.info(f"GPU {gpu_id}: Job stdout/stderr log: {job_stdout_log}")

                # --- Start monitoring thread ---
                # Pass the original job_hash, not one calculated with modified args
                monitoring_thread = threading.Thread(
                    target=self._monitor_screen,
                    args=(session_name, job_id, script_path, gpu_id, temp_script_path, start_time, mode_tag, job_hash, marker_file, exit_code_file),
                    name=f"ScreenMonitor-GPU{gpu_id}-{mode_tag}-{job_id[:6]}",
                    daemon=True # Allow main program to exit even if monitors are running (though cleanup is preferred)
                )
                monitoring_thread.start()

                return True # Launch success

            except subprocess.CalledProcessError as e:
                 logger.error(f"GPU {gpu_id}: Failed to launch screen session '{session_name}'. Return code: {e.returncode}")
                 logger.error(f"Stdout: {e.stdout}")
                 logger.error(f"Stderr: {e.stderr}")
                 # Fall through to general exception handling for cleanup
            except Exception as e:
                logger.error(f"GPU {gpu_id}: Error setting up or launching screen session for job ID {job_id} (Hash: {job_hash}, Mode: {mode_tag}): {e}", exc_info=True)
                # Cleanup temp files if created
                if temp_script_path and temp_script_path.exists():
                    try: temp_script_path.unlink(missing_ok=True)
                    except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")
                marker_file.unlink(missing_ok=True)
                exit_code_file.unlink(missing_ok=True)
                job_stdout_log.unlink(missing_ok=True)
                return False # Launch failure


    def _monitor_screen(self, session_name: str, job_id: str, script_path: str, gpu_id: int,
                        temp_script_path: Optional[Path], start_time: datetime, mode_tag: str,
                        job_hash: Optional[str], marker_file: Path, exit_code_file: Path):
        """
        Monitors a specific screen session until it terminates or the scheduler stops.
        Determines job success based on the marker file created by the wrapper script.
        Cleans up temporary files and calls _release_gpu.
        This runs in a dedicated thread started by _run_with_screen.
        """
        job_name = Path(script_path).name
        log_prefix = f"GPU {gpu_id}: Job ID {job_id} (Hash: {job_hash}, Screen: {session_name}, Mode: {mode_tag})"
        logger.info(f"{log_prefix} Monitoring screen session...")
        active = True
        final_success = False # Assume failure unless marker file proves otherwise
        exit_code = -1 # Default exit code if not found or script fails early
        check_interval = 15 # Seconds between checks

        # Loop while the screen session appears active and the scheduler hasn't stopped
        while active and not self.stop_event.is_set():
            try:
                # Check if screen session exists using `screen -ls` and grep
                # Use a more precise regex to avoid matching substrings
                # Match start of line, optional whitespace, PID, dot, exact session name, space, parenthesis
                cmd = f"screen -ls | grep -qE '^\\s*[0-9]+\\.{re.escape(session_name)}\\s+\\(' "
                # Run the command. check=True will raise CalledProcessError if grep finds nothing (exit code 1)
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                # If check_output succeeds, the session exists
                active = True
                logger.debug(f"{log_prefix} Screen session is active.")
            except subprocess.CalledProcessError:
                # grep returned non-zero (no match found), session has likely ended
                active = False
                logger.info(f"{log_prefix} Screen session finished or not found.")
            except FileNotFoundError:
                 logger.error(f"{log_prefix} 'screen' command not found during monitoring. Cannot check session status.")
                 active = False # Cannot monitor, assume finished to release GPU
            except Exception as e:
                logger.error(f"{log_prefix} Error checking screen session: {e}. Assuming finished.")
                active = False # Assume finished on error to release GPU

            if active:
                # Wait interruptibly for the next check interval
                if self.stop_event.wait(timeout=check_interval):
                    logger.warning(f"{log_prefix} Stop event received during monitoring.")
                    active = False # Stop monitoring
                    final_success = False # Treat interruption as failure
                    # Don't break here, proceed to cleanup and release GPU

        # --- Screen session ended or scheduler stopped ---
        logger.info(f"{log_prefix} Monitoring finished.")

        # Determine final success status based on the marker file *after* the loop
        if marker_file.exists():
            final_success = True
            logger.info(f"{log_prefix} Success marker file found ({marker_file}).")
            try: marker_file.unlink(missing_ok=True)
            except OSError as e: logger.warning(f"{log_prefix} Error removing marker file {marker_file}: {e}")
        else:
            final_success = False
            # Log as warning only if the stop event wasn't the primary reason for stopping monitoring
            if not self.stop_event.is_set():
                 logger.warning(f"{log_prefix} Success marker file NOT found. Assuming failure.")
            else:
                 logger.info(f"{log_prefix} Success marker file not found (scheduler stopping).")


        # Read the exit code from the file if it exists
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
        if self.stop_event.is_set() and not active: # Check if stop event was set AND monitoring loop exited
            reason = f"scheduler shutdown during screen session (Final Status - Exit Code: {exit_code}, Success: {final_success})"

        # --- Call _release_gpu from the monitor thread ---
        # This ensures the GPU is marked available and hash is managed correctly
        self._release_gpu(gpu_id, job_id, job_name, start_time, mode_tag, job_hash, final_success, reason)


    def _run_directly(self, job_tuple: Tuple, gpu_id: int, env: Dict, start_time: datetime, mode_tag: str) -> bool:
        """
        Runs a job directly as a subprocess (without screen).
        The job_tuple's args should include the scheduler-added '--cuda <gpu_id>'.

        Returns:
            bool: True if the job script exited with code 0, False otherwise.
        """
        # Unpack the job tuple which includes the modified arguments
        priority, job_id, script_path, conda_env, args_with_cuda, _, job_hash = job_tuple
        script_p = Path(script_path)
        job_name = script_p.name
        # Create a descriptive log filename
        log_filename = Path(f"job_{job_name}_{mode_tag}_{job_hash[:8] if job_hash else 'nohash'}_gpu{gpu_id}_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
        log_prefix = f"GPU {gpu_id}: Job ID {job_id} (Hash: {job_hash}, Name: '{job_name}', Mode: {mode_tag})"
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
            log_file.write(f"Mode Tag: {mode_tag}\n")
            log_file.write(f"Script: {script_path}\n")
            log_file.write(f"Conda Env: {conda_env or 'None'}\n")
            # Log arguments *including* the added --cuda
            log_file.write(f"Arguments (with added --cuda): {args_with_cuda or 'None'}\n")
            log_file.write(f"GPU ID (Assigned): {gpu_id}\n")
            log_file.write(f"Start Time: {start_time.isoformat()}\n")
            log_file.write("-" * 60 + "\n")
            log_file.flush()

            command_list = []
            shell_mode = False # Generally avoid shell=True unless necessary

            # --- Build command ---
            if conda_env:
                # Use a temporary bash script to handle conda activation reliably
                shell_script_lines = ["#!/bin/bash", f"# Job ID: {job_id}", f"# Hash: {job_hash}", "set -e", "set -o pipefail"]
                # Conda activation logic (same robust finding as in screen method)
                conda_base_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh"
                # ... [Copy the robust conda path finding logic here from _run_with_screen] ...
                # (For brevity, assuming the logic is copied here)
                # Example placeholder for the copied logic:
                conda_path_found = False # Assume logic sets this
                # ...
                if not conda_path_found: logger.warning(f"{log_prefix}: Could not reliably determine conda base path for direct run.")
                # ...

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
                # Add the python command (including --cuda) to the bash script
                shell_script_lines.append(shlex.join(python_cmd_list))

                # Write and prepare temp script execution
                script_id = f"{start_time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_{job_id[:4]}"
                temp_script_dir = Path("/tmp")
                temp_script_path = temp_script_dir / f"direct_job_{script_id}.sh"
                with open(temp_script_path, 'w') as f: f.write("\n".join(shell_script_lines))
                os.chmod(temp_script_path, 0o755)
                command_list = [str(temp_script_path)] # Execute the wrapper script
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
                text=True, bufsize=1, shell=shell_mode # Use shell=False
            )

            # --- Real-time output handling ---
            # Use separate threads to read stdout and stderr to prevent blocking
            def stream_output(pipe, prefix, log_f, logger_func):
                try:
                    # Read line by line until the pipe closes
                    for line in iter(pipe.readline, ''):
                        line_stripped = line.rstrip()
                        # Log to main scheduler log
                        logger_func(f"[{log_prefix} {prefix}] {line_stripped}")
                        # Write to the specific job log file
                        if log_f:
                            try:
                                log_f.write(f"{prefix}: {line}")
                                log_f.flush() # Ensure it's written immediately
                            except Exception as write_e:
                                logger.error(f"Error writing to log file {log_filename}: {write_e}")
                except Exception as e:
                    # Catch errors during stream reading (e.g., pipe closed unexpectedly)
                    logger.error(f"Stream reading error ({log_prefix} {prefix}): {e}")
                finally:
                    # Ensure the pipe is closed (though Popen.wait() should handle this)
                    try: pipe.close()
                    except Exception: pass

            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "OUT", log_file, logger.info), daemon=True)
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERR", log_file, logger.warning), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process completion and get the exit code
            return_code = process.wait()

            # Wait for output threads to finish reading any remaining buffered output
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
            return_code = -1 # Ensure failure is recorded
        finally:
            # --- Cleanup ---
            if log_file:
                try: log_file.close()
                except Exception: pass
            # Clean up temp script if created
            if temp_script_path and temp_script_path.exists():
                try: temp_script_path.unlink()
                except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")

            # Return True only if the process exit code was explicitly 0
            return return_code == 0


    # --- Job Addition Methods ---
    def add_job(self, script: str, conda_env: Optional[str] = None, args: Optional[List[str]] = None,
                priority: int = 0, allowed_gpus: Optional[List[int]] = None,
                original_line: Optional[str] = None):
        """
        Adds a single job with a unique ID to the queue.
        Uses 'allowed_gpus' parameter to constrain GPU choice.
        If original_line is provided (meaning it came from the jobs file),
        it calculates a hash based on original parameters and adds the job
        only if the hash isn't already managed.
        """
        script_p = Path(script)
        # Basic validation: Check if the script file exists
        if not script_p.is_file(): # Check if it's a file specifically
            logger.error(f"Script path not found or is not a file: {script}. Job not added.")
            return

        job_id = str(uuid.uuid4()) # Generate a unique ID for every job addition attempt
        job_hash = None # Will be calculated only if original_line is present
        log_source = "Manual" if original_line is None else "File"
        args_list = args or [] # Ensure args is a list

        # --- Determine final allowed GPUs based ONLY on input parameter ---
        final_allowed_gpus: Optional[List[int]] = None
        if allowed_gpus is not None:
            initial_count = len(allowed_gpus)
            # Filter out invalid GPU IDs based on configured num_gpus
            valid_gpus_from_param = [gpu_id for gpu_id in allowed_gpus if 0 <= gpu_id < self.num_gpus]
            if not valid_gpus_from_param:
                logger.error(f"No valid GPUs in allowed_gpus list {allowed_gpus} for script {script} (Job ID {job_id}). Max GPU ID is {self.num_gpus - 1}. Job not added.")
                return
            elif len(valid_gpus_from_param) < initial_count:
                 logger.warning(f"Invalid GPU IDs removed from allowed_gpus for script {script} (Job ID {job_id}). Original: {allowed_gpus}, Valid: {valid_gpus_from_param}")
            final_allowed_gpus = sorted(list(set(valid_gpus_from_param))) # Ensure sorted unique list
        # If allowed_gpus is None, final_allowed_gpus remains None (meaning any GPU)

        # --- Hash calculation and tracking for jobs from file ---
        if original_line is not None:
            log_source = "File"
            try:
                # Calculate hash based on original job parameters (including original args)
                job_hash = self._calculate_job_hash(priority, script, conda_env, args_list, final_allowed_gpus)
                with self.lock: # Lock needed for checking/modifying shared managed_job_hashes
                    if job_hash in self.managed_job_hashes:
                        # Log if the job is already running or queued (based on hash)
                        existing_job_id = self.hash_to_job_id.get(job_hash, "Unknown")
                        logger.debug(f"Job from line '{original_line.strip()}' (Hash: {job_hash}, Existing/Queued JobID: {existing_job_id}) is already managed. Skipping add.")
                        return # Don't add duplicates
                    else:
                        # Add the hash and map it to the new job_id
                        self.managed_job_hashes.add(job_hash)
                        self.hash_to_job_id[job_hash] = job_id
                        logger.info(f"Adding job from file (Hash: {job_hash}, JobID: {job_id}). Marked as managed.")
            except Exception as e:
                logger.error(f"Error calculating hash or managing state for job from line '{original_line.strip()}': {e}. Job not added.")
                # Clean up if hash was partially added
                if job_hash:
                     with self.lock:
                         self.managed_job_hashes.discard(job_hash)
                         self.hash_to_job_id.pop(job_hash, None)
                return
        # -------------------------------------------------------

        # Job tuple format: (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash)
        # Store the *original* args_list. The --cuda flag is added at execution time.
        job_tuple = (priority, job_id, str(script_p), conda_env, args_list, final_allowed_gpus, job_hash)
        try:
            self.job_queue.put(job_tuple)
            # Log the allowed GPUs based on the job definition (not runtime)
            logger.info(f"Job ID {job_id} ({log_source}, Hash: {job_hash or 'N/A'}) added to queue: '{script_p.name}' (Prio: {priority}, GPUs: {final_allowed_gpus or 'Any'})")
        except Exception as e:
            logger.error(f"Failed to add job ID {job_id} ('{script_p.name}') to queue: {e}")
            # If adding to queue fails, remove from managed set if it was added
            if job_hash:
                with self.lock:
                    if job_hash in self.managed_job_hashes:
                        self.managed_job_hashes.remove(job_hash)
                        self.hash_to_job_id.pop(job_hash, None)
                        logger.warning(f"Removed hash {job_hash} from managed set due to queue insertion failure.")


    def add_jobs_from_file(self, file_path: str, initial_load: bool = False):
        """Adds multiple jobs from a file, generating unique IDs and hashes."""
        log_prefix = "[File Monitor]" if not initial_load else "[Initial Load]"
        logger.info(f"{log_prefix} Attempting to process jobs from file: {file_path}")
        jobs_processed = 0
        jobs_added = 0 # Track newly added jobs this scan
        file_p = Path(file_path)

        if not file_p.is_file():
            # Log differently for initial load vs monitor
            if initial_load:
                logger.error(f"{log_prefix} Job file not found: {file_path}. No initial jobs loaded.")
            else:
                # This is expected if the file is temporarily removed/renamed
                logger.debug(f"{log_prefix} Job file not found: {file_path}. Will check again later.")
            return

        try:
            with open(file_p, 'r') as f:
                current_lines = f.readlines() # Read all lines at once

            for i, line in enumerate(current_lines):
                line_num = i + 1
                original_line_content = line # Keep original for logging/hashing context
                line = line.strip()
                if not line or line.startswith('#'):
                    continue # Skip empty lines and comments

                jobs_processed += 1
                # --- Parse line ---
                try:
                    # Format: priority,script_path[,conda_env[,arguments[,allowed_gpus]]]
                    # Split carefully, allowing empty optional fields
                    parts = [p.strip() for p in line.split(',', maxsplit=4)]
                    if len(parts) < 2 or not parts[0] or not parts[1]:
                         logger.error(f"{log_prefix} Invalid job format on line {line_num}: '{line}'. Requires non-empty priority,script_path.")
                         continue

                    priority = int(parts[0])
                    script = parts[1]
                    # Handle optional parts carefully, treat empty strings as None
                    conda_env = parts[2] if len(parts) > 2 and parts[2] else None
                    args_str = parts[3] if len(parts) > 3 and parts[3] else None
                    allowed_gpus_str = parts[4] if len(parts) > 4 and parts[4] else None

                    # Parse arguments using shlex
                    # Arguments string should NOT contain the --cuda flag; it's added by the scheduler
                    args_list = shlex.split(args_str) if args_str else None
                    if args_list and '--cuda' in args_list:
                        logger.warning(f"{log_prefix} Found '--cuda' in arguments on line {line_num}: '{line}'. This flag will be ignored by the scheduler's job definition and added automatically at runtime based on assigned GPU. Remove it from jobs.txt arguments to avoid confusion.")
                        # We keep the original args_list for hashing consistency.

                    # Parse allowed GPUs string (e.g., "0,1,3-5")
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
                                    else: logger.warning(f"{log_prefix} Invalid range '{part}' (start > end) on line {line_num}")
                                except ValueError: logger.warning(f"{log_prefix} Invalid range format '{part}' on line {line_num}")
                            else:
                                try: allowed_gpus_list.append(int(part))
                                except ValueError: logger.warning(f"{log_prefix} Invalid GPU ID '{part}' on line {line_num}")
                        # We validate against num_gpus inside add_job

                    # --- Call add_job with original_line for hash checking ---
                    # Pass the parsed arguments and allowed_gpus list
                    # add_job handles checking if the hash is already managed
                    # We capture the state *before* calling add_job to see if it was newly added
                    job_hash_before = self._calculate_job_hash(priority, script, conda_env, args_list, allowed_gpus_list) # Recalculate hash for check
                    is_managed_before = False
                    with self.lock:
                        is_managed_before = job_hash_before in self.managed_job_hashes

                    self.add_job(script, conda_env, args_list, priority, allowed_gpus_list, original_line=original_line_content)

                    # Check if it's managed *after* calling add_job
                    is_managed_after = False
                    with self.lock:
                        is_managed_after = job_hash_before in self.managed_job_hashes

                    if not is_managed_before and is_managed_after:
                        jobs_added += 1 # Count only if it was newly added

                except ValueError as ve:
                    logger.error(f"{log_prefix} Invalid priority or number format on line {line_num}: '{line}'. Error: {ve}")
                except Exception as e:
                    logger.error(f"{log_prefix} Error parsing job on line {line_num}: '{line}'. Error: {e}", exc_info=True)

            log_message = f"{log_prefix} Finished processing job file '{file_path}'. Processed {jobs_processed} non-comment lines."
            if not initial_load: # Only log added count for monitor scans
                 log_message += f" Added {jobs_added} new jobs to the queue."
            logger.info(log_message)


        except FileNotFoundError:
             # This might happen if the file is deleted between the check and the open
             logger.error(f"{log_prefix} Job file disappeared during processing: {file_path}")
        except Exception as e:
             logger.error(f"{log_prefix} Failed to read or process job file {file_path}: {e}", exc_info=True)


    # --- File Monitoring ---
    def _monitor_jobs_file(self):
        """Periodically scans the jobs file and adds new/unmanaged jobs."""
        logger.info(f"Starting job file monitor for '{self.jobs_file_path}' (Interval: {self.file_monitor_interval}s)")
        while not self.stop_event.is_set():
            if self.jobs_file_path:
                try:
                    logger.debug(f"Checking jobs file: {self.jobs_file_path}")
                    # Re-process the entire file each time. add_job handles duplicates via hash check.
                    self.add_jobs_from_file(str(self.jobs_file_path), initial_load=False)
                except Exception as e:
                    logger.error(f"[File Monitor] Error during periodic check of {self.jobs_file_path}: {e}", exc_info=True)

            # Wait for the specified interval OR until stop event is set
            if self.stop_event.wait(timeout=self.file_monitor_interval):
                break # Stop event was set, exit loop

        logger.info("Job file monitor stopped.")

    # --- Start/Stop Modifications ---
    def enable_screen(self):
       """Checks if GNU Screen is available and enables its use."""
       try:
           # Check if 'screen' command exists and is executable
           subprocess.run(['screen', '-v'], check=True, capture_output=True, text=True)
           self.use_screen = True
           logger.info("Screen functionality enabled.")
       except FileNotFoundError:
            logger.error("GNU Screen command ('screen') not found. Screen functionality disabled.")
            self.use_screen = False
       except subprocess.CalledProcessError as e:
            logger.error(f"Error running 'screen -v': {e}. Screen functionality disabled.")
            self.use_screen = False
       except Exception as e: # Catch any other unexpected errors
           logger.error(f"Screen check failed unexpectedly: {e}. Screen functionality disabled.")
           self.use_screen = False


# --- Command Line Interface (Modified for Job File) ---
def main():
    # Try to detect GPUs early for default value
    detected_gpus = 0
    try:
        gpus_list = GPUtil.getGPUs()
        detected_gpus = len(gpus_list) if gpus_list else 0
    except Exception as e:
        logger.warning(f"Could not detect GPUs using GPUtil: {e}. Defaulting --gpus might be inaccurate.")
        detected_gpus = 1 # Fallback to 1 if detection fails

    parser = argparse.ArgumentParser(description='GPU Job Scheduler v2.2 (Auto --cuda, Fixed Worker)', formatter_class=argparse.RawTextHelpFormatter) # Updated description
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- Start Command ---
    start_parser = subparsers.add_parser('start', help='Start the scheduler daemon process.')
    start_parser.add_argument('--gpus', type=int, default=detected_gpus, help=f'Number of GPUs to manage (default: {detected_gpus} detected)')
    # Changed --jobs to --jobs-file and made it optional for monitoring
    start_parser.add_argument('--jobs-file', type=str, default=DEFAULT_JOBS_FILE, help=f'Path to jobs file for initial load and monitoring (default: {DEFAULT_JOBS_FILE}). If file exists, it will be monitored.')
    start_parser.add_argument('--no-monitor', action='store_true', help='Disable dynamic monitoring of the jobs file, only perform initial load if file exists.')
    start_parser.add_argument('--screen', action='store_true', help='Enable GNU Screen sessions for job execution.')
    start_parser.add_argument('--mem-threshold', type=float, default=0.8, help='GPU memory utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--load-threshold', type=float, default=0.8, help='GPU load utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--monitor-interval', type=int, default=FILE_MONITOR_INTERVAL_S, help=f'Interval (seconds) to check the jobs file (default: {FILE_MONITOR_INTERVAL_S})')
    start_parser.add_argument('--max-assign-attempts', type=int, default=MAX_ASSIGNMENT_ATTEMPTS, help=f'Max attempts to assign a job before requeuing (default: {MAX_ASSIGNMENT_ATTEMPTS})')
    start_parser.add_argument('--assign-retry-wait', type=int, default=ASSIGNMENT_RETRY_WAIT_S, help=f'Base wait (s) between assignment attempts (default: {ASSIGNMENT_RETRY_WAIT_S})')


    # --- Add Command (Manual addition) ---
    add_parser = subparsers.add_parser('add', help='Manually add a new job to the queue (requires IPC or scheduler restart).')
    add_parser.add_argument('script', type=str, help='Path to the Python script.')
    add_parser.add_argument('--conda', type=str, help='Conda environment name (optional).')
    # Arguments here should NOT include --cuda
    add_parser.add_argument('--args', type=str, help='Script arguments (quote if needed, use shlex syntax). DO NOT include --cuda.')
    add_parser.add_argument('--priority', type=int, default=0, help='Job priority (lower=higher, default: 0).')
    add_parser.add_argument('--gpus', type=str, help='Allowed GPU IDs (e.g., "0,1" or "0-2,5", optional). If omitted, any GPU is allowed.')

    # --- Add File Command (Manual one-time load) ---
    # This is less relevant if start monitor is used, but kept for manual batch loading
    add_file_parser = subparsers.add_parser('add-file', help='Manually add jobs from a file (one-time load, requires IPC/restart).')
    add_file_parser.add_argument('file_path', type=str, help='Path to the job definition file.')

    # --- Other Commands (Status, Pause, Resume, Screens) ---
    subparsers.add_parser('status', help='Show status of GPUs and job queue (queue info might be limited without IPC).')
    pause_parser = subparsers.add_parser('pause', help='Pause a GPU (modifies state file).')
    pause_parser.add_argument('gpu_id', type=int, help='ID of the GPU to pause.')
    resume_parser = subparsers.add_parser('resume', help='Resume a paused GPU (modifies state file).')
    resume_parser.add_argument('gpu_id', type=int, help='ID of the GPU to resume.')
    subparsers.add_parser('screens', help='List active GNU Screen sessions created by the scheduler (matching pattern).')

    args = parser.parse_args()

    # --- Determine Number of GPUs for non-start commands ---
    # Try reading from state file first, then detect, then fallback
    default_num_gpus = 8 # Fallback
    try:
        state_file = Path("gpu_scheduler_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f: state = json.load(f)
            if 'gpu_status' in state and isinstance(state['gpu_status'], list):
                default_num_gpus = len(state['gpu_status'])
                logger.debug(f"Determined GPU count ({default_num_gpus}) from state file for control command.")
            else:
                 raise ValueError("State file exists but lacks valid 'gpu_status' list.")
        else:
             # State file doesn't exist, try detection
             gpus_list = GPUtil.getGPUs()
             if gpus_list:
                 default_num_gpus = len(gpus_list)
                 logger.debug(f"Determined GPU count ({default_num_gpus}) via GPUtil for control command.")
             else:
                 logger.warning(f"State file not found and GPUtil detection failed/returned none. Using fallback GPU count: {default_num_gpus}")
                 default_num_gpus = default_num_gpus # Use fallback
    except Exception as e:
        logger.warning(f"Error determining GPU count for control command: {e}. Using fallback: {default_num_gpus}")


    # --- Execute Commands ---
    if args.command == 'start':
        jobs_file_to_monitor = args.jobs_file if not args.no_monitor else None
        jobs_file_path_obj = Path(args.jobs_file) if args.jobs_file else None

        # Check existence for initial load message, but pass path regardless for monitor
        if jobs_file_to_monitor and not jobs_file_path_obj.exists():
            logger.warning(f"Jobs file '{jobs_file_to_monitor}' does not exist at start. Monitoring will check periodically.")
        elif not jobs_file_to_monitor and jobs_file_path_obj and jobs_file_path_obj.exists():
             logger.info(f"Jobs file '{args.jobs_file}' exists but monitoring is disabled (--no-monitor). Performing initial load only.")
             # Set jobs_file_to_monitor to None so the monitor thread doesn't start
             jobs_file_to_monitor = None # Ensure monitor is disabled
             # The initial load will happen inside scheduler init/start based on jobs_file_path
             jobs_file_path_for_init = str(jobs_file_path_obj)
        elif jobs_file_to_monitor and jobs_file_path_obj.exists():
             logger.info(f"Jobs file '{args.jobs_file}' found. Initial load and monitoring enabled.")
             jobs_file_path_for_init = jobs_file_to_monitor
        else: # No jobs file specified or monitoring disabled
             jobs_file_path_for_init = None


        logger.info(f"Starting scheduler process with {args.gpus} GPUs.")
        logger.info(f"Jobs file path configured: {args.jobs_file}")
        logger.info(f"File Monitoring: {'ENABLED' if jobs_file_to_monitor else 'DISABLED'}")
        if jobs_file_to_monitor:
            logger.info(f"Monitor Interval: {args.monitor_interval}s")

        scheduler = GPUJobScheduler(
            num_gpus=args.gpus,
            gpu_memory_threshold=args.mem_threshold,
            gpu_load_threshold=args.load_threshold,
            # Pass the path that should be monitored (None if disabled)
            jobs_file_path=jobs_file_to_monitor,
            monitor_interval=args.monitor_interval,
            max_assignment_attempts=args.max_assign_attempts,
            assignment_retry_wait=args.assign_retry_wait
        )
        if args.screen: scheduler.enable_screen()

        # Initial load from --jobs-file now happens inside scheduler.start() if jobs_file_path is set
        # and the file exists at that time.

        scheduler.start() # This now handles initial load and starts monitor/workers
        try:
            # Keep main thread alive, workers and monitor run in background
            while True:
                # Check if worker threads are alive periodically (optional health check)
                alive_workers = [t.is_alive() for t in scheduler.worker_threads]
                if not all(alive_workers) and not scheduler.stop_event.is_set():
                    logger.warning(f"Some worker threads have unexpectedly stopped! Alive status: {alive_workers}")
                    # Consider adding logic to restart workers if desired
                time.sleep(60) # Sleep for a minute
        except KeyboardInterrupt:
            logger.info("Ctrl+C received. Initiating shutdown...")
        finally:
            scheduler.stop()
            logger.info("Scheduler shutdown complete.")

    else:
        # --- Control Commands ---
        # These commands still have the limitation of not directly interacting
        # with the running scheduler's queue via IPC. They primarily read/write
        # the state file or simulate actions.
        scheduler_control = GPUJobScheduler(num_gpus=default_num_gpus) # Instance for state/method access

        if args.command == 'add':
                print("WARNING: 'add' command requires IPC to affect a running scheduler.")
                print("Simulating manual job addition parameters:")
                sim_job_id = str(uuid.uuid4())
                print(f"  Job ID (generated): {sim_job_id}")
                print(f"  Script: {args.script}")
                print(f"  Conda: {args.conda or 'None'}")
                print(f"  Args: {args.args or 'None'} (Note: --cuda will be added automatically by scheduler)")
                print(f"  Priority: {args.priority}")
                print(f"  Allowed GPUs: {args.gpus or 'Any'}")
                # To actually add, you'd need to implement IPC or add to jobs.txt
                # and let the monitor pick it up.

        elif args.command == 'add-file':
                print("WARNING: 'add-file' command requires IPC to affect a running scheduler.")
                print(f"Use '--jobs-file' with 'start' command for dynamic loading,")
                print(f"or add lines to the monitored file (default: {DEFAULT_JOBS_FILE}).")
                # scheduler_control.add_jobs_from_file(args.file_path) # This wouldn't affect running queue

        elif args.command == 'status':
                # Status display (GPU part is accurate from state file + GPUtil)
                status = scheduler_control.get_gpu_status()
                print("\n--- GPU Status (from state file & GPUtil) ---") # Updated title
                print(f"{'GPU ID':<8} {'State':<12} {'Memory Util':<15} {'Load Util':<15}")
                print("-" * 55)
                for gpu in status: print(f"{gpu['gpu_id']:<8} {gpu['state']:<12} {gpu['memory_util']:<15} {gpu['load']:<15}")

                # Queue info is NOT live from the daemon without IPC
                print("\n--- Job Queue (Info potentially limited without IPC) ---")
                queued_jobs = scheduler_control.get_job_queue_info() # Gets info from THIS instance's queue (likely empty)
                if not queued_jobs:
                    print("No jobs found in this command's local queue instance.")
                    print("(For live queue status, IPC implementation is needed or check scheduler logs)") # Updated message
                else:
                    # Display format adjusted for hash
                    print(f"{'Priority':<10} {'Job ID':<15} {'Hash':<10} {'Script':<30} {'Conda Env':<15} {'Allowed GPUs':<15} {'Arguments'}")
                    print("-" * 130)
                    for job in queued_jobs:
                            # Job tuple: (priority, job_id, script_path, conda_env, args, allowed_gpus, job_hash)
                            priority, job_id, script, conda, job_args, allowed_gpus, job_hash = job
                            script_name = Path(script).name if script else "N/A" # Handle potential None script in sentinel
                            args_str = shlex.join(job_args) if job_args else ""
                            # Format allowed_gpus nicely
                            allowed_gpus_str = ','.join(map(str, allowed_gpus)) if allowed_gpus is not None else 'Any'
                            print(f"{priority:<10} {job_id[:8]+'...':<15 if job_id else 15} {job_hash[:8] if job_hash else 'N/A':<10} {script_name:<30} {conda or 'None':<15} {allowed_gpus_str:<15} {args_str[:40]}") # Truncate long args

        elif args.command == 'pause':
                if scheduler_control.pause_gpu(args.gpu_id): print(f"GPU {args.gpu_id} paused (state file updated).")
                else: print(f"Failed to pause GPU {args.gpu_id}.")

        elif args.command == 'resume':
                if scheduler_control.resume_gpu(args.gpu_id): print(f"GPU {args.gpu_id} resumed (state file updated).")
                else: print(f"Failed to resume GPU {args.gpu_id}.")

        elif args.command == 'screens':
                print("Listing active GNU Screen sessions (matching gpujob_* pattern)...")
                active_sessions = list_screens()
                if not active_sessions: print("No matching screen sessions found.")
                else:
                    print("\n--- Active GPU Job Screen Sessions ---")
                    # Display sessions (name only)
                    for i, session in enumerate(active_sessions):
                        print(f"{i+1}. {session} (Attach: screen -r {session})")


if __name__ == "__main__":
    main()
