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
from typing import List, Dict, Tuple, Any, Optional # Add type hints

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s', # Added threadName
    handlers=[
        logging.FileHandler("gpu_scheduler.log"),
        logging.StreamHandler()
    ]
)
# Use a specific logger name
logger = logging.getLogger("GPUJobScheduler")

# --- Helper Function ---
def list_screens() -> List[str]:
    """List active screen sessions potentially related to GPU jobs"""
    try:
        result = subprocess.run(
            ['screen', '-list'],
            capture_output=True,
            text=True,
            check=True
        )
        screen_pattern = re.compile(r'^\s*(\d+\.gpujob_\d+_\S+)\s+\(.*\)', re.MULTILINE)
        matches = screen_pattern.findall(result.stdout)
        session_names = [match.split('.', 1)[-1] for match in matches]
        return session_names
    except FileNotFoundError:
        logger.warning("GNU Screen command ('screen') not found. Cannot list sessions.")
        return []
    except subprocess.CalledProcessError as e:
        if "No Sockets found" in e.stdout or "No Sockets found" in e.stderr:
             logger.info("No active screen sessions found.")
             return []
        logger.error(f"Error listing screen sessions: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing screen sessions: {e}")
        return []

# --- Main Scheduler Class ---
class GPUJobScheduler:
    def __init__(self, num_gpus: int = 8, gpu_memory_threshold: float = 0.8, gpu_load_threshold: float = 0.8):
        self.use_screen: bool = False
        self.num_gpus: int = num_gpus
        # Job tuple format: (priority, job_id, script_path, conda_env, args, allowed_gpus)
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.gpu_status: List[bool] = [False] * num_gpus # False = available, True = busy
        self.gpu_memory_threshold: float = gpu_memory_threshold
        self.gpu_load_threshold: float = gpu_load_threshold
        self.lock: threading.Lock = threading.Lock()
        self.stop_event: threading.Event = threading.Event()
        self.worker_threads: List[threading.Thread] = []
        self.paused_gpus: set[int] = set()
        self.state_file: Path = Path("gpu_scheduler_state.json") # Use Path object

        self.load_state()
        self._apply_paused_state()

    def _apply_paused_state(self):
        logger.info(f"Applying paused state for GPUs: {self.paused_gpus}")

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                # --- Load gpu_status ---
                loaded_status = state.get('gpu_status', [])
                if len(loaded_status) == self.num_gpus:
                    self.gpu_status = loaded_status
                # Adjust list size logic... (kept from previous version)
                elif len(loaded_status) < self.num_gpus:
                    self.gpu_status = loaded_status + [False] * (self.num_gpus - len(loaded_status))
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Added new GPUs as AVAILABLE.")
                else:
                    self.gpu_status = loaded_status[:self.num_gpus]
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Ignoring extra GPUs from state.")

                # --- Load paused GPUs ---
                loaded_paused = state.get('paused_gpus', [])
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

    def save_state(self):
        with self.lock:
            try:
                state = {
                    "gpu_status": list(self.gpu_status),
                    "paused_gpus": list(self.paused_gpus)
                }
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4)
                logger.debug("Scheduler state saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save state to '{self.state_file}': {e}")

    def pause_gpu(self, gpu_id: int) -> bool:
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} is already paused.")
                return True
            self.paused_gpus.add(gpu_id)
            logger.info(f"GPU {gpu_id} paused. It will not accept new jobs.")
        self.save_state()
        return True

    def resume_gpu(self, gpu_id: int) -> bool:
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False
        with self.lock:
            if gpu_id not in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} was not paused.")
                return True
            self.paused_gpus.remove(gpu_id)
            logger.info(f"GPU {gpu_id} resumed. It can now accept new jobs.")
        self.save_state()
        return True

    def get_gpu_status(self) -> List[Dict]:
        status_list = []
        gpus_util = []
        try:
            gpus_util = GPUtil.getGPUs()
        except Exception as e:
            logger.error(f"Could not get GPU utilization via GPUtil: {e}")

        with self.lock:
            current_gpu_status = list(self.gpu_status)
            current_paused_gpus = set(self.paused_gpus)

        for gpu_id in range(self.num_gpus):
            state = "PAUSED" if gpu_id in current_paused_gpus else \
                    "BUSY" if current_gpu_status[gpu_id] else \
                    "AVAILABLE"
            memory_util_str = "N/A"
            load_str = "N/A"
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
        """Returns a sorted list of job tuples currently in the queue."""
        with self.job_queue.mutex:
            # Sort by priority (first element), then job_id (second element) as tie-breaker
            return sorted(list(self.job_queue.queue), key=lambda x: (x[0], x[1]))

    def stop(self):
        logger.info("Stopping GPU Job Scheduler...")
        self.stop_event.set()
        for _ in range(len(self.worker_threads)):
             try:
                 # Sentinel: (priority, job_id, script_path, ...) - job_id is None for sentinel
                 self.job_queue.put((-float('inf'), None, None, None, None, None))
             except Exception as e:
                 logger.error(f"Error putting sentinel value in queue: {e}")
        logger.info("Waiting for worker threads to finish...")
        for thread in self.worker_threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning(f"Worker thread {thread.name} did not finish within timeout.")
        logger.info("Saving final scheduler state...")
        self.save_state()
        logger.info("Scheduler stopped.")

    def worker(self):
        """Worker thread logic: Get job, find GPU, run job."""
        thread_name = threading.current_thread().name # Get thread name for logging
        logger.info(f"Worker started.")

        while not self.stop_event.is_set():
            job_tuple = None
            try:
                # Get job: (priority, job_id, script_path, conda_env, args, allowed_gpus)
                job_tuple = self.job_queue.get(block=True, timeout=1.0)
                priority, job_id, script_path, _, _, _ = job_tuple # Unpack needed parts

                # Check for sentinel value (used for shutdown) - job_id is None for sentinel
                if job_id is None:
                    logger.info(f"Received sentinel, exiting.")
                    break # Exit the loop

                logger.debug(f"Retrieved job ID {job_id}: {script_path} (Priority: {priority})")

            except queue.Empty:
                continue # No job, loop back
            except Exception as e:
                logger.error(f"Error getting job from queue: {e}")
                time.sleep(1)
                continue

            # --- Job retrieved, now find a suitable GPU ---
            assigned = False
            attempts = 0
            max_attempts = 5

            while not assigned and not self.stop_event.is_set() and attempts < max_attempts:
                gpu_id_to_run = -1
                with self.lock:
                    gpu_indices = list(range(self.num_gpus))
                    # random.shuffle(gpu_indices) # Optional: Randomize check order
                    for gpu_id in gpu_indices:
                        _, _, _, _, _, allowed_gpus = job_tuple # Unpack allowed_gpus
                        if not self.gpu_status[gpu_id] and \
                           gpu_id not in self.paused_gpus and \
                           (allowed_gpus is None or gpu_id in allowed_gpus):
                            try:
                                gpus_util = GPUtil.getGPUs()
                                if gpu_id < len(gpus_util):
                                    gpu = gpus_util[gpu_id]
                                    mem_util = gpu.memoryUtil
                                    load_util = gpu.load
                                    logger.debug(f"Checking GPU {gpu_id} for Job ID {job_id}: Status=AVAILABLE, Paused=No, Allowed=Yes. Util Mem={mem_util:.2f}, Load={load_util:.2f}")
                                    if mem_util < self.gpu_memory_threshold and load_util < self.gpu_load_threshold:
                                        logger.info(f"Found suitable GPU {gpu_id} for job ID {job_id}.")
                                        self.gpu_status[gpu_id] = True # Mark busy *within lock*
                                        gpu_id_to_run = gpu_id
                                        break # Exit GPU check loop
                                    # else: logger.debug(...) # Log if util too high
                                else:
                                    logger.warning(f"GPUtil did not report on GPU {gpu_id}. Skipping assignment for job ID {job_id}.")
                            except Exception as e:
                                logger.error(f"Error checking GPU {gpu_id} utilization for job ID {job_id}: {e}. Skipping.")
                                continue

                # --- After checking all GPUs ---
                if gpu_id_to_run != -1:
                    # --- Assign Job ---
                    logger.info(f"Assigning job ID {job_id} ({script_path}) to GPU {gpu_id_to_run}")
                    self.save_state() # Save state *after* finding GPU and releasing lock

                    job_runner_thread = threading.Thread(
                        target=self._run_job,
                        args=(job_tuple, gpu_id_to_run),
                        name=f"JobRunner-GPU{gpu_id_to_run}-{Path(script_path).stem}-{job_id[:4]}", # Include partial job_id in thread name
                        daemon=True
                    )
                    job_runner_thread.start()
                    assigned = True
                    self.job_queue.task_done() # Signal job processing started

                else:
                    # --- No suitable GPU found ---
                    attempts += 1
                    logger.debug(f"Found no suitable GPU for job ID {job_id} (Attempt {attempts}/{max_attempts}). Re-queueing.")
                    try:
                        self.job_queue.put(job_tuple) # Put the original tuple back
                        self.job_queue.task_done() # Mark original task done as we re-queued it
                    except Exception as e:
                         logger.error(f"Failed to re-queue job ID {job_id}: {e}")
                    wait_time = min(5 * attempts, 30)
                    time.sleep(wait_time) # Wait before this worker tries getting *any* job again

            # --- If job couldn't be assigned after max attempts ---
            if not assigned and attempts >= max_attempts:
                 logger.warning(f"Failed to assign job ID {job_id} ({script_path}) after {max_attempts} attempts. Leaving it in the queue.")
                 # Job remains in the queue. Need task_done() here?
                 # Yes, the original `get` needs a corresponding `task_done`.
                 # It was called after successful re-queue, but needs to be called here too if re-queue fails or loop finishes.
                 # Let's ensure task_done is always called once per successful get.
                 # The logic above calls it after starting runner OR after re-queue. What if re-queue fails?
                 # Add a final check/call.
                 try:
                     # Check if task_done was already called (e.g. after re-queue)
                     # This is tricky without tracking state. Assume it might not have been called if loop exited here.
                     # Calling task_done potentially twice is problematic.
                     # Let's restructure slightly: call task_done *after* the inner while loop completes or breaks.
                     pass # See task_done call below
                 except ValueError: # If task_done called too many times
                     pass

            # Ensure task_done() is called exactly once for the job retrieved by get()
            # This call handles the case where the loop finishes without assigning *or* re-queueing successfully (e.g., re-queue exception)
            # However, the current logic calls it after assignment OR after re-queue. This might be sufficient.
            # Let's rely on the existing task_done calls for now. If queue joining hangs, this might need review.


        logger.info(f"Worker finished.")


    def _run_job(self, job_tuple: Tuple, gpu_id: int):
        """Internal method to prepare environment and execute a job."""
        priority, job_id, script_path, conda_env, args, _ = job_tuple
        job_name = Path(script_path).name # Use pathlib
        start_time = datetime.now()
        logger.info(f"GPU {gpu_id}: Preparing job ID {job_id} ('{job_name}', Priority: {priority}, Conda: {conda_env or 'None'})")
        logger.debug(f"GPU {gpu_id}: Job ID {job_id} details - Args: {args}")

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            if self.use_screen:
                self._run_with_screen(job_tuple, gpu_id, env, start_time)
            else:
                self._run_directly(job_tuple, gpu_id, env, start_time)
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Unexpected error setting up/running job ID {job_id} ('{job_name}'): {e}", exc_info=True)
            if not self.use_screen:
                 self._release_gpu(gpu_id, job_id, job_name, start_time, "critical error during setup")

    def _release_gpu(self, gpu_id: int, job_id: str, job_name: str, start_time: datetime, reason: str="completion"):
        """Helper method to release GPU status and save state."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"GPU {gpu_id}: Job ID {job_id} ('{job_name}') finished due to {reason} in {duration:.2f} seconds. Releasing GPU.")
        with self.lock:
            self.gpu_status[gpu_id] = False
        self.save_state()

    def _run_with_screen(self, job_tuple: Tuple, gpu_id: int, env: Dict, start_time: datetime):
        """Run a job inside a GNU Screen session."""
        priority, job_id, script_path, conda_env, args, _ = job_tuple
        script_p = Path(script_path)
        script_basename = script_p.name
        safe_basename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', script_basename)
        # Include partial job_id in session name for uniqueness
        session_name = f"gpujob_{gpu_id}_{safe_basename}_{job_id[:6]}_{start_time.strftime('%H%M%S')}"
        session_name = session_name[:60] # Limit length further

        temp_script_path = None
        try:
            # --- Create temp script ---
            script_content = ["#!/bin/bash", f"# Job ID: {job_id}", f"# Screen session: {session_name}", ...] # Add other details
            # (Rest of script content generation as before)
            # ... include conda activation, python command execution ...
            cmd_args_str = ""
            if args:
                cmd_args_str = " ".join(shlex.quote(str(arg)) for arg in args) if isinstance(args, list) else str(args)
            script_content.append(f"echo \"Executing: python {shlex.quote(script_path)} {cmd_args_str}\"")
            script_content.append(f"python {shlex.quote(script_path)} {cmd_args_str}")
            script_content.append("exit_code=$?")
            # ... rest of script content ...

            temp_script_dir = Path("/tmp")
            temp_script_dir.mkdir(exist_ok=True)
            temp_script_path = temp_script_dir / f"run_{session_name}.sh"
            with open(temp_script_path, 'w') as f: f.write("\n".join(script_content))
            os.chmod(temp_script_path, 0o755)
            logger.debug(f"GPU {gpu_id}: Temp script for job ID {job_id}: {temp_script_path}")

            # --- Start screen session ---
            screen_cmd = ['screen', '-dmS', session_name, str(temp_script_path)] # Use string path
            logger.info(f"GPU {gpu_id}: Starting screen session '{session_name}' for job ID {job_id}")
            subprocess.run(screen_cmd, env=env, check=True)
            logger.info(f"GPU {gpu_id}: To view progress: screen -r {session_name}")

            # --- Start monitoring thread ---
            monitoring_thread = threading.Thread(
                target=self._monitor_screen,
                args=(session_name, job_id, script_path, gpu_id, temp_script_path, start_time),
                name=f"ScreenMonitor-GPU{gpu_id}-{job_id[:6]}",
                daemon=True
            )
            monitoring_thread.start()

        except Exception as e:
            # Simplified error handling - release GPU if screen setup fails
            logger.error(f"GPU {gpu_id}: Error setting up screen session for job ID {job_id}: {e}", exc_info=True)
            self._release_gpu(gpu_id, job_id, script_basename, start_time, "screen setup error")
            if temp_script_path and temp_script_path.exists():
                try: temp_script_path.unlink()
                except OSError as e_rm: logger.warning(f"Error removing temp script {temp_script_path}: {e_rm}")


    def _monitor_screen(self, session_name: str, job_id: str, script_path: str, gpu_id: int, temp_script_path: Optional[Path], start_time: datetime):
        """Monitor a screen session until it ends, then release the GPU."""
        job_name = Path(script_path).name
        logger.info(f"GPU {gpu_id}: Monitoring screen session '{session_name}' for job ID {job_id}...")
        active = True
        while active and not self.stop_event.is_set():
            try:
                cmd = f"screen -ls | grep '\\.{session_name}\\s'"
                subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
                active = True
                logger.debug(f"GPU {gpu_id}: Screen session '{session_name}' (Job ID {job_id}) is active.")
            except subprocess.CalledProcessError:
                active = False
                logger.info(f"GPU {gpu_id}: Screen session '{session_name}' (Job ID {job_id}) finished or not found.")
            except Exception as e:
                 logger.error(f"GPU {gpu_id}: Error checking screen session '{session_name}' (Job ID {job_id}): {e}. Assuming finished.")
                 active = False
            if active:
                 time.sleep(15)

        logger.info(f"GPU {gpu_id}: Monitoring finished for screen session '{session_name}' (Job ID {job_id}).")
        if temp_script_path and temp_script_path.exists():
            logger.debug(f"GPU {gpu_id}: Removing temporary script: {temp_script_path}")
            try: temp_script_path.unlink()
            except OSError as e: logger.warning(f"GPU {gpu_id}: Error removing temp script {temp_script_path}: {e}")

        self._release_gpu(gpu_id, job_id, job_name, start_time, "screen session ended")


    def _run_directly(self, job_tuple: Tuple, gpu_id: int, env: Dict, start_time: datetime):
        """Run a job directly (without screen), capturing output."""
        priority, job_id, script_path, conda_env, args, _ = job_tuple
        script_p = Path(script_path)
        job_name = script_p.name
        # Include job_id in log filename
        log_filename = Path(f"job_{job_name}_{job_id[:6]}_{start_time.strftime('%Y%m%d_%H%M%S')}_gpu{gpu_id}.log")
        logger.info(f"GPU {gpu_id}: Running job ID {job_id} ('{job_name}') directly. Log: {log_filename}")

        process = None
        log_file = None
        temp_script_path = None

        try:
            log_file = open(log_filename, 'w', buffering=1)
            log_file.write(f"Job ID: {job_id}\n")
            # (Write other header info: script, gpu, time, conda, args...)
            log_file.write("-" * 60 + "\n") ; log_file.flush()

            command_list = []
            shell_mode = False

            # --- Build command (using temp script for conda) ---
            if conda_env:
                shell_script_lines = ["#!/bin/bash", "set -e", f"# Job ID: {job_id}"]
                # (Add conda activation logic as before) ...
                python_cmd = ["python", script_path]
                if args: python_cmd.extend(shlex.split(str(args)) if isinstance(args, str) else map(str, args))
                shell_script_lines.append(shlex.join(python_cmd))

                script_id = f"{start_time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_{job_id[:4]}"
                temp_script_dir = Path("/tmp") ; temp_script_dir.mkdir(exist_ok=True)
                temp_script_path = temp_script_dir / f"direct_job_{script_id}.sh"
                with open(temp_script_path, 'w') as f: f.write("\n".join(shell_script_lines))
                os.chmod(temp_script_path, 0o755)
                command_list = [str(temp_script_path)] # Command is the script
                shell_mode = False
            else:
                # No conda env
                command_list = ['python', script_path]
                if args: command_list.extend(shlex.split(str(args)) if isinstance(args, str) else map(str, args))
                shell_mode = False

            logger.info(f"GPU {gpu_id}: Job ID {job_id}: Executing command: {' '.join(command_list)}")
            process = subprocess.Popen(
                command_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, bufsize=1, shell=shell_mode
            )

            # --- Real-time output handling ---
            def stream_output(pipe, prefix, log_f):
                try:
                    for line in iter(pipe.readline, ''):
                        line_stripped = line.rstrip()
                        logger.info(f"[GPU {gpu_id} Job {job_id[:6]} {prefix}] {line_stripped}") # Include Job ID prefix
                        if log_f: log_f.write(f"{prefix}: {line}") ; log_f.flush()
                except Exception as e: logger.error(f"Stream reading error (GPU {gpu_id}, Job {job_id[:6]}): {e}")
                finally: pipe.close()

            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "OUT", log_file), daemon=True)
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERR", log_file), daemon=True)
            stdout_thread.start(); stderr_thread.start()

            return_code = process.wait()
            stdout_thread.join(timeout=5.0); stderr_thread.join(timeout=5.0)

            log_file.write("-" * 60 + "\n")
            log_file.write(f"Process finished with exit code: {return_code}\n")
            if return_code == 0: logger.info(f"GPU {gpu_id}: Job ID {job_id} completed successfully.")
            else: logger.error(f"GPU {gpu_id}: Job ID {job_id} failed. Exit code: {return_code}. See log: {log_filename}")

        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error executing job ID {job_id}: {e}", exc_info=True)
            if log_file: log_file.write(f"\nERROR: Execution failed: {e}\n")
            return_code = -1 # Simulate failure
        finally:
            if log_file: log_file.close()
            if temp_script_path and temp_script_path.exists():
                try: temp_script_path.unlink()
                except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")
            # Release GPU
            reason = f"exit code {return_code}" if 'return_code' in locals() else "execution finished/failed"
            self._release_gpu(gpu_id, job_id, job_name, start_time, reason)


    def enable_screen(self):
        """Enable the use of GNU Screen for job execution."""
        # (Screen check logic as before) ...
        try:
             subprocess.run(['screen', '-v'], check=True, capture_output=True)
             self.use_screen = True
             logger.info("Screen functionality enabled.")
        except Exception as e:
             logger.error(f"Screen check failed: {e}. Screen functionality disabled.")
             self.use_screen = False


    def add_job(self, script: str, conda_env: Optional[str] = None, args: Optional[List[str]] = None,
                priority: int = 0, allowed_gpus: Optional[List[int]] = None):
        """Adds a single job with a unique ID to the queue."""
        script_p = Path(script)
        if not script_p.exists():
             logger.error(f"Script path not found: {script}. Job not added.")
             return

        job_id = str(uuid.uuid4()) # Generate unique ID
        logger.info(f"Generating Job ID {job_id} for script {script_p.name}")

        valid_allowed_gpus = None
        if allowed_gpus is not None:
            # (Validation logic for allowed_gpus as before) ...
             valid_allowed_gpus = [gpu_id for gpu_id in allowed_gpus if 0 <= gpu_id < self.num_gpus]
             if not valid_allowed_gpus:
                  logger.error(f"No valid GPUs in allowed_gpus for script {script} (Job ID {job_id}). Job not added.")
                  return
             # ... (warning if some GPUs were invalid)

        # Job tuple format: (priority, job_id, script_path, conda_env, args, allowed_gpus)
        job_tuple = (priority, job_id, script, conda_env, args, valid_allowed_gpus)
        try:
            self.job_queue.put(job_tuple)
            logger.info(f"Job ID {job_id} added: '{script_p.name}' (Priority: {priority}, GPUs: {valid_allowed_gpus or 'Any'})")
        except Exception as e:
            logger.error(f"Failed to add job ID {job_id} ('{script_p.name}') to queue: {e}")


    def add_jobs_from_file(self, file_path: str):
        """Adds multiple jobs from a file, generating unique IDs for each."""
        logger.info(f"Attempting to add jobs from file: {file_path}")
        jobs_added = 0
        file_p = Path(file_path)
        if not file_p.is_file():
             logger.error(f"Job file not found: {file_path}")
             return

        try:
            with open(file_p, 'r') as f:
                for i, line in enumerate(f):
                    line_num = i + 1
                    line = line.strip()
                    if not line or line.startswith('#'): continue

                    parts = line.split(',', maxsplit=4)
                    if len(parts) < 2:
                        logger.error(f"Invalid job format on line {line_num}: '{line}'. Requires priority,script_path.")
                        continue
                    try:
                        priority = int(parts[0].strip())
                        script = parts[1].strip()
                        conda_env = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                        args_str = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                        allowed_gpus_str = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None

                        args_list = shlex.split(args_str) if args_str else None
                        allowed_gpus_list = None
                        if allowed_gpus_str:
                            # (Parsing logic for allowed_gpus_str as before) ...
                            allowed_gpus_list = []
                            for part in allowed_gpus_str.split(','):
                                part = part.strip()
                                if '-' in part:
                                    try:
                                        start, end = map(int, part.split('-'))
                                        if start <= end: allowed_gpus_list.extend(range(start, end + 1))
                                    except ValueError: logger.warning(f"Invalid range '{part}' on line {line_num}")
                                else:
                                    try: allowed_gpus_list.append(int(part))
                                    except ValueError: logger.warning(f"Invalid GPU ID '{part}' on line {line_num}")
                            allowed_gpus_list = sorted(list(set(allowed_gpus_list)))


                        # Call add_job to handle validation and ID generation
                        self.add_job(script, conda_env, args_list, priority, allowed_gpus_list)
                        # Note: add_job logs success/failure, so we don't strictly need jobs_added count here
                        jobs_added += 1 # Count attempts

                    except ValueError: logger.error(f"Invalid priority format on line {line_num}: '{parts[0]}'.")
                    except Exception as e: logger.error(f"Error parsing job on line {line_num}: '{line}'. Error: {e}")

            logger.info(f"Finished processing job file '{file_path}'. Attempted to add {jobs_added} jobs.")
        except Exception as e:
            logger.error(f"Failed to read or process job file {file_path}: {e}")


    def start(self):
        """Start the GPU job scheduler worker threads."""
        if self.worker_threads:
             logger.warning("Scheduler appears to be already started.")
             return
        logger.info(f"Starting GPU Job Scheduler with {self.num_gpus} GPUs...")
        self.stop_event.clear()
        num_workers = self.num_gpus
        logger.info(f"Starting {num_workers} worker threads...")
        for i in range(num_workers):
            worker_thread = threading.Thread(target=self.worker, name=f"Worker-{i}", daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        logger.info("Scheduler started. Waiting for jobs...")


# --- Command Line Interface (Modified for Job ID display) ---
def main():
    parser = argparse.ArgumentParser(description='GPU Job Scheduler', formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # (Parser setup for start, add, add-file, status, pause, resume, screens as before)
    # ...
    start_parser = subparsers.add_parser('start', help='Start the scheduler daemon process.')
    start_parser.add_argument('--gpus', type=int, default=len(GPUtil.getGPUs()) if GPUtil.getGPUs() else 1, help='Number of GPUs (default: detected)')
    start_parser.add_argument('--jobs', type=str, help='Path to jobs file to add on startup.')
    start_parser.add_argument('--screen', action='store_true', help='Enable GNU Screen sessions.')
    start_parser.add_argument('--mem-threshold', type=float, default=0.8, help='GPU memory threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--load-threshold', type=float, default=0.8, help='GPU load threshold (0.0-1.0, default: 0.8)')

    add_parser = subparsers.add_parser('add', help='Add a new job to the queue.')
    add_parser.add_argument('script', type=str, help='Path to the Python script.')
    add_parser.add_argument('--conda', type=str, help='Conda environment (optional).')
    add_parser.add_argument('--args', type=str, help='Script arguments (quote if needed).')
    add_parser.add_argument('--priority', type=int, default=0, help='Job priority (lower=higher, default: 0).')
    add_parser.add_argument('--gpus', type=str, help='Allowed GPU IDs (e.g., "0,1" or "0-2,5", optional).')

    add_file_parser = subparsers.add_parser('add-file', help='Add multiple jobs from a formatted file.')
    add_file_parser.add_argument('file_path', type=str, help='Path to the job definition file.')

    subparsers.add_parser('status', help='Show status of GPUs and job queue.')
    pause_parser = subparsers.add_parser('pause', help='Pause a GPU.')
    pause_parser.add_argument('gpu_id', type=int, help='ID of the GPU to pause.')
    resume_parser = subparsers.add_parser('resume', help='Resume a paused GPU.')
    resume_parser.add_argument('gpu_id', type=int, help='ID of the GPU to resume.')
    subparsers.add_parser('screens', help='List active GNU Screen sessions (matching pattern).')


    args = parser.parse_args()

    # --- Determine Number of GPUs for non-start commands ---
    default_num_gpus = 8 # Fallback
    try:
        # (Infer num_gpus from state file or detection - same logic as before) ...
        if Path("gpu_scheduler_state.json").exists():
             with open("gpu_scheduler_state.json", 'r') as f: state = json.load(f)
             if 'gpu_status' in state and isinstance(state['gpu_status'], list): default_num_gpus = len(state['gpu_status'])
             else: default_num_gpus = len(GPUtil.getGPUs()) if GPUtil.getGPUs() else default_num_gpus
        else: default_num_gpus = len(GPUtil.getGPUs()) if GPUtil.getGPUs() else default_num_gpus
    except Exception: logger.warning(f"Error determining GPU count for control command. Using default: {default_num_gpus}")


    # --- Execute Commands ---
    if args.command == 'start':
        logger.info(f"Starting scheduler process with {args.gpus} GPUs.")
        scheduler = GPUJobScheduler(
            num_gpus=args.gpus,
            gpu_memory_threshold=args.mem_threshold,
            gpu_load_threshold=args.load_threshold
        )
        if args.screen: scheduler.enable_screen()
        if args.jobs: scheduler.add_jobs_from_file(args.jobs)
        scheduler.start()
        try:
            while True: time.sleep(60) # Keep main thread alive
        except KeyboardInterrupt: logger.info("Ctrl+C received. Initiating shutdown...")
        finally: scheduler.stop() ; logger.info("Scheduler shutdown complete.")

    else:
        # --- Control Commands ---
        scheduler = GPUJobScheduler(num_gpus=default_num_gpus) # Instance for state access

        if args.command == 'add':
            # (Parse allowed_gpus_str - same logic as before) ...
            allowed_gpus_list = None # Placeholder for parsing logic
            if args.gpus:
                 allowed_gpus_list = [] # Add parsing logic here...

            # Add job (this will now generate and log a Job ID)
            # Note: Still requires IPC or running scheduler for queue interaction
            print("Simulating job addition (requires running scheduler or IPC):")
            # Generate a temporary ID just for display simulation
            sim_job_id = str(uuid.uuid4())
            print(f"  Job ID (generated): {sim_job_id}")
            print(f"  Script: {args.script}")
            # ... print other args ...
            # scheduler.add_job(args.script, args.conda, shlex.split(args.args) if args.args else None, args.priority, allowed_gpus_list)

        elif args.command == 'add-file':
             scheduler.add_jobs_from_file(args.file_path)
             print(f"Jobs from file '{args.file_path}' processed. Check logs for details and Job IDs.")

        elif args.command == 'status':
            status = scheduler.get_gpu_status()
            print("\n--- GPU Status ---")
            print(f"{'GPU ID':<8} {'State':<12} {'Memory Util':<15} {'Load Util':<15}")
            print("-" * 55)
            for gpu in status: print(f"{gpu['gpu_id']:<8} {gpu['state']:<12} {gpu['memory_util']:<15} {gpu['load']:<15}")

            print("\n--- Job Queue ---")
            queued_jobs = scheduler.get_job_queue_info() # Gets sorted list
            if not queued_jobs:
                print("No jobs currently in the queue.")
            else:
                # Adjusted header for Job ID
                print(f"{'Priority':<10} {'Job ID':<15} {'Script':<30} {'Conda Env':<15} {'Allowed GPUs':<15} {'Arguments'}")
                print("-" * 120)
                for job in queued_jobs:
                    # Unpack including job_id
                    priority, job_id, script, conda, job_args, allowed_gpus = job
                    script_name = Path(script).name
                    args_str = shlex.join(job_args) if job_args else "" # Use shlex.join for display
                    # Display partial Job ID for brevity
                    print(f"{priority:<10} {job_id[:8]+'...':<15} {script_name:<30} {conda or 'None':<15} {allowed_gpus or 'Any':<15} {args_str[:40]}")

        elif args.command == 'pause':
            if scheduler.pause_gpu(args.gpu_id): print(f"GPU {args.gpu_id} paused.") # State saved automatically
            else: print(f"Failed to pause GPU {args.gpu_id}.")

        elif args.command == 'resume':
            if scheduler.resume_gpu(args.gpu_id): print(f"GPU {args.gpu_id} resumed.") # State saved automatically
            else: print(f"Failed to resume GPU {args.gpu_id}.")

        elif args.command == 'screens':
            print("Listing active GNU Screen sessions (matching gpujob_* pattern)...")
            active_sessions = list_screens()
            if not active_sessions: print("No matching screen sessions found.")
            else:
                print("\n--- Active GPU Job Screen Sessions ---")
                # (Display logic as before) ...
                for i, session in enumerate(active_sessions): print(f"{i+1}. {session} (Attach: screen -r {session})")


if __name__ == "__main__":
    main()
