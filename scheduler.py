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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_scheduler.log"),
        logging.StreamHandler()
    ]
)
# Use a specific logger name
logger = logging.getLogger("GPUJobScheduler")

# --- Helper Function ---
def list_screens():
    """List active screen sessions potentially related to GPU jobs"""
    try:
        # Get list of screen sessions
        result = subprocess.run(
            ['screen', '-list'],
            capture_output=True,
            text=True,
            check=True # Raise exception on error
        )

        # Parse screen output for sessions matching the naming convention
        # Making the pattern slightly more general but anchored
        screen_pattern = re.compile(r'^\s*(\d+\.gpujob_\d+_\S+)\s+\(.*\)', re.MULTILINE)
        matches = screen_pattern.findall(result.stdout)

        # Extract just the session name (e.g., "gpujob_0_script_...")
        session_names = []
        for match in matches:
             # Extract the part after the initial PID and dot (e.g., 12345.gpujob...)
             name_part = match.split('.', 1)[-1]
             session_names.append(name_part)
        return session_names
    except FileNotFoundError:
        logger.warning("GNU Screen command ('screen') not found. Cannot list sessions.")
        return []
    except subprocess.CalledProcessError as e:
        # Handle cases like "No Sockets found" gracefully
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
    def __init__(self, num_gpus=8, gpu_memory_threshold=0.8, gpu_load_threshold=0.8):
        """
        Initializes the GPU Job Scheduler.

        Args:
            num_gpus (int): Number of GPUs available for scheduling.
            gpu_memory_threshold (float): Max memory utilization (0.0-1.0) before a GPU is considered busy.
            gpu_load_threshold (float): Max load utilization (0.0-1.0) before a GPU is considered busy.
        """
        self.use_screen = False
        self.num_gpus = num_gpus
        # Use PriorityQueue for automatic job prioritization
        # Lower priority number means higher priority
        self.job_queue = queue.PriorityQueue()
        # Status: False = available, True = busy
        self.gpu_status = [False] * num_gpus
        self.gpu_memory_threshold = gpu_memory_threshold
        self.gpu_load_threshold = gpu_load_threshold
        self.lock = threading.Lock() # Lock for accessing shared state (gpu_status, paused_gpus, state file)
        self.stop_event = threading.Event() # Signal for graceful shutdown
        self.worker_threads = []
        self.paused_gpus = set() # Track paused (manually disabled) GPUs
        self.state_file = "gpu_scheduler_state.json"

        # Load previous state if available
        self.load_state()
        # Apply paused status from loaded state
        self._apply_paused_state()

    def _apply_paused_state(self):
        """Ensure paused GPUs are correctly reflected internally."""
        # No lock needed here as it's called during init or after loading state
        logger.info(f"Applying paused state for GPUs: {self.paused_gpus}")
        # Note: gpu_status reflects BUSY/AVAILABLE from the scheduler's perspective.
        # Paused state is checked separately in the worker logic.

    def load_state(self):
        """Load GPU scheduler state (status, paused GPUs) from a file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                # --- Load gpu_status (Busy/Available) ---
                loaded_status = state.get('gpu_status', [])
                # Adjust list size if current config differs from saved state
                if len(loaded_status) == self.num_gpus:
                    self.gpu_status = loaded_status
                elif len(loaded_status) < self.num_gpus:
                    # Add False for new GPUs
                    self.gpu_status = loaded_status + [False] * (self.num_gpus - len(loaded_status))
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Added new GPUs as AVAILABLE.")
                else:
                    # Truncate if fewer GPUs now
                    self.gpu_status = loaded_status[:self.num_gpus]
                    logger.warning(f"Loaded state had {len(loaded_status)} GPUs, configured for {self.num_gpus}. Ignoring extra GPUs from state.")

                # --- Load paused GPUs ---
                # Ensure paused_gpus contains only valid IDs for the current config
                loaded_paused = state.get('paused_gpus', [])
                self.paused_gpus = set(gpu_id for gpu_id in loaded_paused if 0 <= gpu_id < self.num_gpus)

                logger.info(f"State loaded successfully. Status: {self.gpu_status}, Paused: {self.paused_gpus}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode state file '{self.state_file}': {e}. Starting fresh.")
                self.gpu_status = [False] * self.num_gpus
                self.paused_gpus = set()
            except Exception as e:
                logger.error(f"Failed to load state from '{self.state_file}': {e}. Starting fresh.")
                self.gpu_status = [False] * self.num_gpus
                self.paused_gpus = set()
        else:
            logger.info("No state file found. Starting with all GPUs available.")
            self.gpu_status = [False] * self.num_gpus
            self.paused_gpus = set()

    def save_state(self):
        """Save the current scheduler state (status, paused GPUs) to file."""
        # This should be called whenever gpu_status or paused_gpus changes.
        # Acquire lock to ensure atomicity of reading status/paused and writing file.
        with self.lock:
            try:
                state = {
                    # Save a copy of the lists/sets
                    "gpu_status": list(self.gpu_status),
                    "paused_gpus": list(self.paused_gpus)
                }
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4) # Use indent for readability
                logger.debug("Scheduler state saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save state to '{self.state_file}': {e}")

    def pause_gpu(self, gpu_id):
        """Pause a specific GPU, preventing new jobs from being assigned to it."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False

        with self.lock:
            if gpu_id in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} is already paused.")
                return True # Idempotent
            self.paused_gpus.add(gpu_id)
            logger.info(f"GPU {gpu_id} paused. It will not accept new jobs.")

        # Save state *after* releasing lock
        self.save_state()
        return True

    def resume_gpu(self, gpu_id):
        """Resume a paused GPU, allowing new jobs to be assigned to it."""
        if not 0 <= gpu_id < self.num_gpus:
            logger.error(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.num_gpus - 1}.")
            return False

        with self.lock:
            if gpu_id not in self.paused_gpus:
                logger.warning(f"GPU {gpu_id} was not paused.")
                return True # Idempotent
            self.paused_gpus.remove(gpu_id)
            logger.info(f"GPU {gpu_id} resumed. It can now accept new jobs.")

        # Save state *after* releasing lock
        self.save_state()
        return True

    def get_gpu_status(self):
        """Get the current status (state, utilization) of all GPUs."""
        status_list = []
        gpus_util = []
        try:
            # Get utilization data once
            gpus_util = GPUtil.getGPUs()
        except Exception as e:
            logger.error(f"Could not get GPU utilization via GPUtil: {e}")

        # Use lock to read the current internal state consistently
        with self.lock:
            current_gpu_status = list(self.gpu_status) # Make a copy
            current_paused_gpus = set(self.paused_gpus) # Make a copy

        for gpu_id in range(self.num_gpus):
            # Determine state: PAUSED > BUSY > AVAILABLE
            if gpu_id in current_paused_gpus:
                state = "PAUSED"
            elif current_gpu_status[gpu_id]:
                state = "BUSY"
            else:
                state = "AVAILABLE"

            # Get utilization if available
            memory_util_str = "N/A"
            load_str = "N/A"
            if gpu_id < len(gpus_util):
                try:
                    gpu = gpus_util[gpu_id]
                    memory_util_str = f"{gpu.memoryUtil * 100:.1f}%"
                    load_str = f"{gpu.load * 100:.1f}%"
                except Exception as e:
                    # Log error for specific GPU if util fails
                    logger.error(f"Error getting utilization for GPU {gpu_id}: {e}")

            status_list.append({
                "gpu_id": gpu_id,
                "state": state, # Reflects scheduler view + paused status
                "memory_util": memory_util_str,
                "load": load_str
            })

        return status_list

    def get_job_queue_info(self):
        """Returns a list of jobs currently in the queue."""
        # Accessing queue.queue directly is not recommended for PriorityQueue
        # We can create a temporary list for display purposes
        with self.job_queue.mutex: # Use the queue's internal lock
            return sorted(list(self.job_queue.queue), key=lambda x: x[0]) # Sort by priority

    def stop(self):
        """Stop the scheduler gracefully."""
        logger.info("Stopping GPU Job Scheduler...")
        self.stop_event.set() # Signal all worker threads to stop

        # Add sentinel values to the queue to unblock workers waiting on queue.get()
        # One sentinel per worker thread
        for _ in range(len(self.worker_threads)):
             try:
                 # Use highest priority to ensure sentinels are processed quickly
                 # None signifies a sentinel value
                 self.job_queue.put((-float('inf'), None, None, None, None))
             except Exception as e:
                 logger.error(f"Error putting sentinel value in queue: {e}")


        logger.info("Waiting for worker threads to finish...")
        for thread in self.worker_threads:
            thread.join(timeout=10.0) # Wait with timeout
            if thread.is_alive():
                logger.warning(f"Worker thread {thread.name} did not finish within timeout.")

        # Final state save after workers have stopped
        logger.info("Saving final scheduler state...")
        self.save_state()
        logger.info("Scheduler stopped.")

    def worker(self):
        """Worker thread logic: Get job, find GPU, run job."""
        thread_name = threading.current_thread().name
        logger.info(f"{thread_name} started.")

        while not self.stop_event.is_set():
            job_tuple = None
            try:
                # Get the highest priority job. Blocks if queue is empty.
                # Timeout allows checking stop_event periodically.
                job_tuple = self.job_queue.get(block=True, timeout=1.0)
                priority, script_path, _, _, _ = job_tuple # Unpack needed parts

                # Check for sentinel value (used for shutdown)
                if script_path is None:
                    logger.info(f"{thread_name} received sentinel, exiting.")
                    break # Exit the loop

                logger.debug(f"{thread_name} retrieved job: {script_path} (Priority: {priority})")

            except queue.Empty:
                # Queue was empty during timeout, just loop back and check stop_event
                continue
            except Exception as e:
                logger.error(f"{thread_name} error getting job from queue: {e}")
                time.sleep(1) # Avoid busy-looping on error
                continue

            # --- Job retrieved, now find a suitable GPU ---
            assigned = False
            attempts = 0
            max_attempts = 5 # Limit re-queue attempts for a job in one go

            while not assigned and not self.stop_event.is_set() and attempts < max_attempts:
                gpu_id_to_run = -1
                # Check GPUs for availability
                with self.lock: # Lock needed to check/modify gpu_status consistently
                    # Consider shuffling or round-robin for fairness? For now, sequential.
                    gpu_indices = list(range(self.num_gpus))
                    # random.shuffle(gpu_indices) # Optional: Randomize GPU check order

                    for gpu_id in gpu_indices:
                        # Check internal status, paused status, and job's allowed GPUs
                        _, _, _, _, allowed_gpus = job_tuple # Unpack allowed_gpus
                        if not self.gpu_status[gpu_id] and \
                           gpu_id not in self.paused_gpus and \
                           (allowed_gpus is None or gpu_id in allowed_gpus):

                            # --- Check real-time utilization ---
                            try:
                                gpus_util = GPUtil.getGPUs()
                                if gpu_id < len(gpus_util):
                                    gpu = gpus_util[gpu_id]
                                    mem_util = gpu.memoryUtil
                                    load_util = gpu.load
                                    logger.debug(f"Checking GPU {gpu_id}: Status=AVAILABLE, Paused=No, Allowed=Yes. Util Mem={mem_util:.2f}, Load={load_util:.2f}")

                                    if mem_util < self.gpu_memory_threshold and load_util < self.gpu_load_threshold:
                                        # --- Found suitable GPU! ---
                                        logger.info(f"{thread_name} found suitable GPU {gpu_id} for job {script_path}.")
                                        self.gpu_status[gpu_id] = True # Mark busy *within lock*
                                        gpu_id_to_run = gpu_id
                                        # Save state immediately after marking busy
                                        # self.save_state() # Moved outside lock for performance
                                        break # Exit GPU check loop
                                    else:
                                        logger.debug(f"GPU {gpu_id} available but utilization too high (Mem: {mem_util:.2f} > {self.gpu_memory_threshold:.2f} or Load: {load_util:.2f} > {self.gpu_load_threshold:.2f})")
                                else:
                                    # GPUtil didn't report on this GPU, maybe unavailable? Be cautious.
                                    logger.warning(f"GPUtil did not report on GPU {gpu_id}. Skipping assignment.")
                                    # Alternatively, could assume it's available if not reported:
                                    # logger.warning(f"GPUtil did not report on GPU {gpu_id}. Assuming available and assigning.")
                                    # self.gpu_status[gpu_id] = True
                                    # gpu_id_to_run = gpu_id
                                    # break

                            except Exception as e:
                                logger.error(f"Error checking GPU {gpu_id} utilization: {e}. Skipping assignment to this GPU.")
                                continue # Skip this GPU if check fails

                # --- After checking all GPUs ---
                if gpu_id_to_run != -1:
                    # --- Assign Job ---
                    logger.info(f"{thread_name} assigning job {script_path} (Priority: {priority}) to GPU {gpu_id_to_run}")
                    # Save state *after* finding GPU and releasing lock
                    self.save_state()

                    # Run the actual job in a separate thread.
                    # This allows the worker to quickly become available again
                    # to potentially schedule another job if more GPUs are free.
                    job_runner_thread = threading.Thread(
                        target=self._run_job,
                        args=(job_tuple, gpu_id_to_run),
                        name=f"JobRunner-GPU{gpu_id_to_run}-{os.path.basename(script_path)}",
                        daemon=True # Ensures these threads don't block program exit if main thread finishes
                    )
                    job_runner_thread.start()
                    assigned = True # Mark as assigned, break the inner loop
                    self.job_queue.task_done() # Signal that the retrieved task is being processed

                else:
                    # --- No suitable GPU found ---
                    attempts += 1
                    logger.debug(f"{thread_name} found no suitable GPU for job {script_path} (Attempt {attempts}/{max_attempts}). Will re-queue and wait.")
                    # Put the job back into the queue.
                    # Do this *outside* the lock.
                    try:
                        self.job_queue.put(job_tuple)
                        self.job_queue.task_done() # Mark original task done as we re-queued it
                    except Exception as e:
                         logger.error(f"{thread_name} failed to re-queue job {script_path}: {e}")
                         # Job might be lost if re-queue fails

                    # Wait before the *same worker* tries again for *any* job.
                    # Other workers might pick up this job sooner if a GPU frees up.
                    wait_time = min(5 * attempts, 30) # Exponential backoff capped at 30s
                    logger.debug(f"{thread_name} waiting {wait_time}s before getting next job.")
                    time.sleep(wait_time)


            # --- If job couldn't be assigned after max attempts ---
            if not assigned and attempts >= max_attempts:
                 logger.warning(f"{thread_name} failed to assign job {script_path} after {max_attempts} attempts. Leaving it in the queue.")
                 # The job remains in the queue for other workers or later attempts.


        logger.info(f"{thread_name} finished.")


    def _run_job(self, job_tuple, gpu_id):
        """
        Internal method to prepare environment and execute a job.
        This runs in a dedicated thread started by the worker.
        """
        priority, script_path, conda_env, args, _ = job_tuple # Unpack job details
        job_name = os.path.basename(script_path)
        start_time = datetime.now()
        logger.info(f"GPU {gpu_id}: Preparing job '{job_name}' (Priority: {priority}, Conda: {conda_env or 'None'})")
        logger.debug(f"GPU {gpu_id}: Job details - Args: {args}")

        # Prepare environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Optional: Add other env vars if needed
        # env['MY_CUSTOM_VAR'] = 'value'

        try:
            if self.use_screen:
                # Screen mode uses a separate monitoring thread (_monitor_screen)
                # which is responsible for releasing the GPU status.
                self._run_with_screen(script_path, conda_env, args, gpu_id, env, start_time)
            else:
                # Direct mode runs synchronously within this thread.
                # Release GPU status in the finally block here.
                self._run_directly(script_path, conda_env, args, gpu_id, env, start_time)
        except Exception as e:
            # Catch unexpected errors during job launch/execution setup
            logger.error(f"GPU {gpu_id}: Unexpected error setting up or running job '{job_name}': {e}", exc_info=True)
            # Ensure GPU is released if launch fails critically *before* _run_directly/_run_with_screen handles it
            if not self.use_screen: # Direct run handles its own release in finally
                 self._release_gpu(gpu_id, job_name, start_time, "critical error during setup")

        # Note: For screen mode, the GPU release happens in _monitor_screen's finally block.
        # For direct mode, it happens in _run_directly's finally block.

    def _release_gpu(self, gpu_id, job_name, start_time, reason="completion"):
        """Helper method to release GPU status and save state."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"GPU {gpu_id}: Job '{job_name}' finished due to {reason} in {duration:.2f} seconds. Releasing GPU.")
        with self.lock:
            self.gpu_status[gpu_id] = False
        # Save state *after* releasing lock
        self.save_state()

    def _run_with_screen(self, script_path, conda_env, args, gpu_id, env, start_time):
        """Run a job inside a GNU Screen session."""
        script_basename = os.path.basename(script_path)
        # Sanitize script basename for session name
        safe_basename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', script_basename)
        session_name = f"gpujob_{gpu_id}_{safe_basename}_{start_time.strftime('%H%M%S')}"
        # Screen names can have limitations, further sanitize if needed
        session_name = session_name[:50] # Limit length

        temp_script_path = None # Define here for cleanup
        try:
            # --- Create a temporary shell script for screen ---
            script_content = ["#!/bin/bash"]
            script_content.append(f"# Screen session: {session_name}")
            script_content.append(f"# Job: {script_path}")
            script_content.append(f"# GPU: {gpu_id}")
            script_content.append(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            script_content.append("echo \"-------------------------------------------------\"")
            script_content.append(f"echo \"Running {script_path} on GPU {gpu_id} in screen session {session_name}\"")
            script_content.append(f"echo \"Started at: {start_time}\"")
            script_content.append("echo \"To detach: Press Ctrl+A, then D\"")
            script_content.append("echo \"-------------------------------------------------\"")
            script_content.append("set -e") # Exit immediately if a command exits with a non-zero status.

            # Add conda activation if needed
            if conda_env:
                # Try to find conda path robustly
                conda_base_cmd = "conda info --base"
                try:
                     conda_base_path = subprocess.check_output(conda_base_cmd, shell=True, text=True).strip()
                     if conda_base_path:
                          conda_sh_path = os.path.join(conda_base_path, "etc/profile.d/conda.sh")
                          if os.path.exists(conda_sh_path):
                               script_content.append(f"source \"{conda_sh_path}\"")
                               script_content.append(f"conda activate \"{conda_env}\"")
                          else:
                               logger.warning(f"conda.sh not found at expected path: {conda_sh_path}. Conda activation might fail.")
                               script_content.append(f"# Warning: conda.sh not found at {conda_sh_path}")
                               script_content.append(f"conda activate \"{conda_env}\" || echo 'Failed to activate conda env {conda_env}'")
                     else:
                          logger.warning("Could not determine conda base path. Conda activation might fail.")
                          script_content.append("# Warning: Could not determine conda base path.")
                          script_content.append(f"conda activate \"{conda_env}\" || echo 'Failed to activate conda env {conda_env}'")

                except Exception as conda_e:
                     logger.warning(f"Error finding conda base path ({conda_base_cmd}): {conda_e}. Conda activation might fail.")
                     script_content.append(f"# Warning: Error finding conda: {conda_e}")
                     script_content.append(f"conda activate \"{conda_env}\" || echo 'Failed to activate conda env {conda_env}'")


            # Prepare command arguments
            cmd_args_str = ""
            if args:
                if isinstance(args, list):
                    # Quote arguments properly for shell script
                    cmd_args_str = " ".join(shlex.quote(str(arg)) for arg in args)
                else: # Assume string
                    # If it's already a string, assume user quoted it if necessary
                    cmd_args_str = str(args)

            # Add the actual command
            script_content.append(f"echo \"Executing: python {shlex.quote(script_path)} {cmd_args_str}\"")
            script_content.append(f"python {shlex.quote(script_path)} {cmd_args_str}")
            script_content.append("exit_code=$?")
            script_content.append("echo \"-------------------------------------------------\"")
            script_content.append("echo \"Job finished with exit code: $exit_code\"")
            # Add a small pause before screen exits, allowing user to see the message if attached
            script_content.append("echo \"Screen session will exit shortly...\"")
            script_content.append("sleep 5")

            # --- Write script to temp file ---
            # Use a more secure temp file creation if possible (e.g., tempfile module)
            temp_script_dir = "/tmp" # Or use tempfile.gettempdir()
            os.makedirs(temp_script_dir, exist_ok=True)
            temp_script_path = os.path.join(temp_script_dir, f"run_{session_name}.sh")

            with open(temp_script_path, 'w') as f:
                f.write("\n".join(script_content))
            os.chmod(temp_script_path, 0o755) # Make executable

            logger.debug(f"GPU {gpu_id}: Temporary script created at {temp_script_path}")

            # --- Start screen session ---
            screen_cmd = ['screen', '-dmS', session_name, temp_script_path]
            logger.info(f"GPU {gpu_id}: Starting screen session '{session_name}' with command: {' '.join(screen_cmd)}")
            subprocess.run(screen_cmd, env=env, check=True) # check=True raises error if screen fails

            logger.info(f"GPU {gpu_id}: Started job '{script_basename}' in screen session: {session_name}")
            logger.info(f"GPU {gpu_id}: To view progress, run: screen -r {session_name}")

            # --- Start monitoring thread ---
            # This thread will wait for the screen session to end and then release the GPU.
            monitoring_thread = threading.Thread(
                target=self._monitor_screen,
                args=(session_name, script_path, gpu_id, temp_script_path, start_time),
                 name=f"ScreenMonitor-GPU{gpu_id}-{safe_basename}",
                daemon=True
            )
            monitoring_thread.start()

        except FileNotFoundError:
             logger.error(f"GPU {gpu_id}: GNU Screen command ('screen') not found. Cannot start job in screen.")
             self._release_gpu(gpu_id, script_basename, start_time, "screen command not found")
             # Clean up temp script if created
             if temp_script_path and os.path.exists(temp_script_path):
                  try: os.remove(temp_script_path)
                  except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"GPU {gpu_id}: Failed to start screen session '{session_name}'. Command: '{e.cmd}'. Return code: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            logger.error(f"Stdout: {e.stdout}")
            self._release_gpu(gpu_id, script_basename, start_time, "failed to start screen")
            if temp_script_path and os.path.exists(temp_script_path):
                 try: os.remove(temp_script_path)
                 except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error setting up screen session for '{script_basename}': {e}", exc_info=True)
            # Release GPU as the monitoring thread won't start
            self._release_gpu(gpu_id, script_basename, start_time, "screen setup error")
            # Clean up temp script
            if temp_script_path and os.path.exists(temp_script_path):
                try: os.remove(temp_script_path)
                except OSError as e: logger.warning(f"Error removing temp script {temp_script_path}: {e}")


    def _monitor_screen(self, session_name, script_path, gpu_id, temp_script_path, start_time):
        """Monitor a screen session until it ends, then release the GPU."""
        job_name = os.path.basename(script_path)
        logger.info(f"GPU {gpu_id}: Monitoring screen session '{session_name}' for job '{job_name}'...")
        active = True
        while active and not self.stop_event.is_set():
            try:
                # Check if screen session exists
                # Use '-ls' or '-list' which should return 0 even if no sessions exist
                # Grep for the specific session name
                cmd = f"screen -ls | grep '\\.{session_name}\\s'"
                # We expect return code 0 if found, non-zero otherwise
                subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
                # If check_output succeeds, the session exists.
                active = True
                logger.debug(f"GPU {gpu_id}: Screen session '{session_name}' is still active.")
            except subprocess.CalledProcessError:
                # Grep returned non-zero, session likely finished or never started properly
                active = False
                logger.info(f"GPU {gpu_id}: Screen session '{session_name}' finished or not found.")
            except FileNotFoundError:
                 logger.error(f"GPU {gpu_id}: 'screen' or 'grep' command not found during monitoring. Cannot track session '{session_name}'. Releasing GPU prematurely.")
                 active = False # Stop monitoring
            except Exception as e:
                 logger.error(f"GPU {gpu_id}: Error checking screen session '{session_name}': {e}. Assuming finished.")
                 active = False # Stop monitoring on unexpected error

            if active:
                 # Wait before checking again
                 time.sleep(15) # Check every 15 seconds

        # --- Screen session ended or monitoring stopped ---
        logger.info(f"GPU {gpu_id}: Monitoring finished for screen session '{session_name}'.")

        # Clean up the temporary script file
        if temp_script_path and os.path.exists(temp_script_path):
            logger.debug(f"GPU {gpu_id}: Removing temporary script: {temp_script_path}")
            try:
                os.remove(temp_script_path)
            except OSError as e:
                logger.warning(f"GPU {gpu_id}: Error removing temp script {temp_script_path}: {e}")

        # Release the GPU - THIS IS CRITICAL
        self._release_gpu(gpu_id, job_name, start_time, "screen session ended")


    def _run_directly(self, script_path, conda_env, args, gpu_id, env, start_time):
        """Run a job directly (without screen), capturing output."""
        job_name = os.path.basename(script_path)
        log_filename = f"job_{job_name}_{start_time.strftime('%Y%m%d_%H%M%S')}_gpu{gpu_id}.log"
        logger.info(f"GPU {gpu_id}: Running job '{job_name}' directly. Output logged to: {log_filename}")

        process = None
        log_file = None
        temp_script_path = None # For conda activation script

        try:
            # Open log file (line buffered)
            log_file = open(log_filename, 'w', buffering=1)
            log_file.write(f"Starting job: {job_name}\n")
            log_file.write(f"Script: {script_path}\n")
            log_file.write(f"GPU ID: {gpu_id}\n")
            log_file.write(f"Start Time: {start_time}\n")
            log_file.write(f"Conda Env: {conda_env or 'None'}\n")
            log_file.write(f"Arguments: {args}\n")
            log_file.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n")
            log_file.write("-" * 60 + "\n")
            log_file.flush()

            command_list = []
            shell_mode = False # Default to False unless using temp script

            # --- Build command ---
            if conda_env:
                # --- Use a temporary shell script for conda activation ---
                shell_script_lines = ["#!/bin/bash", "set -e"] # Exit on error

                # Find conda and activate
                try:
                     conda_base_path = subprocess.check_output("conda info --base", shell=True, text=True).strip()
                     if conda_base_path:
                          conda_sh_path = os.path.join(conda_base_path, "etc/profile.d/conda.sh")
                          if os.path.exists(conda_sh_path):
                               shell_script_lines.append(f"source \"{conda_sh_path}\"")
                               shell_script_lines.append(f"conda activate \"{conda_env}\"")
                          else: shell_script_lines.append(f"# Warning: conda.sh not found at {conda_sh_path}")
                     else: shell_script_lines.append("# Warning: Could not determine conda base path.")
                except Exception as conda_e: shell_script_lines.append(f"# Warning: Error finding conda: {conda_e}")

                # Prepare python command and arguments
                python_cmd = ["python", script_path]
                if args:
                    if isinstance(args, list):
                        python_cmd.extend(args)
                    else: # Assume string, split carefully
                         python_cmd.extend(shlex.split(str(args)))

                # Add the quoted python command to the shell script
                shell_script_lines.append("echo \"Executing Python script...\"")
                shell_script_lines.append(shlex.join(python_cmd)) # Use shlex.join for proper quoting

                # Write temp script
                script_id = f"{start_time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
                temp_script_dir = "/tmp"
                os.makedirs(temp_script_dir, exist_ok=True)
                temp_script_path = os.path.join(temp_script_dir, f"direct_job_{script_id}.sh")
                with open(temp_script_path, 'w') as f:
                    f.write("\n".join(shell_script_lines))
                os.chmod(temp_script_path, 0o755)

                command_list = [temp_script_path] # Command is now the script path
                shell_mode = False # We run the script directly, not via shell
                logger.debug(f"GPU {gpu_id}: Using temp script for conda activation: {temp_script_path}")

            else:
                # --- No conda env, run python directly ---
                command_list = ['python', script_path]
                if args:
                    if isinstance(args, list):
                        command_list.extend(args)
                    else: # Assume string, split carefully
                        command_list.extend(shlex.split(str(args)))
                shell_mode = False
                logger.debug(f"GPU {gpu_id}: Running python directly.")


            logger.info(f"GPU {gpu_id}: Executing command: {' '.join(command_list)}")

            # --- Start the process ---
            process = subprocess.Popen(
                command_list,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True, # Decode output as text
                bufsize=1, # Line buffered
                shell=shell_mode # Should typically be False now
            )

            # --- Real-time output handling ---
            def stream_output(pipe, prefix, log_f):
                try:
                    for line in iter(pipe.readline, ''):
                        line_stripped = line.rstrip()
                        # Log to scheduler console/log
                        logger.info(f"[GPU {gpu_id} {prefix}] {line_stripped}")
                        # Log to job-specific log file
                        if log_f:
                            log_f.write(f"{prefix}: {line}") # Keep newline in file
                            log_f.flush()
                except Exception as e:
                     logger.error(f"Error reading {prefix} stream for GPU {gpu_id}: {e}")
                finally:
                    pipe.close() # Ensure pipe is closed

            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "OUT", log_file), daemon=True)
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERR", log_file), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process completion
            return_code = process.wait()

            # Wait for output threads to finish reading any remaining output
            stdout_thread.join(timeout=5.0)
            stderr_thread.join(timeout=5.0)

            # Log final status
            log_file.write("-" * 60 + "\n")
            log_file.write(f"Process finished with exit code: {return_code}\n")
            log_file.write(f"End Time: {datetime.now()}\n")

            if return_code == 0:
                logger.info(f"GPU {gpu_id}: Successfully completed job '{job_name}'. Exit code: {return_code}")
            else:
                logger.error(f"GPU {gpu_id}: Job '{job_name}' failed. Exit code: {return_code}. See log: {log_filename}")

        except FileNotFoundError as e:
             logger.error(f"GPU {gpu_id}: Command not found for job '{job_name}'. Ensure Python/script path is correct. Error: {e}")
             if log_file: log_file.write(f"\nERROR: Command not found: {e}\n")
             # Return code simulation for failure indication
             return_code = -1
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error executing job '{job_name}': {e}", exc_info=True)
            if log_file: log_file.write(f"\nERROR: Execution failed: {e}\n")
            # Return code simulation for failure indication
            return_code = -1
        finally:
            # --- Cleanup and GPU Release ---
            if log_file:
                log_file.close()
            if temp_script_path and os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                    logger.debug(f"GPU {gpu_id}: Removed temporary script: {temp_script_path}")
                except OSError as e:
                    logger.warning(f"GPU {gpu_id}: Error removing temp script {temp_script_path}: {e}")

            # Release the GPU - THIS IS CRITICAL for direct mode
            reason = f"exit code {return_code}" if 'return_code' in locals() else "execution finished/failed"
            self._release_gpu(gpu_id, job_name, start_time, reason)


    def enable_screen(self):
        """Enable the use of GNU Screen for job execution."""
        # Check if screen command exists
        try:
             subprocess.run(['screen', '-v'], check=True, capture_output=True)
             self.use_screen = True
             logger.info("Screen functionality enabled. Jobs will run in detached screen sessions.")
        except FileNotFoundError:
             logger.error("GNU Screen command ('screen') not found. Screen functionality disabled.")
             self.use_screen = False
        except subprocess.CalledProcessError as e:
             logger.error(f"Error checking screen version: {e}. Screen functionality disabled.")
             self.use_screen = False
        except Exception as e:
             logger.error(f"Unexpected error checking for screen: {e}. Screen functionality disabled.")
             self.use_screen = False


    def add_job(self, script, conda_env=None, args=None, priority=0, allowed_gpus=None):
        """
        Add a single job to the priority queue.

        Args:
            script (str): Path to the Python script to execute.
            conda_env (str, optional): Name of the conda environment to activate. Defaults to None.
            args (str or list, optional): Arguments to pass to the script. Defaults to None.
            priority (int, optional): Job priority (lower number = higher priority). Defaults to 0.
            allowed_gpus (list[int], optional): List of specific GPU IDs this job can run on.
                                               Defaults to None (can run on any GPU).
        """
        if not os.path.exists(script):
             logger.error(f"Script path not found: {script}. Job not added.")
             return

        # Validate allowed_gpus if provided
        valid_allowed_gpus = None
        if allowed_gpus is not None:
            if not isinstance(allowed_gpus, list):
                logger.error(f"Invalid allowed_gpus format for script {script}: Must be a list of integers. Job not added.")
                return
            valid_allowed_gpus = [gpu_id for gpu_id in allowed_gpus if 0 <= gpu_id < self.num_gpus]
            if not valid_allowed_gpus:
                 logger.error(f"No valid GPUs specified in allowed_gpus for script {script} (requested: {allowed_gpus}, available: {self.num_gpus}). Job not added.")
                 return
            if len(valid_allowed_gpus) != len(allowed_gpus):
                 logger.warning(f"Some GPUs in allowed_gpus for script {script} were invalid/out of range (requested: {allowed_gpus}, valid: {valid_allowed_gpus}).")

        # Add job tuple to the priority queue
        job_tuple = (priority, script, conda_env, args, valid_allowed_gpus)
        try:
            self.job_queue.put(job_tuple)
            logger.info(f"Job added: '{os.path.basename(script)}' (Priority: {priority}, Conda: {conda_env or 'None'}, GPUs: {valid_allowed_gpus or 'Any'})")
        except Exception as e:
            logger.error(f"Failed to add job '{os.path.basename(script)}' to queue: {e}")


    def add_jobs_from_file(self, file_path):
        """
        Add multiple jobs from a file.
        Expected format per line: priority,script_path[,conda_env[,arguments[,allowed_gpus]]]
        Example:
        0,/path/to/script1.py,myenv,"--arg1 val1",0,1
        1,/path/to/script2.py,,--flag
        0,/path/to/script3.py,baseenv,,0-3 # GPU range
        """
        logger.info(f"Attempting to add jobs from file: {file_path}")
        jobs_added = 0
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    line_num = i + 1
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue # Skip empty lines and comments

                    parts = line.split(',', maxsplit=4) # Split max 4 times for 5 potential parts

                    if len(parts) < 2:
                        logger.error(f"Invalid job format on line {line_num} in {file_path}: '{line}'. Requires at least priority,script_path.")
                        continue

                    try:
                        priority = int(parts[0].strip())
                        script = parts[1].strip()
                        conda_env = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                        args_str = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                        allowed_gpus_str = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None

                        # Parse args string into list if needed (simple split)
                        # For complex args, user should quote properly in the file
                        args = shlex.split(args_str) if args_str else None

                        # Parse allowed GPUs string (e.g., "0,1,3" or "0-2,5")
                        allowed_gpus = None
                        if allowed_gpus_str:
                            allowed_gpus = []
                            for part in allowed_gpus_str.split(','):
                                part = part.strip()
                                if '-' in part:
                                    try:
                                        start, end = map(int, part.split('-'))
                                        if start <= end:
                                             allowed_gpus.extend(range(start, end + 1))
                                        else:
                                             logger.warning(f"Invalid GPU range '{part}' on line {line_num}. Ignoring.")
                                    except ValueError:
                                        logger.warning(f"Invalid GPU range format '{part}' on line {line_num}. Ignoring.")
                                else:
                                    try:
                                        allowed_gpus.append(int(part))
                                    except ValueError:
                                        logger.warning(f"Invalid GPU ID '{part}' on line {line_num}. Ignoring.")
                            allowed_gpus = sorted(list(set(allowed_gpus))) # Remove duplicates and sort

                        # Add the job
                        self.add_job(script, conda_env, args, priority, allowed_gpus)
                        jobs_added += 1

                    except ValueError:
                        logger.error(f"Invalid priority format on line {line_num} in {file_path}: '{parts[0]}'. Must be an integer.")
                    except Exception as e:
                        logger.error(f"Error parsing job on line {line_num} in {file_path}: '{line}'. Error: {e}")

            logger.info(f"Finished processing job file '{file_path}'. Added {jobs_added} jobs.")

        except FileNotFoundError:
            logger.error(f"Job file not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read or process job file {file_path}: {e}")


    def start(self):
        """Start the GPU job scheduler worker threads."""
        if self.worker_threads:
             logger.warning("Scheduler appears to be already started.")
             return

        logger.info(f"Starting GPU Job Scheduler with {self.num_gpus} GPUs...")
        self.stop_event.clear() # Ensure stop event is not set

        # Create and start worker threads
        # Consider adjusting the number of workers (e.g., num_gpus/2 or a fixed number)
        num_workers = self.num_gpus # Or adjust as needed
        logger.info(f"Starting {num_workers} worker threads...")
        for i in range(num_workers):
            worker_thread = threading.Thread(target=self.worker, name=f"Worker-{i}", daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)

        logger.info("Scheduler started. Waiting for jobs...")


# --- Command Line Interface ---
def main():
    parser = argparse.ArgumentParser(
        description='GPU Job Scheduler: Manage and run jobs on available GPUs.',
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting in help messages
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- Start Command ---
    start_parser = subparsers.add_parser('start', help='Start the scheduler daemon process.')
    start_parser.add_argument('--gpus', type=int, default=GPUtil.getAvailable(limit=100), # Default to available GPUs
                              help='Number of GPUs to manage (default: detected available GPUs)')
    start_parser.add_argument('--jobs', type=str, help='Path to a file containing jobs to add on startup.')
    start_parser.add_argument('--screen', action='store_true', help='Enable running jobs in detached GNU Screen sessions.')
    start_parser.add_argument('--mem-threshold', type=float, default=0.8, help='GPU memory utilization threshold (0.0-1.0, default: 0.8)')
    start_parser.add_argument('--load-threshold', type=float, default=0.8, help='GPU load utilization threshold (0.0-1.0, default: 0.8)')

    # --- Add Job Command ---
    add_parser = subparsers.add_parser('add', help='Add a new job to the queue.')
    add_parser.add_argument('script', type=str, help='Path to the Python script to run.')
    add_parser.add_argument('--conda', type=str, help='Name of the conda environment to activate (optional).')
    add_parser.add_argument('--args', type=str, help='Arguments for the script (quote if containing spaces).')
    add_parser.add_argument('--priority', type=int, default=0, help='Job priority (lower number = higher priority, default: 0).')
    add_parser.add_argument('--gpus', type=str, help='Comma-separated list/range of allowed GPU IDs (e.g., "0,1" or "0-2,5", optional).')

    # --- Add Jobs File Command ---
    add_file_parser = subparsers.add_parser('add-file', help='Add multiple jobs from a formatted file.')
    add_file_parser.add_argument('file_path', type=str, help='Path to the job definition file.')

    # --- Status Command ---
    subparsers.add_parser('status', help='Show the current status of GPUs and the job queue.')

    # --- Pause Command ---
    pause_parser = subparsers.add_parser('pause', help='Pause a GPU (prevents new jobs from starting on it).')
    pause_parser.add_argument('gpu_id', type=int, help='ID of the GPU to pause.')

    # --- Resume Command ---
    resume_parser = subparsers.add_parser('resume', help='Resume a previously paused GPU.')
    resume_parser.add_argument('gpu_id', type=int, help='ID of the GPU to resume.')

    # --- List Screens Command ---
    subparsers.add_parser('screens', help='List active GNU Screen sessions potentially started by this scheduler.')

    # --- Stop Command ---
    # Note: A separate 'stop' command might be tricky if 'start' runs indefinitely.
    # Usually, Ctrl+C is used to stop the 'start' command gracefully.
    # We can add it for completeness, but it might require finding the process ID.
    # stop_parser = subparsers.add_parser('stop', help='Signal the running scheduler to stop gracefully.')


    args = parser.parse_args()

    # --- Determine Number of GPUs for non-start commands ---
    # For commands like status, pause, resume, add, we need to know the number of GPUs
    # the potentially running scheduler *might* be configured with.
    # We can try loading the state file to infer this, or use a reasonable default/detection.
    default_num_gpus = 8 # A fallback default
    try:
        if os.path.exists("gpu_scheduler_state.json"):
            with open("gpu_scheduler_state.json", 'r') as f:
                 state = json.load(f)
                 # Infer from gpu_status length if possible
                 if 'gpu_status' in state and isinstance(state['gpu_status'], list):
                      default_num_gpus = len(state['gpu_status'])
                      logger.debug(f"Inferred num_gpus={default_num_gpus} from state file for control command.")
                 else:
                      # Fallback to detection if state file is incomplete/invalid
                      default_num_gpus = len(GPUtil.getGPUs()) if GPUtil.getGPUs() else default_num_gpus
                      logger.debug(f"Could not infer num_gpus from state, detected {default_num_gpus}.")
        else:
             # Fallback to detection if no state file
             default_num_gpus = len(GPUtil.getGPUs()) if GPUtil.getGPUs() else default_num_gpus
             logger.debug(f"No state file, detected num_gpus={default_num_gpus} for control command.")
    except Exception as e:
        logger.warning(f"Error reading state file to determine GPU count for control command: {e}. Using default: {default_num_gpus}")


    # --- Execute Commands ---
    if args.command == 'start':
        # --- Start the Scheduler ---
        logger.info(f"Starting scheduler process with {args.gpus} GPUs.")
        scheduler = GPUJobScheduler(
            num_gpus=args.gpus,
            gpu_memory_threshold=args.mem_threshold,
            gpu_load_threshold=args.load_threshold
        )
        if args.screen:
            scheduler.enable_screen()
        if args.jobs:
            scheduler.add_jobs_from_file(args.jobs)

        scheduler.start()

        # Keep main thread alive, listen for Ctrl+C
        try:
            while True:
                # Can add periodic checks or tasks here if needed
                time.sleep(60) # Sleep for a minute
                # Example: Log queue size periodically
                q_size = scheduler.job_queue.qsize()
                logger.debug(f"Scheduler running... Job queue size: {q_size}")

        except KeyboardInterrupt:
            logger.info("Ctrl+C received. Initiating shutdown...")
            scheduler.stop()
            logger.info("Scheduler shutdown complete.")
        except Exception as e:
             logger.error(f"Scheduler main loop encountered an error: {e}", exc_info=True)
             logger.info("Attempting graceful shutdown due to error...")
             scheduler.stop()

    else:
        # --- Control Commands (interact with state file or potentially running instance) ---
        # For simplicity, these commands currently interact by modifying the state file
        # and printing info. A more advanced setup might use IPC (e.g., sockets)
        # to communicate with a running daemon process.
        # This implementation assumes the commands modify the state for the *next*
        # time the scheduler starts, or reflect the state based on the file.

        # Initialize scheduler instance mainly to use its methods for state manipulation/reading
        # Use the inferred/detected GPU count
        scheduler = GPUJobScheduler(num_gpus=default_num_gpus)

        if args.command == 'add':
            # Parse allowed GPUs string
            allowed_gpus_list = None
            if args.gpus:
                allowed_gpus_list = []
                for part in args.gpus.split(','):
                    part = part.strip()
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            if start <= end: allowed_gpus_list.extend(range(start, end + 1))
                        except ValueError: pass # Ignore invalid format
                    else:
                        try: allowed_gpus_list.append(int(part))
                        except ValueError: pass # Ignore invalid format
                allowed_gpus_list = sorted(list(set(allowed_gpus_list)))

            # Add job (this currently only adds to the queue if scheduler was just started,
            # otherwise it might need IPC or rely on file persistence if not running)
            # For now, let's assume it adds to the queue of this temporary instance,
            # which isn't useful unless combined with 'start'.
            # A better approach for 'add' might be to directly modify a jobs file
            # that 'start' reads, or use IPC.
            # Let's modify it to *print* the action, assuming it *would* be added.
            print(f"Attempting to add job (requires running scheduler or IPC):")
            print(f"  Script: {args.script}")
            print(f"  Conda: {args.conda or 'None'}")
            print(f"  Args: {args.args or 'None'}")
            print(f"  Priority: {args.priority}")
            print(f"  Allowed GPUs: {allowed_gpus_list or 'Any'}")
            # In a real daemon scenario, you'd send this job info via IPC.
            # scheduler.add_job(args.script, args.conda, args.args, args.priority, allowed_gpus_list)
            print("Note: 'add' command currently simulates addition. Use 'start --jobs file' or 'add-file' for persistence.")


        elif args.command == 'add-file':
             # This command can add jobs by calling the scheduler method,
             # assuming the scheduler instance loads/saves state correctly.
             scheduler.add_jobs_from_file(args.file_path)
             print(f"Jobs from file '{args.file_path}' processed. Check logs for details.")
             # Note: This adds to the *in-memory* queue of this temporary instance.
             # For persistence, the scheduler needs to be running or state saved/loaded.

        elif args.command == 'status':
            status = scheduler.get_gpu_status()
            print("\n--- GPU Status ---")
            # Header
            print(f"{'GPU ID':<8} {'State':<12} {'Memory Util':<15} {'Load Util':<15}")
            print("-" * 55)
            for gpu in status:
                print(f"{gpu['gpu_id']:<8} {gpu['state']:<12} {gpu['memory_util']:<15} {gpu['load']:<15}")

            print("\n--- Job Queue ---")
            # Get queue info (might be empty if scheduler isn't running and state wasn't loaded with jobs)
            # For PriorityQueue, accessing internal .queue is okay if protected by mutex
            with scheduler.job_queue.mutex:
                 queued_jobs = sorted(list(scheduler.job_queue.queue), key=lambda x: x[0])

            if not queued_jobs:
                print("No jobs currently in the queue (or scheduler not running).")
            else:
                print(f"{'Priority':<10} {'Script':<30} {'Conda Env':<15} {'Allowed GPUs':<15} {'Arguments'}")
                print("-" * 100)
                for i, job in enumerate(queued_jobs):
                    priority, script, conda, job_args, allowed_gpus = job
                    script_name = os.path.basename(script)
                    args_str = ""
                    if isinstance(job_args, list): args_str = " ".join(map(str, job_args))
                    elif job_args: args_str = str(job_args)

                    print(f"{priority:<10} {script_name:<30} {conda or 'None':<15} {allowed_gpus or 'Any':<15} {args_str[:50]}") # Truncate long args

        elif args.command == 'pause':
            if scheduler.pause_gpu(args.gpu_id):
                print(f"GPU {args.gpu_id} paused. State file updated.")
            else:
                print(f"Failed to pause GPU {args.gpu_id}. Check ID.")

        elif args.command == 'resume':
            if scheduler.resume_gpu(args.gpu_id):
                print(f"GPU {args.gpu_id} resumed. State file updated.")
            else:
                print(f"Failed to resume GPU {args.gpu_id}. Check ID.")

        elif args.command == 'screens':
            print("Listing active GNU Screen sessions (matching gpujob_* pattern)...")
            active_sessions = list_screens() # Use the helper function
            if not active_sessions:
                print("No active GPU job screen sessions found.")
            else:
                print("\n--- Active GPU Job Screen Sessions ---")
                print(f"{'Index':<6} {'Session Name':<50} {'Attach Command'}")
                print("-" * 80)
                for i, session in enumerate(active_sessions):
                     print(f"{i+1:<6} {session:<50} screen -r {session}")
                print("-" * 80)


if __name__ == "__main__":
    main()
