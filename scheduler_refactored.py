"""
Refactored GPU Job Scheduler with improved architecture and performance.

Key improvements:
1. Dataclasses for Job and Config
2. Separate classes for different responsibilities
3. Optimized GPU monitoring with caching
4. Fine-grained locking
5. Async state persistence
6. Better testability with dependency injection
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
import re
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set, Protocol
import hashlib
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import OrderedDict

# --- Configuration ---
@dataclass
class SchedulerConfig:
    """Centralized configuration"""
    # File paths
    default_jobs_file: str = "jobs.txt"
    default_llm_config_file: str = "llm_config.json"
    default_state_file: str = "gpu_scheduler_state.json"
    
    # Timing
    file_monitor_interval_s: int = 30
    state_check_interval_s: int = 20
    gpu_monitor_interval_s: int = 5
    
    # GPU settings
    gpu_allocation_precision: float = 0.01
    gpu_memory_threshold: float = 0.8
    gpu_load_threshold: float = 0.8
    
    # Job settings
    max_assignment_attempts: int = 5
    assignment_retry_wait_s: int = 5
    max_managed_hashes: int = 10000

# --- Data Classes ---
@dataclass
class Job:
    """Job representation"""
    priority: int
    job_id: str
    script_path: str
    conda_env: Optional[str]
    args: List[str]
    allowed_gpus: Optional[List[int]]
    job_hash: Optional[str]
    required_gpus: float
    llm_name: Optional[str]
    original_line: Optional[str]
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.job_id is None:
            self.job_id = str(uuid.uuid4())

@dataclass
class GPUStats:
    """GPU statistics"""
    gpu_id: int
    memory_util: float
    load: float
    allocation: float = 0.0
    is_paused: bool = False

@dataclass
class GPUAssignment:
    """GPU assignment result"""
    success: bool
    gpu_ids: List[int] = field(default_factory=list)
    reason: str = ""

# --- Interfaces ---
class GPUProvider(Protocol):
    def get_gpu_stats(self) -> List[GPUStats]: ...

class FileSystemProvider(Protocol):
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def file_exists(self, path: str) -> bool: ...
    def get_file_mtime(self, path: str) -> float: ...

# --- Implementations ---
class GPUtilProvider:
    """GPUtil-based GPU provider"""
    def get_gpu_stats(self) -> List[GPUStats]:
        try:
            gpus = GPUtil.getGPUs()
            return [GPUStats(i, gpu.memoryUtil, gpu.load) for i, gpu in enumerate(gpus)]
        except Exception:
            return []

class StandardFileSystemProvider:
    """Standard filesystem operations"""
    def read_file(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()
    
    def write_file(self, path: str, content: str) -> None:
        with open(path, 'w') as f:
            f.write(content)
    
    def file_exists(self, path: str) -> bool:
        return Path(path).exists()
    
    def get_file_mtime(self, path: str) -> float:
        return Path(path).stat().st_mtime

# --- Core Components ---
class GPUMonitor:
    """Centralized GPU monitoring with caching"""
    def __init__(self, config: SchedulerConfig, gpu_provider: GPUProvider):
        self.config = config
        self.gpu_provider = gpu_provider
        self.cached_stats: List[GPUStats] = []
        self.last_update = 0
        self.lock = threading.RLock()
    
    def get_current_stats(self) -> List[GPUStats]:
        with self.lock:
            current_time = time.time()
            if current_time - self.last_update > self.config.gpu_monitor_interval_s:
                self.cached_stats = self.gpu_provider.get_gpu_stats()
                self.last_update = current_time
            return self.cached_stats.copy()

class JobHashManager:
    """Job hash management with LRU cleanup"""
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.managed_hashes: OrderedDict[str, str] = OrderedDict()
        self.lock = threading.RLock()
    
    def is_managed(self, job_hash: str) -> bool:
        with self.lock:
            return job_hash in self.managed_hashes
    
    def add_hash(self, job_hash: str, job_id: str) -> bool:
        with self.lock:
            if job_hash in self.managed_hashes:
                return False
            self.managed_hashes[job_hash] = job_id
            self._cleanup_if_needed()
            return True
    
    def remove_hash(self, job_hash: str) -> bool:
        with self.lock:
            return self.managed_hashes.pop(job_hash, None) is not None
    
    def _cleanup_if_needed(self):
        while len(self.managed_hashes) > self.config.max_managed_hashes:
            self.managed_hashes.popitem(last=False)

class GPUResourceManager:
    """GPU allocation and resource management"""
    def __init__(self, config: SchedulerConfig, num_gpus: int):
        self.config = config
        self.num_gpus = num_gpus
        self.gpu_status: List[float] = [0.0] * num_gpus
        self.paused_gpus: Set[int] = set()
        self.lock = threading.RLock()
    
    def find_suitable_gpus(self, required_gpus: float, allowed_gpus: Optional[List[int]], 
                          gpu_stats: List[GPUStats]) -> GPUAssignment:
        with self.lock:
            candidates = self._get_candidates(allowed_gpus)
            
            if required_gpus <= 1.0 + self.config.gpu_allocation_precision:
                return self._find_fractional_gpu(required_gpus, candidates, gpu_stats)
            else:
                return self._find_multi_gpu(required_gpus, candidates, gpu_stats)
    
    def allocate_gpus(self, gpu_ids: List[int], required_gpus: float) -> bool:
        with self.lock:
            if required_gpus <= 1.0 + self.config.gpu_allocation_precision:
                if len(gpu_ids) == 1:
                    self.gpu_status[gpu_ids[0]] += required_gpus
                    return True
            else:
                for gpu_id in gpu_ids:
                    self.gpu_status[gpu_id] = 1.0
                return True
        return False
    
    def release_gpus(self, gpu_ids: List[int], required_gpus: float):
        with self.lock:
            release_amount = required_gpus if required_gpus <= 1.0 else 1.0
            for gpu_id in gpu_ids:
                if 0 <= gpu_id < len(self.gpu_status):
                    self.gpu_status[gpu_id] = max(0.0, self.gpu_status[gpu_id] - release_amount)
    
    def _get_candidates(self, allowed_gpus: Optional[List[int]]) -> List[int]:
        return [i for i in range(self.num_gpus) 
                if i not in self.paused_gpus and 
                (allowed_gpus is None or i in allowed_gpus)]
    
    def _find_fractional_gpu(self, required_gpus: float, candidates: List[int], 
                           gpu_stats: List[GPUStats]) -> GPUAssignment:
        for gpu_id in candidates:
            current_alloc = self.gpu_status[gpu_id]
            if current_alloc + required_gpus <= 1.0 + self.config.gpu_allocation_precision:
                if self._passes_utilization_check(gpu_id, current_alloc, gpu_stats):
                    return GPUAssignment(True, [gpu_id])
        return GPUAssignment(False, reason="No suitable GPU for fractional allocation")
    
    def _find_multi_gpu(self, required_gpus: float, candidates: List[int], 
                       gpu_stats: List[GPUStats]) -> GPUAssignment:
        num_needed = math.ceil(required_gpus)
        free_gpus = [i for i in candidates if self.gpu_status[i] < self.config.gpu_allocation_precision]
        
        if len(free_gpus) < num_needed:
            return GPUAssignment(False, reason=f"Need {num_needed} GPUs, only {len(free_gpus)} free")
        
        suitable = []
        for gpu_id in free_gpus:
            if self._passes_utilization_check(gpu_id, 0.0, gpu_stats):
                suitable.append(gpu_id)
                if len(suitable) == num_needed:
                    break
        
        if len(suitable) >= num_needed:
            return GPUAssignment(True, suitable[:num_needed])
        else:
            return GPUAssignment(False, reason=f"Only {len(suitable)} of {num_needed} GPUs passed checks")
    
    def _passes_utilization_check(self, gpu_id: int, current_alloc: float, 
                                gpu_stats: List[GPUStats]) -> bool:
        if current_alloc <= self.config.gpu_allocation_precision:
            return True
        if gpu_id >= len(gpu_stats):
            return False
        stat = gpu_stats[gpu_id]
        return (stat.memory_util < self.config.gpu_memory_threshold and 
                stat.load < self.config.gpu_load_threshold)
    
    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'gpu_status': self.gpu_status.copy(),
                'paused_gpus': list(self.paused_gpus)
            }
    
    def set_paused_gpus(self, paused: Set[int]):
        with self.lock:
            self.paused_gpus = paused.copy()

class StateManager:
    """Async state persistence"""
    def __init__(self, config: SchedulerConfig, state_file: Path, fs_provider: FileSystemProvider):
        self.config = config
        self.state_file = state_file
        self.fs_provider = fs_provider
        self.save_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.save_thread: Optional[threading.Thread] = None
    
    def start(self):
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
    
    def queue_save(self, state_data: Dict[str, Any]):
        try:
            # Replace pending saves
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                except queue.Empty:
                    break
            self.save_queue.put(state_data)
        except queue.Full:
            pass
    
    def load_state(self) -> Dict[str, Any]:
        if not self.fs_provider.file_exists(str(self.state_file)):
            return {}
        try:
            content = self.fs_provider.read_file(str(self.state_file))
            return json.loads(content)
        except Exception:
            return {}
    
    def _save_worker(self):
        while not self.stop_event.is_set():
            try:
                state_data = self.save_queue.get(timeout=2.0)
                time.sleep(0.1)  # Brief delay for batching
                
                # Get latest state
                while not self.save_queue.empty():
                    try:
                        state_data = self.save_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self._write_state(state_data)
            except queue.Empty:
                continue
    
    def _write_state(self, state_data: Dict[str, Any]):
        try:
            temp_file = self.state_file.with_suffix(f".tmp_{os.getpid()}")
            content = json.dumps(state_data, indent=4)
            self.fs_provider.write_file(str(temp_file), content)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save state: {e}")

class LLMConfigManager:
    """LLM configuration management"""
    def __init__(self, config_path: Path, fs_provider: FileSystemProvider):
        self.config_path = config_path
        self.fs_provider = fs_provider
        self.requirements: Dict[str, float] = {}
        self.default_requirement: float = 1.0
        self._load_config()
    
    def get_gpu_requirement(self, llm_name: Optional[str]) -> float:
        if llm_name and llm_name in self.requirements:
            return self.requirements[llm_name]
        return self.default_requirement
    
    def _load_config(self):
        if not self.fs_provider.file_exists(str(self.config_path)):
            return
        try:
            content = self.fs_provider.read_file(str(self.config_path))
            config = json.loads(content)
            self.default_requirement = float(config.get("default_requirement", 1.0))
            self.requirements = {k: float(v) for k, v in config.get("models", {}).items()}
        except Exception:
            pass

class JobQueue:
    """Thread-safe priority queue"""
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self.job_counter = 0
        self.lock = threading.RLock()
    
    def put(self, job: Job):
        with self.lock:
            self.queue.put((job.priority, str(self.job_counter), job))
            self.job_counter += 1
    
    def get(self, timeout: float = 1.0) -> Optional[Job]:
        try:
            _, _, job = self.queue.get(timeout=timeout)
            return job
        except queue.Empty:
            return None
    
    def task_done(self):
        self.queue.task_done()

# --- Main Scheduler ---
class GPUJobScheduler:
    """Refactored main scheduler"""
    
    def __init__(self, 
                 config: Optional[SchedulerConfig] = None,
                 num_gpus: int = 8,
                 jobs_file_path: Optional[str] = None,
                 gpu_provider: Optional[GPUProvider] = None,
                 fs_provider: Optional[FileSystemProvider] = None):
        
        self.config = config or SchedulerConfig()
        self.num_gpus = num_gpus
        self.stop_event = threading.Event()
        
        # Dependencies
        self.gpu_provider = gpu_provider or GPUtilProvider()
        self.fs_provider = fs_provider or StandardFileSystemProvider()
        
        # Components
        self.gpu_monitor = GPUMonitor(self.config, self.gpu_provider)
        self.hash_manager = JobHashManager(self.config)
        self.resource_manager = GPUResourceManager(self.config, num_gpus)
        self.job_queue = JobQueue(self.config)
        self.llm_config = LLMConfigManager(Path(self.config.default_llm_config_file), self.fs_provider)
        self.state_manager = StateManager(self.config, Path(self.config.default_state_file), self.fs_provider)
        
        # Threading
        self.worker_threads: List[threading.Thread] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.jobs_file_path = jobs_file_path
        self._load_initial_state()
    
    def _load_initial_state(self):
        state = self.state_manager.load_state()
        if 'gpu_status' in state and len(state['gpu_status']) == self.num_gpus:
            self.resource_manager.gpu_status = [max(0.0, min(1.0, float(s))) for s in state['gpu_status']]
        if 'paused_gpus' in state:
            paused = set(gpu_id for gpu_id in state['paused_gpus'] if 0 <= gpu_id < self.num_gpus)
            self.resource_manager.set_paused_gpus(paused)
    
    def start(self):
        logging.getLogger(__name__).info("Starting refactored GPU scheduler")
        self.state_manager.start()
        
        # Start workers
        for i in range(self.num_gpus):
            worker = threading.Thread(target=self._worker, name=f"Worker-{i}", daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        # Start monitor
        self.monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self.monitor_thread.start()
        
        # Initial file load
        if self.jobs_file_path:
            self._process_job_file(self.jobs_file_path, True)
    
    def stop(self):
        logging.getLogger(__name__).info("Stopping scheduler")
        self.stop_event.set()
        
        # Stop workers
        for _ in self.worker_threads:
            self.job_queue.put(self._sentinel_job())
        
        for worker in self.worker_threads:
            worker.join(timeout=10.0)
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.state_manager.stop()
        self._save_state()
    
    def _worker(self):
        """Simplified worker loop"""
        while not self.stop_event.is_set():
            job = self.job_queue.get()
            if not job or job.job_id == "SENTINEL":
                break
            
            if self.stop_event.is_set():
                self.job_queue.put(job)
                break
            
            assigned = self._assign_job(job)
            if assigned:
                self._launch_job(job, assigned.gpu_ids)
            else:
                self._requeue_job(job)
            
            self.job_queue.task_done()
    
    def _assign_job(self, job: Job) -> Optional[GPUAssignment]:
        """Assign job to GPUs"""
        for attempt in range(self.config.max_assignment_attempts):
            gpu_stats = self.gpu_monitor.get_current_stats()
            assignment = self.resource_manager.find_suitable_gpus(
                job.required_gpus, job.allowed_gpus, gpu_stats)
            
            if assignment.success:
                if self.resource_manager.allocate_gpus(assignment.gpu_ids, job.required_gpus):
                    self._save_state()
                    return assignment
            
            if attempt < self.config.max_assignment_attempts - 1:
                wait_time = min(self.config.assignment_retry_wait_s * (attempt + 1), 30)
                if self.stop_event.wait(timeout=wait_time):
                    break
        
        return None
    
    def _launch_job(self, job: Job, gpu_ids: List[int]):
        """Launch job execution"""
        def release_callback(success: bool):
            self.resource_manager.release_gpus(gpu_ids, job.required_gpus)
            self._save_state()
            if job.job_hash and not success:
                self.hash_manager.remove_hash(job.job_hash)
        
        # Simple execution - in real version this would handle screen/direct modes
        job_thread = threading.Thread(
            target=lambda: self._execute_job(job, gpu_ids, release_callback),
            daemon=True
        )
        job_thread.start()
    
    def _execute_job(self, job: Job, gpu_ids: List[int], release_callback):
        """Execute job (simplified)"""
        try:
            # Simulate job execution
            time.sleep(1)  # Placeholder
            release_callback(True)  # Assume success
        except Exception:
            release_callback(False)
    
    def _requeue_job(self, job: Job):
        self.job_queue.put(job)
    
    def _monitor_files(self):
        """Monitor job files"""
        last_mtime = 0
        
        while not self.stop_event.is_set():
            try:
                if self.jobs_file_path:
                    current_mtime = self.fs_provider.get_file_mtime(self.jobs_file_path)
                    if current_mtime > last_mtime:
                        self._process_job_file(self.jobs_file_path)
                        last_mtime = current_mtime
                
                self._check_external_state()
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Monitor error: {e}")
            
            time.sleep(min(self.config.file_monitor_interval_s, self.config.state_check_interval_s))
    
    def _process_job_file(self, file_path: str, initial: bool = False):
        """Process jobs from file"""
        try:
            content = self.fs_provider.read_file(file_path)
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                job = self._parse_job_line(line)
                if job and self._should_add_job(job):
                    self.job_queue.put(job)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error processing {file_path}: {e}")
    
    def _parse_job_line(self, line: str) -> Optional[Job]:
        """Parse job from line"""
        try:
            parts = [p.strip() for p in line.split(',', 4)]
            if len(parts) < 2:
                return None
            
            priority = int(parts[0])
            script_path = parts[1]
            conda_env = parts[2] if len(parts) > 2 and parts[2] else None
            args_str = parts[3] if len(parts) > 3 and parts[3] else None
            allowed_gpus_str = parts[4] if len(parts) > 4 and parts[4] else None
            
            args = shlex.split(args_str) if args_str else []
            llm_name = self._extract_arg_value(args, '--llm')
            required_gpus = self.llm_config.get_gpu_requirement(llm_name)
            allowed_gpus = self._parse_gpu_list(allowed_gpus_str) if allowed_gpus_str else None
            
            job_hash = self._calculate_job_hash(priority, script_path, conda_env, args, allowed_gpus, required_gpus)
            
            return Job(
                priority=priority,
                job_id=str(uuid.uuid4()),
                script_path=script_path,
                conda_env=conda_env,
                args=args,
                allowed_gpus=allowed_gpus,
                job_hash=job_hash,
                required_gpus=required_gpus,
                llm_name=llm_name,
                original_line=line
            )
        except Exception:
            return None
    
    def _should_add_job(self, job: Job) -> bool:
        if job.job_hash:
            return self.hash_manager.add_hash(job.job_hash, job.job_id)
        return True
    
    def _save_state(self):
        state_data = self.resource_manager.get_status()
        self.state_manager.queue_save(state_data)
    
    def _check_external_state(self):
        """Check for external state changes"""
        state = self.state_manager.load_state()
        if 'paused_gpus' in state:
            new_paused = set(state['paused_gpus'])
            if new_paused != self.resource_manager.paused_gpus:
                self.resource_manager.set_paused_gpus(new_paused)
    
    # Utility methods
    def _extract_arg_value(self, args: List[str], key: str) -> Optional[str]:
        try:
            index = args.index(key)
            return args[index + 1] if index + 1 < len(args) else None
        except ValueError:
            return None
    
    def _parse_gpu_list(self, gpu_str: str) -> List[int]:
        gpu_ids = []
        for part in gpu_str.replace(" ", "").split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                gpu_ids.extend(range(start, end + 1))
            else:
                gpu_ids.append(int(part))
        return sorted(list(set(gpu_ids)))
    
    def _calculate_job_hash(self, priority: int, script: str, conda_env: Optional[str], 
                          args: List[str], allowed_gpus: Optional[List[int]], 
                          required_gpus: float) -> str:
        hasher = hashlib.md5()
        hasher.update(str(priority).encode())
        hasher.update(str(script).encode())
        hasher.update(str(conda_env or '').encode())
        hasher.update(str(sorted(args)).encode())
        hasher.update(str(sorted(allowed_gpus) if allowed_gpus else []).encode())
        hasher.update(str(required_gpus).encode())
        return hasher.hexdigest()
    
    def _sentinel_job(self) -> Job:
        return Job(float('inf'), "SENTINEL", "", None, [], None, None, 0.0, None, None)
    
    # Public API
    def add_job(self, script: str, conda_env: Optional[str] = None, 
               args: Optional[List[str]] = None, priority: int = 0,
               allowed_gpus: Optional[List[int]] = None) -> str:
        args = args or []
        llm_name = self._extract_arg_value(args, '--llm')
        required_gpus = self.llm_config.get_gpu_requirement(llm_name)
        
        job = Job(priority, str(uuid.uuid4()), script, conda_env, args, 
                 allowed_gpus, None, required_gpus, llm_name, None)
        self.job_queue.put(job)
        return job.job_id
    
    def get_gpu_status(self) -> List[Dict[str, Any]]:
        gpu_stats = self.gpu_monitor.get_current_stats()
        resource_status = self.resource_manager.get_status()
        
        status_list = []
        for i in range(self.num_gpus):
            allocation = resource_status['gpu_status'][i]
            is_paused = i in resource_status['paused_gpus']
            
            state = "PAUSED" if is_paused else \
                   f"BUSY ({allocation*100:.0f}%)" if allocation > self.config.gpu_allocation_precision else \
                   "AVAILABLE"
            
            memory_util = load = "N/A"
            if i < len(gpu_stats):
                memory_util = f"{gpu_stats[i].memory_util * 100:.1f}%"
                load = f"{gpu_stats[i].load * 100:.1f}%"
            
            status_list.append({
                "gpu_id": i, "state": state, "allocation": f"{allocation:.2f}",
                "memory_util": memory_util, "load": load
            })
        
        return status_list

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[logging.FileHandler("gpu_scheduler_refactored.log"), logging.StreamHandler()]
)

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Refactored GPU Job Scheduler')
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--jobs-file', type=str)
    args = parser.parse_args()
    
    scheduler = GPUJobScheduler(num_gpus=args.gpus, jobs_file_path=args.jobs_file)
    scheduler.start()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        scheduler.stop()

if __name__ == "__main__":
    main() 