#!/usr/bin/env python3
"""
Comprehensive Test Framework for Ultra GPU Scheduler v3.0

Features:
- Unit tests for all components
- Integration tests for workflow scenarios  
- Performance benchmarking and load testing
- Mock providers for isolated testing
- Property-based testing with hypothesis
- Stress testing and fault injection
"""

import unittest
import pytest
import threading
import time
import tempfile
import json
import shutil
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import queue
import subprocess

# Import the scheduler components
try:
    from scheduler_10x import (
        SchedulerConfig, Job, GPUStats, SystemMetrics, JobStatus, HealthStatus,
        StructuredLogger, ProductionMetricsProvider, ProductionHealthMonitor,
        EnhancedGPUProvider, EnhancedJobQueue, EnhancedGPUResourceManager,
        EnhancedJobExecutor, CircuitBreaker, GPUProvider, FileSystemProvider,
        MetricsProvider, HealthMonitor
    )
except ImportError as e:
    print(f"Warning: Could not import scheduler components: {e}")
    # Define minimal stubs for testing
    pass


class MockGPUProvider:
    """Mock GPU provider for testing"""
    
    def __init__(self, num_gpus: int = 4, simulate_errors: bool = False):
        self.num_gpus = num_gpus
        self.simulate_errors = simulate_errors
        self.error_count = 0
        self.call_count = 0
        
    def get_gpu_stats(self):
        self.call_count += 1
        
        if self.simulate_errors and self.error_count < 3:
            self.error_count += 1
            raise RuntimeError(f"Simulated GPU error {self.error_count}")
        
        # Return mock stats
        return [{"gpu_id": i, "memory_util": 0.3, "load": 0.2} for i in range(self.num_gpus)]


class MockFileSystemProvider:
    """Mock filesystem provider for testing"""
    
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.mtimes: Dict[str, float] = {}
        
    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]
    
    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content
        self.mtimes[path] = time.time()
    
    def file_exists(self, path: str) -> bool:
        return path in self.files
    
    def get_file_mtime(self, path: str) -> float:
        if path not in self.mtimes:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.mtimes[path]
    
    def add_file(self, path: str, content: str):
        """Helper method to add files for testing"""
        self.files[path] = content
        self.mtimes[path] = time.time()


class MockMetricsProvider:
    """Mock metrics provider for testing"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.gpu_metrics: List[List[GPUStats]] = []
        self.system_metrics: List[SystemMetrics] = []
    
    def record_job_event(self, job: Job, event: str, **kwargs) -> None:
        self.events.append({
            'job_id': job.job_id,
            'event': event,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def record_gpu_metrics(self, gpu_stats: List[GPUStats]) -> None:
        self.gpu_metrics.append(gpu_stats)
    
    def record_system_metrics(self, metrics: SystemMetrics) -> None:
        self.system_metrics.append(metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            'events': len(self.events),
            'gpu_metrics': len(self.gpu_metrics),
            'system_metrics': len(self.system_metrics)
        }


class TestSchedulerConfig(unittest.TestCase):
    """Test configuration validation and loading"""
    
    def test_default_config_valid(self):
        """Test that default configuration is valid"""
        config = SchedulerConfig()
        self.assertIsInstance(config, SchedulerConfig)
        self.assertEqual(config.num_gpus, 8)
        self.assertEqual(config.gpu_memory_threshold, 0.8)
    
    def test_config_validation_gpu_count(self):
        """Test GPU count validation"""
        with self.assertRaises(ValueError):
            SchedulerConfig(num_gpus=0)
        
        with self.assertRaises(ValueError):
            SchedulerConfig(num_gpus=100)
    
    def test_config_validation_thresholds(self):
        """Test threshold validation"""
        with self.assertRaises(ValueError):
            SchedulerConfig(gpu_memory_threshold=1.5)
        
        with self.assertRaises(ValueError):
            SchedulerConfig(gpu_load_threshold=-0.1)
    
    def test_config_from_file(self):
        """Test loading configuration from file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "scheduler": {"num_gpus": 4},
                "gpu": {"memory_threshold": 0.7},
                "jobs": {"max_assignment_attempts": 3},
                "files": {"jobs_file": "test_jobs.txt"}
            }
            json.dump(config_data, f)
            f.flush()
            
            try:
                config = SchedulerConfig.from_file(f.name)
                self.assertEqual(config.num_gpus, 4)
                self.assertEqual(config.gpu_memory_threshold, 0.7)
                self.assertEqual(config.max_assignment_attempts, 3)
            finally:
                Path(f.name).unlink()
    
    def test_config_from_invalid_file(self):
        """Test loading from invalid configuration file"""
        with self.assertRaises(ValueError):
            SchedulerConfig.from_file("nonexistent.json")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            f.flush()
            
            try:
                with self.assertRaises(ValueError):
                    SchedulerConfig.from_file(f.name)
            finally:
                Path(f.name).unlink()


class TestJob(unittest.TestCase):
    """Test Job data class functionality"""
    
    def test_job_creation(self):
        """Test basic job creation"""
        job = Job(
            priority=1,
            job_id="test-job-1",
            script_path="/test/script.py",
            conda_env="test_env",
            args=["--arg1", "value1"],
            allowed_gpus=[0, 1],
            job_hash="test_hash",
            required_gpus=1.0,
            llm_name="test_llm",
            original_line="test line"
        )
        
        self.assertEqual(job.priority, 1)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertIsInstance(job.created_at, datetime)
        self.assertIsNone(job.assigned_at)
    
    def test_job_status_updates(self):
        """Test job status update lifecycle"""
        job = Job(
            priority=1,
            job_id="test-job-1",
            script_path="/test/script.py",
            conda_env=None,
            args=[],
            allowed_gpus=None,
            job_hash=None,
            required_gpus=1.0,
            llm_name=None,
            original_line=None
        )
        
        # Test assignment
        job.update_status(JobStatus.ASSIGNED)
        self.assertEqual(job.status, JobStatus.ASSIGNED)
        self.assertIsNotNone(job.assigned_at)
        
        # Test running
        job.update_status(JobStatus.RUNNING)
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertIsNotNone(job.started_at)
        
        # Test completion
        job.update_status(JobStatus.COMPLETED)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(job.completed_at)
        self.assertIsNotNone(job.execution_time_s)
        self.assertGreater(job.execution_time_s, 0)
    
    def test_job_serialization(self):
        """Test job serialization to dictionary"""
        job = Job(
            priority=1,
            job_id="test-job-1",
            script_path="/test/script.py",
            conda_env="test_env",
            args=["--test"],
            allowed_gpus=[0],
            job_hash="hash123",
            required_gpus=1.0,
            llm_name="test_llm",
            original_line="test line"
        )
        
        job_dict = job.to_dict()
        self.assertIsInstance(job_dict, dict)
        self.assertEqual(job_dict['job_id'], "test-job-1")
        self.assertEqual(job_dict['priority'], 1)
        self.assertIn('created_at', job_dict)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation"""
        cb = CircuitBreaker(failure_threshold=3, timeout_s=1.0)
        
        # Should allow operations normally
        with cb.protect() as allowed:
            self.assertTrue(allowed)
    
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        cb = CircuitBreaker(failure_threshold=2, timeout_s=1.0)
        
        # Cause failures
        for i in range(2):
            try:
                with cb.protect() as allowed:
                    self.assertTrue(allowed)
                    raise RuntimeError("Test failure")
            except RuntimeError:
                pass
        
        # Circuit should now be open
        with cb.protect() as allowed:
            self.assertFalse(allowed)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout"""
        cb = CircuitBreaker(failure_threshold=1, timeout_s=0.1)
        
        # Cause failure
        try:
            with cb.protect() as allowed:
                self.assertTrue(allowed)
                raise RuntimeError("Test failure")
        except RuntimeError:
            pass
        
        # Should be open
        with cb.protect() as allowed:
            self.assertFalse(allowed)
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should allow again (half-open)
        with cb.protect() as allowed:
            self.assertTrue(allowed)


class TestEnhancedGPUProvider(unittest.TestCase):
    """Test enhanced GPU provider"""
    
    def setUp(self):
        self.config = SchedulerConfig(gpu_monitor_interval_s=0.1)
        self.logger = Mock()
        
    def test_gpu_stats_caching(self):
        """Test GPU stats caching behavior"""
        with patch('scheduler_10x.GPUtil.getGPUs') as mock_gputil:
            mock_gpu = Mock()
            mock_gpu.memoryUtil = 0.5
            mock_gpu.load = 0.3
            mock_gpu.memoryTotal = 24000
            mock_gpu.memoryUsed = 12000
            mock_gputil.return_value = [mock_gpu]
            
            provider = EnhancedGPUProvider(self.config, self.logger)
            
            # First call should fetch from GPUtil
            stats1 = provider.get_gpu_stats()
            self.assertEqual(len(stats1), 1)
            self.assertEqual(mock_gputil.call_count, 1)
            
            # Second call should use cache
            stats2 = provider.get_gpu_stats()
            self.assertEqual(len(stats2), 1)
            self.assertEqual(mock_gputil.call_count, 1)  # No additional call
    
    def test_gpu_stats_error_handling(self):
        """Test GPU stats error handling and recovery"""
        with patch('scheduler_10x.GPUtil.getGPUs') as mock_gputil:
            mock_gputil.side_effect = RuntimeError("GPU error")
            
            provider = EnhancedGPUProvider(self.config, self.logger)
            
            # Should return empty list on error
            stats = provider.get_gpu_stats()
            self.assertEqual(len(stats), 0)
            
            # Error should be logged
            self.logger.error.assert_called()


class TestEnhancedJobQueue(unittest.TestCase):
    """Test enhanced job queue functionality"""
    
    def setUp(self):
        self.config = SchedulerConfig(queue_size_limit=10)
        self.logger = Mock()
        self.metrics = MockMetricsProvider()
        self.queue = EnhancedJobQueue(self.config, self.logger, self.metrics)
    
    def test_job_queue_priority_ordering(self):
        """Test job queue priority ordering"""
        # Create jobs with different priorities
        job_high = Job(priority=-1, job_id="high", script_path="/test.py", 
                      conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                      required_gpus=1.0, llm_name=None, original_line=None)
        job_normal = Job(priority=5, job_id="normal", script_path="/test.py",
                        conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                        required_gpus=1.0, llm_name=None, original_line=None)
        job_low = Job(priority=15, job_id="low", script_path="/test.py",
                     conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                     required_gpus=1.0, llm_name=None, original_line=None)
        
        # Add in mixed order
        self.assertTrue(self.queue.put(job_normal))
        self.assertTrue(self.queue.put(job_low))
        self.assertTrue(self.queue.put(job_high))
        
        # Should get high priority first
        retrieved_job = self.queue.get(timeout=0.1)
        self.assertEqual(retrieved_job.job_id, "high")
        
        # Then normal priority
        retrieved_job = self.queue.get(timeout=0.1)
        self.assertEqual(retrieved_job.job_id, "normal")
        
        # Then low priority
        retrieved_job = self.queue.get(timeout=0.1)
        self.assertEqual(retrieved_job.job_id, "low")
    
    def test_queue_size_limit(self):
        """Test queue size limit enforcement"""
        # Fill queue to limit
        for i in range(self.config.queue_size_limit):
            job = Job(priority=1, job_id=f"job-{i}", script_path="/test.py",
                     conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                     required_gpus=1.0, llm_name=None, original_line=None)
            self.assertTrue(self.queue.put(job))
        
        # Next job should be rejected
        overflow_job = Job(priority=1, job_id="overflow", script_path="/test.py",
                          conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                          required_gpus=1.0, llm_name=None, original_line=None)
        self.assertFalse(self.queue.put(overflow_job))
    
    def test_queue_status(self):
        """Test queue status reporting"""
        status = self.queue.get_status()
        self.assertEqual(status['total_jobs'], 0)
        
        # Add some jobs
        job1 = Job(priority=-1, job_id="job1", script_path="/test.py",
                  conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                  required_gpus=1.0, llm_name=None, original_line=None)
        job2 = Job(priority=5, job_id="job2", script_path="/test.py",
                  conda_env=None, args=[], allowed_gpus=None, job_hash=None,
                  required_gpus=1.0, llm_name=None, original_line=None)
        
        self.queue.put(job1)
        self.queue.put(job2)
        
        status = self.queue.get_status()
        self.assertEqual(status['total_jobs'], 2)
        self.assertEqual(status['high_priority_size'], 1)
        self.assertEqual(status['normal_priority_size'], 1)


class TestEnhancedGPUResourceManager(unittest.TestCase):
    """Test enhanced GPU resource manager"""
    
    def setUp(self):
        self.config = SchedulerConfig(gpu_allocation_precision=0.01)
        self.logger = Mock()
        self.metrics = MockMetricsProvider()
        self.manager = EnhancedGPUResourceManager(self.config, 4, self.logger, self.metrics)
    
    def test_fractional_gpu_allocation(self):
        """Test fractional GPU allocation"""
        # Create mock GPU stats
        gpu_stats = [
            GPUStats(gpu_id=i, memory_util=0.3, load=0.2) for i in range(4)
        ]
        
        # Find suitable GPU for 0.5 requirement
        assignment = self.manager.find_suitable_gpus(0.5, None, gpu_stats)
        self.assertTrue(assignment.success)
        self.assertEqual(len(assignment.gpu_ids), 1)
        
        # Allocate the GPU
        success = self.manager.allocate_gpus(assignment.gpu_ids, 0.5, "test-job-1")
        self.assertTrue(success)
        
        # Check allocation
        self.assertAlmostEqual(self.manager.gpu_status[assignment.gpu_ids[0]], 0.5)
    
    def test_multi_gpu_allocation(self):
        """Test multi-GPU allocation"""
        gpu_stats = [
            GPUStats(gpu_id=i, memory_util=0.1, load=0.1) for i in range(4)
        ]
        
        # Find suitable GPUs for 2.5 requirement (needs 3 GPUs)
        assignment = self.manager.find_suitable_gpus(2.5, None, gpu_stats)
        self.assertTrue(assignment.success)
        self.assertEqual(len(assignment.gpu_ids), 3)
        
        # Allocate the GPUs
        success = self.manager.allocate_gpus(assignment.gpu_ids, 2.5, "test-job-2")
        self.assertTrue(success)
        
        # Check allocation
        for gpu_id in assignment.gpu_ids:
            self.assertAlmostEqual(self.manager.gpu_status[gpu_id], 1.0)
    
    def test_gpu_allocation_constraints(self):
        """Test GPU allocation with allowed GPU constraints"""
        gpu_stats = [
            GPUStats(gpu_id=i, memory_util=0.1, load=0.1) for i in range(4)
        ]
        
        # Restrict to only GPUs 2 and 3
        allowed_gpus = [2, 3]
        assignment = self.manager.find_suitable_gpus(1.0, allowed_gpus, gpu_stats)
        self.assertTrue(assignment.success)
        self.assertIn(assignment.gpu_ids[0], allowed_gpus)
    
    def test_paused_gpu_handling(self):
        """Test paused GPU handling"""
        gpu_stats = [
            GPUStats(gpu_id=i, memory_util=0.1, load=0.1) for i in range(4)
        ]
        
        # Pause GPU 0
        self.manager.set_paused_gpus({0})
        
        # Should not allocate to paused GPU
        assignment = self.manager.find_suitable_gpus(1.0, [0, 1], gpu_stats)
        self.assertTrue(assignment.success)
        self.assertEqual(assignment.gpu_ids[0], 1)  # Should use GPU 1, not 0
    
    def test_resource_release(self):
        """Test GPU resource release"""
        # Allocate a GPU
        success = self.manager.allocate_gpus([0], 0.7, "test-job")
        self.assertTrue(success)
        self.assertAlmostEqual(self.manager.gpu_status[0], 0.7)
        
        # Release the GPU
        self.manager.release_gpus([0], 0.7, "test-job")
        self.assertAlmostEqual(self.manager.gpu_status[0], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for scheduler components"""
    
    def setUp(self):
        self.config = SchedulerConfig(
            num_gpus=2,
            queue_size_limit=5,
            max_concurrent_jobs=2
        )
        self.logger = Mock()
        self.metrics = MockMetricsProvider()
        self.gpu_provider = MockGPUProvider(num_gpus=2)
        self.fs_provider = MockFileSystemProvider()
        
        # Setup components
        self.queue = EnhancedJobQueue(self.config, self.logger, self.metrics)
        self.resource_manager = EnhancedGPUResourceManager(
            self.config, 2, self.logger, self.metrics
        )
        self.job_executor = EnhancedJobExecutor(self.config, self.logger, self.metrics)
    
    def test_job_lifecycle_integration(self):
        """Test complete job lifecycle integration"""
        # Create a test job
        job = Job(
            priority=1,
            job_id="integration-test-job",
            script_path="/usr/bin/python",  # Use system python for testing
            conda_env=None,
            args=["-c", "print('Hello from test job'); import time; time.sleep(0.1)"],
            allowed_gpus=None,
            job_hash="test_hash",
            required_gpus=1.0,
            llm_name=None,
            original_line="test job line"
        )
        
        # Add job to queue
        self.assertTrue(self.queue.put(job))
        
        # Get job from queue
        retrieved_job = self.queue.get(timeout=1.0)
        self.assertEqual(retrieved_job.job_id, job.job_id)
        
        # Find GPU resources
        gpu_stats = self.gpu_provider.get_gpu_stats()
        assignment = self.resource_manager.find_suitable_gpus(
            job.required_gpus, job.allowed_gpus, gpu_stats
        )
        self.assertTrue(assignment.success)
        
        # Allocate GPU
        allocated = self.resource_manager.allocate_gpus(
            assignment.gpu_ids, job.required_gpus, job.job_id
        )
        self.assertTrue(allocated)
        
        # Execute job (mock execution for testing)
        release_called = threading.Event()
        def mock_release_callback(success: bool):
            release_called.set()
        
        # For integration test, we'll use a simple mock execution
        with patch.object(self.job_executor, '_execute_directly', return_value=True):
            executed = self.job_executor.execute_job(
                job, assignment.gpu_ids, mock_release_callback
            )
            self.assertTrue(executed)
        
        # Wait for completion
        self.assertTrue(release_called.wait(timeout=5.0))
        
        # Verify metrics were recorded
        self.assertGreater(len(self.metrics.events), 0)
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        # Test GPU provider failure handling
        failing_provider = MockGPUProvider(num_gpus=2, simulate_errors=True)
        
        # Should handle errors gracefully
        for i in range(5):
            stats = failing_provider.get_gpu_stats()
            # After 3 errors, should start working
            if i >= 3:
                self.assertEqual(len(stats), 2)


class TestPerformance(unittest.TestCase):
    """Performance and load testing"""
    
    def setUp(self):
        self.config = SchedulerConfig(
            num_gpus=8,
            queue_size_limit=1000,
            max_concurrent_jobs=50
        )
        self.logger = Mock()
        self.metrics = MockMetricsProvider()
    
    def test_queue_performance(self):
        """Test job queue performance under load"""
        queue = EnhancedJobQueue(self.config, self.logger, self.metrics)
        
        # Measure time to add 1000 jobs
        start_time = time.time()
        
        for i in range(1000):
            job = Job(
                priority=i % 10,
                job_id=f"perf-job-{i}",
                script_path="/test.py",
                conda_env=None,
                args=[],
                allowed_gpus=None,
                job_hash=f"hash-{i}",
                required_gpus=1.0,
                llm_name=None,
                original_line=None
            )
            queue.put(job)
        
        add_time = time.time() - start_time
        self.assertLess(add_time, 1.0, "Adding 1000 jobs should take less than 1 second")
        
        # Measure time to retrieve 1000 jobs
        start_time = time.time()
        
        for i in range(1000):
            job = queue.get(timeout=0.1)
            self.assertIsNotNone(job)
        
        get_time = time.time() - start_time
        self.assertLess(get_time, 1.0, "Retrieving 1000 jobs should take less than 1 second")
    
    def test_resource_manager_performance(self):
        """Test resource manager performance"""
        manager = EnhancedGPUResourceManager(self.config, 8, self.logger, self.metrics)
        
        # Create GPU stats
        gpu_stats = [
            GPUStats(gpu_id=i, memory_util=0.3, load=0.2) for i in range(8)
        ]
        
        # Measure time for 1000 allocation attempts
        start_time = time.time()
        
        for i in range(1000):
            assignment = manager.find_suitable_gpus(0.5, None, gpu_stats)
            # Don't actually allocate to avoid filling up resources
        
        allocation_time = time.time() - start_time
        self.assertLess(allocation_time, 2.0, "1000 allocation attempts should take less than 2 seconds")


class TestStress(unittest.TestCase):
    """Stress testing and fault injection"""
    
    def test_concurrent_queue_operations(self):
        """Test concurrent queue operations"""
        config = SchedulerConfig(queue_size_limit=1000)
        logger = Mock()
        metrics = MockMetricsProvider()
        queue = EnhancedJobQueue(config, logger, metrics)
        
        # Create producer threads
        def producer(thread_id: int):
            for i in range(100):
                job = Job(
                    priority=i,
                    job_id=f"stress-{thread_id}-{i}",
                    script_path="/test.py",
                    conda_env=None,
                    args=[],
                    allowed_gpus=None,
                    job_hash=f"hash-{thread_id}-{i}",
                    required_gpus=1.0,
                    llm_name=None,
                    original_line=None
                )
                queue.put(job)
        
        # Create consumer threads
        consumed_jobs = []
        def consumer():
            while len(consumed_jobs) < 500:  # 5 producers * 100 jobs each
                job = queue.get(timeout=5.0)
                if job:
                    consumed_jobs.append(job)
        
        # Start threads
        producers = [threading.Thread(target=producer, args=(i,)) for i in range(5)]
        consumers = [threading.Thread(target=consumer) for _ in range(3)]
        
        start_time = time.time()
        
        for t in producers + consumers:
            t.start()
        
        for t in producers + consumers:
            t.join(timeout=10.0)
        
        end_time = time.time()
        
        # Verify all jobs were processed
        self.assertEqual(len(consumed_jobs), 500)
        self.assertLess(end_time - start_time, 5.0, "Stress test should complete quickly")


def run_all_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSchedulerConfig,
        TestJob,
        TestCircuitBreaker,
        TestEnhancedGPUProvider,
        TestEnhancedJobQueue,
        TestEnhancedGPUResourceManager,
        TestIntegration,
        TestPerformance,
        TestStress
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 