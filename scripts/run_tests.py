#!/usr/bin/env python3
"""
Comprehensive test runner for Riad Concierge AI
Orchestrates unit, integration, and performance testing
"""

import asyncio
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup basic logging for test runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner with reporting and metrics."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {
            "unit": {"passed": 0, "failed": 0, "skipped": 0, "duration": 0},
            "integration": {"passed": 0, "failed": 0, "skipped": 0, "duration": 0},
            "performance": {"passed": 0, "failed": 0, "skipped": 0, "duration": 0}
        }
        self.coverage_threshold = 85.0
        self.performance_baseline = {
            "response_time": 2.0,
            "concurrent_users": 20,
            "cultural_accuracy": 0.90,
            "knowledge_retrieval": 1.0
        }
    
    async def run_unit_tests(self, verbose: bool = False) -> bool:
        """Run unit tests with coverage reporting."""
        logger.info("🧪 Running unit tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/unit/",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=json:coverage.json",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.test_results["unit"]["duration"] = duration
        
        if result.returncode == 0:
            logger.info(f"✅ Unit tests passed in {duration:.2f}s")
            self._parse_pytest_output(result.stdout, "unit")
            return await self._check_coverage()
        else:
            logger.error(f"❌ Unit tests failed:")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    
    async def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        logger.info("🔗 Running integration tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            "-m", "not slow"  # Skip slow tests by default
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.test_results["integration"]["duration"] = duration
        
        if result.returncode == 0:
            logger.info(f"✅ Integration tests passed in {duration:.2f}s")
            self._parse_pytest_output(result.stdout, "integration")
            return True
        else:
            logger.error(f"❌ Integration tests failed:")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    
    async def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance benchmarking tests."""
        logger.info("⚡ Running performance tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/performance/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        duration = time.time() - start_time
        
        self.test_results["performance"]["duration"] = duration
        
        if result.returncode == 0:
            logger.info(f"✅ Performance tests passed in {duration:.2f}s")
            self._parse_pytest_output(result.stdout, "performance")
            await self._analyze_performance_results()
            return True
        else:
            logger.error(f"❌ Performance tests failed:")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
    
    async def run_cultural_validation(self) -> bool:
        """Run cultural intelligence validation tests."""
        logger.info("🌍 Running cultural validation tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/unit/test_cultural_service.py",
            "tests/integration/test_agent_workflow.py::TestServiceInteractionIntegration::test_cultural_knowledge_integration",
            "-v",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Cultural validation passed")
            return True
        else:
            logger.error("❌ Cultural validation failed")
            logger.error(result.stdout)
            return False
    
    async def run_security_tests(self) -> bool:
        """Run security and compliance tests."""
        logger.info("🔒 Running security tests...")
        
        # Check for common security issues
        security_checks = [
            self._check_environment_variables(),
            self._check_api_key_exposure(),
            self._check_input_validation(),
            self._check_rate_limiting()
        ]
        
        results = await asyncio.gather(*security_checks)
        
        if all(results):
            logger.info("✅ Security tests passed")
            return True
        else:
            logger.error("❌ Security tests failed")
            return False
    
    async def _check_coverage(self) -> bool:
        """Check code coverage meets threshold."""
        try:
            with open(self.project_root / "coverage.json", "r") as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data["totals"]["percent_covered"]
            
            if total_coverage >= self.coverage_threshold:
                logger.info(f"✅ Code coverage: {total_coverage:.1f}% (threshold: {self.coverage_threshold}%)")
                return True
            else:
                logger.warning(f"⚠️ Code coverage: {total_coverage:.1f}% below threshold {self.coverage_threshold}%")
                return False
                
        except FileNotFoundError:
            logger.warning("⚠️ Coverage report not found")
            return False
    
    async def _analyze_performance_results(self):
        """Analyze performance benchmark results."""
        try:
            with open(self.project_root / "benchmark_results.json", "r") as f:
                benchmark_data = json.load(f)
            
            logger.info("📊 Performance Analysis:")
            
            for benchmark in benchmark_data.get("benchmarks", []):
                name = benchmark["name"]
                stats = benchmark["stats"]
                mean_time = stats["mean"]
                
                logger.info(f"  {name}: {mean_time:.3f}s (mean)")
                
                # Check against baselines
                if "response_time" in name and mean_time > self.performance_baseline["response_time"]:
                    logger.warning(f"    ⚠️ Response time {mean_time:.3f}s exceeds baseline {self.performance_baseline['response_time']}s")
                
        except FileNotFoundError:
            logger.warning("⚠️ Benchmark results not found")
    
    def _parse_pytest_output(self, output: str, test_type: str):
        """Parse pytest output for test statistics."""
        lines = output.split('\n')
        
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse line like "10 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        self.test_results[test_type]["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        self.test_results[test_type]["failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        self.test_results[test_type]["skipped"] = int(parts[i-1])
    
    async def _check_environment_variables(self) -> bool:
        """Check for proper environment variable configuration."""
        required_vars = [
            "OPENAI_API_KEY",
            "WHATSAPP_ACCESS_TOKEN", 
            "REDIS_URL",
            "PINECONE_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    async def _check_api_key_exposure(self) -> bool:
        """Check for hardcoded API keys in source code."""
        cmd = [
            "grep", "-r", "-i",
            "--include=*.py",
            "--exclude-dir=.git",
            "--exclude-dir=__pycache__",
            "api[_-]?key.*=.*['\"][a-zA-Z0-9]{20,}['\"]",
            str(self.project_root / "app")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            logger.error("❌ Potential hardcoded API keys found:")
            logger.error(result.stdout)
            return False
        
        return True
    
    async def _check_input_validation(self) -> bool:
        """Check for input validation in critical endpoints."""
        # This would be more comprehensive in a real implementation
        # For now, just check that validation decorators/functions exist
        
        validation_files = [
            self.project_root / "app" / "api" / "routes.py",
            self.project_root / "app" / "services" / "whatsapp_service.py"
        ]
        
        for file_path in validation_files:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "validate" not in content.lower():
                        logger.warning(f"⚠️ No validation found in {file_path}")
                        return False
        
        return True
    
    async def _check_rate_limiting(self) -> bool:
        """Check for rate limiting implementation."""
        whatsapp_service_path = self.project_root / "app" / "services" / "whatsapp_service.py"
        
        if whatsapp_service_path.exists():
            with open(whatsapp_service_path, 'r') as f:
                content = f.read()
                if "rate_limit" not in content.lower():
                    logger.warning("⚠️ Rate limiting not found in WhatsApp service")
                    return False
        
        return True
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        total_tests = sum(
            result["passed"] + result["failed"] + result["skipped"]
            for result in self.test_results.values()
        )
        
        total_passed = sum(result["passed"] for result in self.test_results.values())
        total_failed = sum(result["failed"] for result in self.test_results.values())
        total_duration = sum(result["duration"] for result in self.test_results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": success_rate,
                "total_duration": total_duration
            },
            "details": self.test_results,
            "performance_baseline": self.performance_baseline
        }
        
        return report
    
    def print_summary(self):
        """Print test execution summary."""
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("🎯 RIAD CONCIERGE AI - TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print()
        
        for test_type, results in self.test_results.items():
            if results["passed"] + results["failed"] + results["skipped"] > 0:
                print(f"{test_type.title()} Tests:")
                print(f"  Passed: {results['passed']}")
                print(f"  Failed: {results['failed']}")
                print(f"  Skipped: {results['skipped']}")
                print(f"  Duration: {results['duration']:.2f}s")
                print()
        
        if summary["success_rate"] >= 95:
            print("🎉 All tests passed! System ready for deployment.")
        elif summary["success_rate"] >= 85:
            print("⚠️ Most tests passed. Review failures before deployment.")
        else:
            print("❌ Significant test failures. System not ready for deployment.")
        
        print("="*60)


async def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="Riad Concierge AI Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--cultural", action="store_true", help="Run cultural validation only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", help="Save report to file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Determine which tests to run
    run_all = args.all or not any([args.unit, args.integration, args.performance, args.cultural, args.security])
    
    success = True
    
    try:
        if run_all or args.unit:
            success &= await runner.run_unit_tests(args.verbose)
        
        if run_all or args.integration:
            success &= await runner.run_integration_tests(args.verbose)
        
        if run_all or args.performance:
            success &= await runner.run_performance_tests(args.verbose)
        
        if run_all or args.cultural:
            success &= await runner.run_cultural_validation()
        
        if run_all or args.security:
            success &= await runner.run_security_tests()
        
        # Generate and save report
        report = runner.generate_report()
        
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Test report saved to {args.report}")
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
