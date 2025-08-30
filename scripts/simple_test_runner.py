#!/usr/bin/env python3
"""
Simplified test runner for Riad Concierge AI
Tests core functionality without complex dependencies
"""

import sys
import os
import time
from pathlib import Path
import json
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SimpleTestRunner:
    """Simplified test runner for core functionality validation."""
    
    def __init__(self):
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "duration": 0
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a single test function."""
        logger.info(f"🧪 Running {test_name}...")
        
        try:
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            
            self.test_results["passed"] += 1
            logger.info(f"✅ {test_name} passed ({duration:.3f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results["failed"] += 1
            logger.error(f"❌ {test_name} failed: {e}")
            return False
        
        finally:
            self.test_results["total"] += 1
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        required_dirs = [
            "app",
            "app/agents",
            "app/api", 
            "app/core",
            "app/models",
            "app/services",
            "app/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "scripts",
            "docs"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory {dir_path} not found"
            assert full_path.is_dir(), f"{dir_path} is not a directory"
        
        logger.info("✓ All required directories exist")
    
    def test_core_files_exist(self):
        """Test that core files exist."""
        required_files = [
            "app/main.py",
            "app/agents/riad_agent.py",
            "app/api/routes.py",
            "app/core/config.py",
            "app/models/agent_state.py",
            "app/models/instructor_models.py",
            "app/services/agent_service.py",
            "app/services/whatsapp_service.py",
            "app/services/cultural_service.py",
            "app/services/knowledge_service.py",
            "app/services/pms_service.py",
            "app/services/proactive_service.py",
            "app/utils/logger.py",
            "tests/conftest.py",
            "tests/unit/test_cultural_service.py",
            "tests/unit/test_whatsapp_service.py",
            "tests/integration/test_agent_workflow.py",
            "tests/performance/test_benchmarks.py",
            "scripts/run_tests.py",
            "docs/TESTING_STRATEGY.md",
            "pyproject.toml",
            "README.md"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file {file_path} not found"
            assert full_path.is_file(), f"{file_path} is not a file"
        
        logger.info("✓ All required files exist")
    
    def test_python_syntax(self):
        """Test that Python files have valid syntax."""
        python_files = []
        
        # Find all Python files
        for root, dirs, files in os.walk(project_root / "app"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        for root, dirs, files in os.walk(project_root / "tests"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        for root, dirs, files in os.walk(project_root / "scripts"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to compile the code
                compile(content, str(py_file), 'exec')
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                # Skip files with import errors for now
                pass
        
        assert len(syntax_errors) == 0, f"Syntax errors found: {syntax_errors}"
        logger.info(f"✓ All {len(python_files)} Python files have valid syntax")
    
    def test_configuration_files(self):
        """Test configuration files are valid."""
        # Test pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        with open(pyproject_path, 'r') as f:
            content = f.read()
            assert '[tool.poetry]' in content, "Poetry configuration not found"
            assert 'langgraph' in content, "LangGraph dependency not found"
            assert 'instructor' in content, "Instructor dependency not found"
            assert 'pytest' in content, "Pytest dependency not found"
        
        # Test .env.example exists
        env_example_path = project_root / ".env.example"
        assert env_example_path.exists(), ".env.example not found"
        
        with open(env_example_path, 'r') as f:
            content = f.read()
            assert 'OPENAI_API_KEY' in content, "OpenAI API key not in .env.example"
            assert 'WHATSAPP_ACCESS_TOKEN' in content, "WhatsApp token not in .env.example"
        
        logger.info("✓ Configuration files are valid")
    
    def test_documentation_completeness(self):
        """Test that documentation is complete."""
        readme_path = project_root / "README.md"
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            assert 'Riad Concierge AI' in readme_content, "README missing project title"
            assert 'Architecture' in readme_content, "README missing architecture section"
            assert 'Installation' in readme_content, "README missing installation section"
        
        testing_doc_path = project_root / "docs/TESTING_STRATEGY.md"
        with open(testing_doc_path, 'r') as f:
            testing_content = f.read()
            assert 'Testing Strategy' in testing_content, "Testing strategy doc missing title"
            assert 'Unit Tests' in testing_content, "Testing strategy missing unit tests section"
            assert 'Performance Tests' in testing_content, "Testing strategy missing performance section"
        
        logger.info("✓ Documentation is complete")
    
    def test_test_files_structure(self):
        """Test that test files have proper structure."""
        test_files = [
            project_root / "tests/unit/test_cultural_service.py",
            project_root / "tests/unit/test_whatsapp_service.py",
            project_root / "tests/integration/test_agent_workflow.py",
            project_root / "tests/performance/test_benchmarks.py"
        ]
        
        for test_file in test_files:
            with open(test_file, 'r') as f:
                content = f.read()
                assert 'import pytest' in content, f"{test_file} missing pytest import"
                assert 'class Test' in content, f"{test_file} missing test class"
                assert 'async def test_' in content or 'def test_' in content, f"{test_file} missing test methods"
        
        logger.info("✓ Test files have proper structure")
    
    def run_all_tests(self):
        """Run all simplified tests."""
        logger.info("🚀 Starting simplified test suite...")
        start_time = time.time()
        
        tests = [
            ("Project Structure", self.test_project_structure),
            ("Core Files Exist", self.test_core_files_exist),
            ("Python Syntax", self.test_python_syntax),
            ("Configuration Files", self.test_configuration_files),
            ("Documentation Completeness", self.test_documentation_completeness),
            ("Test Files Structure", self.test_test_files_structure)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        self.test_results["duration"] = time.time() - start_time
        self.print_summary()
    
    def print_summary(self):
        """Print test execution summary."""
        success_rate = (self.test_results["passed"] / self.test_results["total"] * 100) if self.test_results["total"] > 0 else 0
        
        print("\n" + "="*60)
        print("🎯 RIAD CONCIERGE AI - SIMPLIFIED TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.test_results['total']}")
        print(f"Passed: {self.test_results['passed']} ✅")
        print(f"Failed: {self.test_results['failed']} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Duration: {self.test_results['duration']:.2f}s")
        print()
        
        if success_rate == 100:
            print("🎉 All simplified tests passed! Core system structure is valid.")
            print("📝 Next step: Install dependencies and run full test suite.")
        elif success_rate >= 80:
            print("⚠️ Most tests passed. Review failures and fix issues.")
        else:
            print("❌ Significant failures. Core system structure needs attention.")
        
        print("="*60)
        
        return success_rate == 100


def main():
    """Main test execution."""
    runner = SimpleTestRunner()
    success = runner.run_all_tests()
    
    # Save results
    results = {
        "timestamp": time.time(),
        "test_type": "simplified_structure_validation",
        "results": runner.test_results,
        "success": success
    }
    
    with open(project_root / "simple_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
