#!/usr/bin/env python3
"""
Pre-publish analysis script for QuData package.
This script analyzes the project structure, dependencies, and package integrity
before publishing to TestPyPI.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import importlib.util
import ast

def analyze_project_structure() -> Dict[str, Any]:
    """Analyze the project directory structure."""
    print("🔍 Analyzing project structure...")
    
    structure = {}
    src_path = Path("src")
    
    if not src_path.exists():
        return {"error": "src directory not found"}
    
    # Count files by type
    file_counts = {}
    total_files = 0
    
    for root, dirs, files in os.walk(src_path):
        for file in files:
            total_files += 1
            ext = Path(file).suffix
            file_counts[ext] = file_counts.get(ext, 0) + 1
    
    structure["total_files"] = total_files
    structure["file_types"] = file_counts
    
    # Check for required files
    required_files = [
        "pyproject.toml",
        "README.md",
        "src/forge/__init__.py",
        "src/forge/cli.py",
        "src/forge/config.py",
        "src/forge/models.py",
        "src/forge/pipeline.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    structure["missing_required_files"] = missing_files
    structure["has_all_required_files"] = len(missing_files) == 0
    
    return structure

def analyze_dependencies() -> Dict[str, Any]:
    """Analyze package dependencies."""
    print("📦 Analyzing dependencies...")
    
    deps_info = {}
    
    # Read pyproject.toml dependencies
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("⚠️  Warning: Cannot read pyproject.toml - tomllib/tomli not available")
            return {"error": "Cannot read pyproject.toml"}
    
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        
        deps = pyproject.get("project", {}).get("dependencies", [])
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        
        deps_info["main_dependencies"] = deps
        deps_info["optional_dependencies"] = optional_deps
        deps_info["total_main_deps"] = len(deps)
        deps_info["total_optional_deps"] = sum(len(v) for v in optional_deps.values())
        
    except Exception as e:
        deps_info["error"] = f"Error reading pyproject.toml: {e}"
    
    return deps_info

def analyze_imports() -> Dict[str, Any]:
    """Analyze import statements in the codebase."""
    print("🔗 Analyzing imports...")
    
    imports_info = {
        "internal_imports": set(),
        "external_imports": set(),
        "import_errors": []
    }
    
    src_path = Path("src")
    
    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('forge'):
                                imports_info["internal_imports"].add(alias.name)
                            else:
                                imports_info["external_imports"].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            if node.module.startswith('forge'):
                                imports_info["internal_imports"].add(node.module)
                            else:
                                imports_info["external_imports"].add(node.module)
            except SyntaxError as e:
                imports_info["import_errors"].append(f"{py_file}: {e}")
                
        except Exception as e:
            imports_info["import_errors"].append(f"{py_file}: {e}")
    
    # Convert sets to lists for JSON serialization
    imports_info["internal_imports"] = sorted(list(imports_info["internal_imports"]))
    imports_info["external_imports"] = sorted(list(imports_info["external_imports"]))
    
    return imports_info

def test_package_import() -> Dict[str, Any]:
    """Test if the package can be imported successfully."""
    print("🧪 Testing package import...")
    
    import_info = {
        "can_import_main": False,
        "can_import_cli": False,
        "import_errors": []
    }
    
    try:
        # Add src to path temporarily
        sys.path.insert(0, str(Path("src").absolute()))
        
        # Test main package import
        try:
            import qudata
            import_info["can_import_main"] = True
            import_info["package_version"] = getattr(forge, '__version__', 'unknown')
        except Exception as e:
            import_info["import_errors"].append(f"Main package: {e}")
        
        # Test CLI import
        try:
            from qudata import cli
            import_info["can_import_cli"] = True
        except Exception as e:
            import_info["import_errors"].append(f"CLI module: {e}")
        
        # Test key components
        components_to_test = [
            "QuDataPipeline",
            "ConfigManager", 
            "FileTypeDetector",
            "ComprehensiveCleaningPipeline"
        ]
        
        import_info["component_imports"] = {}
        for component in components_to_test:
            try:
                getattr(forge, component)
                import_info["component_imports"][component] = True
            except Exception as e:
                import_info["component_imports"][component] = False
                import_info["import_errors"].append(f"{component}: {e}")
        
    except Exception as e:
        import_info["import_errors"].append(f"General import error: {e}")
    finally:
        # Remove src from path
        if str(Path("src").absolute()) in sys.path:
            sys.path.remove(str(Path("src").absolute()))
    
    return import_info

def check_build_artifacts() -> Dict[str, Any]:
    """Check if build artifacts exist and are valid."""
    print("🏗️  Checking build artifacts...")
    
    build_info = {
        "dist_exists": False,
        "wheel_exists": False,
        "sdist_exists": False,
        "artifacts": []
    }
    
    dist_path = Path("dist")
    if dist_path.exists():
        build_info["dist_exists"] = True
        artifacts = list(dist_path.glob("*"))
        build_info["artifacts"] = [str(a.name) for a in artifacts]
        
        # Check for wheel and sdist
        build_info["wheel_exists"] = any(a.suffix == ".whl" for a in artifacts)
        build_info["sdist_exists"] = any(a.suffix == ".gz" for a in artifacts)
    
    return build_info

def run_tests() -> Dict[str, Any]:
    """Run basic tests to ensure package functionality."""
    print("🧪 Running basic tests...")
    
    test_info = {
        "tests_run": False,
        "tests_passed": False,
        "test_output": ""
    }
    
    try:
        # Run a simple test
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/unit/test_models.py", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=60)
        
        test_info["tests_run"] = True
        test_info["tests_passed"] = result.returncode == 0
        test_info["test_output"] = result.stdout + result.stderr
        test_info["return_code"] = result.returncode
        
    except subprocess.TimeoutExpired:
        test_info["test_output"] = "Tests timed out after 60 seconds"
    except Exception as e:
        test_info["test_output"] = f"Error running tests: {e}"
    
    return test_info

def check_twine_validation() -> Dict[str, Any]:
    """Check if the package passes twine validation."""
    print("✅ Checking twine validation...")
    
    twine_info = {
        "validation_run": False,
        "validation_passed": False,
        "validation_output": ""
    }
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "twine", "check", "dist/*"
        ], capture_output=True, text=True, timeout=30)
        
        twine_info["validation_run"] = True
        twine_info["validation_passed"] = result.returncode == 0
        twine_info["validation_output"] = result.stdout + result.stderr
        twine_info["return_code"] = result.returncode
        
    except subprocess.TimeoutExpired:
        twine_info["validation_output"] = "Twine check timed out after 30 seconds"
    except Exception as e:
        twine_info["validation_output"] = f"Error running twine check: {e}"
    
    return twine_info

def generate_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the analysis results."""
    print("📋 Generating summary...")
    
    summary = {
        "ready_for_publish": True,
        "critical_issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check critical issues
    structure = analysis_results.get("structure", {})
    if not structure.get("has_all_required_files", True):
        summary["critical_issues"].append("Missing required files")
        summary["ready_for_publish"] = False
    
    imports = analysis_results.get("imports", {})
    if imports.get("import_errors"):
        summary["critical_issues"].append("Import errors found")
        summary["ready_for_publish"] = False
    
    package_import = analysis_results.get("package_import", {})
    if not package_import.get("can_import_main", False):
        summary["critical_issues"].append("Cannot import main package")
        summary["ready_for_publish"] = False
    
    build = analysis_results.get("build", {})
    if not (build.get("wheel_exists", False) and build.get("sdist_exists", False)):
        summary["critical_issues"].append("Missing build artifacts")
        summary["ready_for_publish"] = False
    
    twine = analysis_results.get("twine", {})
    if not twine.get("validation_passed", False):
        summary["critical_issues"].append("Twine validation failed")
        summary["ready_for_publish"] = False
    
    # Add warnings
    tests = analysis_results.get("tests", {})
    if not tests.get("tests_passed", False):
        summary["warnings"].append("Some tests are failing")
    
    deps = analysis_results.get("dependencies", {})
    if deps.get("total_main_deps", 0) > 20:
        summary["warnings"].append("Large number of dependencies")
    
    # Add recommendations
    if structure.get("total_files", 0) > 100:
        summary["recommendations"].append("Consider organizing code into fewer, more focused modules")
    
    if not package_import.get("can_import_cli", False):
        summary["recommendations"].append("CLI import issues - check command-line interface")
    
    return summary

def main():
    """Run the complete pre-publish analysis."""
    print("🚀 QuData Pre-Publish Analysis")
    print("=" * 50)
    
    analysis_results = {}
    
    # Run all analyses
    analysis_results["structure"] = analyze_project_structure()
    analysis_results["dependencies"] = analyze_dependencies()
    analysis_results["imports"] = analyze_imports()
    analysis_results["package_import"] = test_package_import()
    analysis_results["build"] = check_build_artifacts()
    analysis_results["tests"] = run_tests()
    analysis_results["twine"] = check_twine_validation()
    analysis_results["summary"] = generate_summary(analysis_results)
    
    # Save results to file
    with open("pre_publish_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Print summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    summary = analysis_results["summary"]
    
    if summary["ready_for_publish"]:
        print("✅ Package is READY for TestPyPI publication!")
    else:
        print("❌ Package has CRITICAL ISSUES that must be fixed:")
        for issue in summary["critical_issues"]:
            print(f"   • {issue}")
    
    if summary["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in summary["warnings"]:
            print(f"   • {warning}")
    
    if summary["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")
    
    # Print key metrics
    print(f"\n📈 Key Metrics:")
    print(f"   • Total files: {analysis_results['structure'].get('total_files', 'unknown')}")
    print(f"   • Main dependencies: {analysis_results['dependencies'].get('total_main_deps', 'unknown')}")
    print(f"   • External imports: {len(analysis_results['imports'].get('external_imports', []))}")
    print(f"   • Build artifacts: {len(analysis_results['build'].get('artifacts', []))}")
    
    print(f"\n📄 Full analysis saved to: pre_publish_analysis.json")
    
    return 0 if summary["ready_for_publish"] else 1

if __name__ == "__main__":
    sys.exit(main())