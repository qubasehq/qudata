#!/usr/bin/env python3
"""
Upload script for TestPyPI publication.
This script provides instructions and handles the upload process to TestPyPI.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met for upload."""
    print("🔍 Checking prerequisites...")
    
    issues = []
    
    # Check if dist directory exists
    if not Path("dist").exists():
        issues.append("dist/ directory not found. Run 'python -m build' first.")
    
    # Check if build artifacts exist
    dist_files = list(Path("dist").glob("*")) if Path("dist").exists() else []
    if not any(f.suffix == ".whl" for f in dist_files):
        issues.append("No wheel (.whl) file found in dist/")
    if not any(f.suffix == ".gz" for f in dist_files):
        issues.append("No source distribution (.tar.gz) file found in dist/")
    
    # Check if twine is available
    try:
        subprocess.run([sys.executable, "-m", "twine", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        issues.append("twine is not installed. Run 'pip install twine'")
    
    return issues

def print_setup_instructions():
    """Print instructions for setting up TestPyPI authentication."""
    print("\n📋 TestPyPI Setup Instructions")
    print("=" * 50)
    print("1. Create a TestPyPI account at: https://test.pypi.org/account/register/")
    print("2. Verify your email address")
    print("3. Generate an API token:")
    print("   - Go to: https://test.pypi.org/manage/account/token/")
    print("   - Click 'Add API token'")
    print("   - Set scope to 'Entire account' (for first upload)")
    print("   - Copy the generated token (starts with 'pypi-')")
    print("4. Configure authentication:")
    print("   Option A - Environment variables:")
    print("     export TWINE_USERNAME=__token__")
    print("     export TWINE_PASSWORD=pypi-your-token-here")
    print("   Option B - .pypirc file:")
    print("     Create ~/.pypirc with:")
    print("     [testpypi]")
    print("     username = __token__")
    print("     password = pypi-your-token-here")
    print("5. Run the upload command")

def upload_to_testpypi():
    """Upload the package to TestPyPI."""
    print("\n🚀 Uploading to TestPyPI...")
    
    # Check if authentication is configured
    has_env_auth = (os.getenv("TWINE_USERNAME") and os.getenv("TWINE_PASSWORD"))
    has_pypirc = Path.home().joinpath(".pypirc").exists()
    
    if not (has_env_auth or has_pypirc):
        print("⚠️  No authentication configured!")
        print("Please set up authentication first (see instructions above)")
        return False
    
    try:
        # Run twine upload
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", "testpypi",
            "--verbose",
            "dist/*"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("✅ Upload successful!")
        print("\n🎉 Package uploaded to TestPyPI!")
        print("📦 View your package at: https://test.pypi.org/project/qudata/")
        print("🧪 Test installation with:")
        print("   pip install -i https://test.pypi.org/simple/ qudata")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed with return code {e.returncode}")
        print("Check the error messages above for details.")
        return False

def test_installation():
    """Test installing the package from TestPyPI."""
    print("\n🧪 Testing installation from TestPyPI...")
    
    try:
        # Create a temporary virtual environment for testing
        import tempfile
        import venv
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_env"
            
            # Create virtual environment
            venv.create(venv_path, with_pip=True)
            
            # Get python executable in venv
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
            
            # Install from TestPyPI
            install_cmd = [
                str(python_exe), "-m", "pip", "install",
                "-i", "https://test.pypi.org/simple/",
                "--extra-index-url", "https://pypi.org/simple/",
                "qudata"
            ]
            
            print(f"Running: {' '.join(install_cmd)}")
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Installation successful!")
                
                # Test import
                test_cmd = [str(python_exe), "-c", "import qudata; print('Import successful!')"]
                test_result = subprocess.run(test_cmd, capture_output=True, text=True)
                
                if test_result.returncode == 0:
                    print("✅ Import test successful!")
                    return True
                else:
                    print("❌ Import test failed:")
                    print(test_result.stderr)
                    return False
            else:
                print("❌ Installation failed:")
                print(result.stderr)
                return False
                
    except Exception as e:
        print(f"❌ Test installation failed: {e}")
        return False

def main():
    """Main function to handle TestPyPI upload process."""
    print("🚀 QuData TestPyPI Upload Script")
    print("=" * 50)
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"   • {issue}")
        return 1
    
    print("✅ Prerequisites check passed!")
    
    # Print setup instructions
    print_setup_instructions()
    
    # Ask user if they want to proceed
    print("\n" + "=" * 50)
    response = input("Do you want to proceed with upload? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Upload cancelled.")
        return 0
    
    # Attempt upload
    if upload_to_testpypi():
        # Ask if user wants to test installation
        test_response = input("\nDo you want to test installation from TestPyPI? (y/N): ").strip().lower()
        if test_response in ['y', 'yes']:
            test_installation()
        
        print("\n🎉 TestPyPI upload process completed!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())