#!/usr/bin/env python3
"""
MAIF Package Build and Test Script

This script builds the MAIF package and runs basic tests to ensure
everything works correctly before publishing to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"🔄 {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   ❌ Stderr: {e.stderr}")
        return False

def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   🗑️  Removed {path}")
    
    # Clean Python cache
    for path in Path('.').rglob('__pycache__'):
        if path.is_dir():
            shutil.rmtree(path)
    
    for path in Path('.').rglob('*.pyc'):
        path.unlink()

def check_dependencies():
    """Check if required build tools are installed."""
    print("🔍 Checking build dependencies...")
    
    required_tools = ['build', 'twine']
    missing_tools = []
    
    for tool in required_tools:
        if not run_command(f"python -m {tool} --help > /dev/null 2>&1", f"Checking {tool}"):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ Missing tools: {missing_tools}")
        print("Install with: pip install build twine")
        return False
    
    print("✅ All build tools available")
    return True

def validate_package_structure():
    """Validate package structure."""
    print("📦 Validating package structure...")
    
    required_files = [
        'pyproject.toml',
        'setup.py',
        'README.md',
        'LICENSE',
        'MANIFEST.in',
        'maif/__init__.py',
        'maif_api.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Package structure valid")
    return True

def build_package():
    """Build the package."""
    print("🏗️  Building package...")
    
    if not run_command("python -m build", "Building wheel and source distribution"):
        return False
    
    # Check if files were created
    dist_files = list(Path('dist').glob('*'))
    if not dist_files:
        print("❌ No distribution files created")
        return False
    
    print(f"✅ Built {len(dist_files)} distribution files:")
    for file in dist_files:
        print(f"   📦 {file}")
    
    return True

def test_package():
    """Test the built package."""
    print("🧪 Testing package...")
    
    # Test import
    test_script = """
import sys
sys.path.insert(0, '.')

try:
    import maif
    print(f"✅ MAIF version: {maif.__version__}")
    
    # Test simple API
    if hasattr(maif, 'create_maif'):
        test_maif = maif.create_maif("test_agent")
        test_maif.add_text("Test content")
        print("✅ Simple API works")
    else:
        print("⚠️  Simple API not available")
    
    # Test core functionality
    from maif.core import MAIFEncoder
    encoder = MAIFEncoder("test_core")
    print("✅ Core functionality works")
    
    print("✅ All tests passed")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
    
    if not run_command(f'python -c "{test_script}"', "Testing package imports"):
        return False
    
    return True

def check_package_metadata():
    """Check package metadata."""
    print("📋 Checking package metadata...")
    
    if not run_command("python -m twine check dist/*", "Validating package metadata"):
        return False
    
    print("✅ Package metadata valid")
    return True

def show_package_info():
    """Show package information."""
    print("📊 Package Information:")
    
    # Show package contents
    if Path('dist').exists():
        print("\n📦 Distribution files:")
        for file in Path('dist').iterdir():
            size = file.stat().st_size / 1024  # KB
            print(f"   {file.name} ({size:.1f} KB)")
    
    # Show package structure
    print("\n📁 Package structure:")
    for root, dirs, files in os.walk('maif'):
        level = root.replace('maif', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.endswith('.pyc'):
                print(f"{subindent}{file}")

def main():
    """Main build process."""
    print("🚀 MAIF Package Build Process")
    print("=" * 40)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    steps = [
        ("Clean build artifacts", clean_build),
        ("Check dependencies", check_dependencies),
        ("Validate package structure", validate_package_structure),
        ("Build package", build_package),
        ("Test package", test_package),
        ("Check package metadata", check_package_metadata),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        if not step_func():
            print(f"\n❌ Build failed at: {step_name}")
            sys.exit(1)
    
    print("\n🎉 Build completed successfully!")
    show_package_info()
    
    print("\n📤 Next steps:")
    print("   1. Test install: pip install dist/*.whl")
    print("   2. Upload to TestPyPI: python -m twine upload --repository testpypi dist/*")
    print("   3. Upload to PyPI: python -m twine upload dist/*")

if __name__ == "__main__":
    main()