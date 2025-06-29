#!/usr/bin/env python3
"""
Helper script to fix TBB "already found in load path" errors with Prophet
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔧 Prophet TBB Issue Fixer")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider using one.")
    
    # Step 1: Uninstall current prophet
    if not run_command("pip uninstall prophet -y", "Uninstalling current Prophet"):
        print("❌ Failed to uninstall Prophet. Please run manually: pip uninstall prophet -y")
        return
    
    # Step 2: Install prophet without dependencies
    if not run_command("pip install prophet --no-deps", "Installing Prophet without dependencies"):
        print("❌ Failed to install Prophet. Please run manually: pip install prophet --no-deps")
        return
    
    # Step 3: Install required dependencies manually
    dependencies = [
        "cmdstanpy",
        "pystan",
        "numpy",
        "pandas",
        "matplotlib"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"❌ Failed to install {dep}")
            return
    
    print("\n🎉 Prophet installation completed!")
    print("\n📝 Next steps:")
    print("1. Restart your Python environment")
    print("2. Run your Flask application")
    print("3. If you still get TBB errors, the app will use simple trend analysis instead")
    
    print("\n💡 Alternative solution:")
    print("If you continue to have issues, you can:")
    print("1. Use the simple trend analysis (already implemented)")
    print("2. Or try: conda install -c conda-forge prophet")

if __name__ == "__main__":
    main() 