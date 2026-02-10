import os
import subprocess
import sys

def run_script(script_path):
    print(f"--- Running {script_path} ---")
    try:
        # Use sys.executable to ensure it uses the same python environment
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        return False
    return True

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    scripts = [
        'utils/prepare_multi_data.py',
        'utils/merge_point2.py',
        'utils/save_conductivity_scaler.py'
    ]
    
    success = True
    for script in scripts:
        if not run_script(script):
            success = False
            break
            
    if success:
        print("\n✅ All data preparation steps completed successfully.")
    else:
        print("\n❌ Data preparation failed.")

if __name__ == "__main__":
    main()
