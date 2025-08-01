import subprocess
import sys
import signal

# Paths to frontend and backend directories
FRONTEND_DIR = "frontend"
BACKEND_DIR = "backend"

# Commands to run
# On Windows, use "npm.cmd" instead of "npm"
BACKEND_CMD = ["uvicorn", "main:app", "--reload"]
FRONTEND_CMD = ["npm.cmd", "start"]

# Store running processes
processes = []

def run_process(command, cwd):
    """
    Launch a process in the specified directory
    """
    return subprocess.Popen(command, cwd=cwd, shell=False)

try:
    print("Starting backend server...")
    backend_process = run_process(BACKEND_CMD, BACKEND_DIR)
    processes.append(backend_process)

    print("Starting frontend server...")
    frontend_process = run_process(FRONTEND_CMD, FRONTEND_DIR)
    processes.append(frontend_process)

    print("\nBoth servers are running! Press Ctrl+C to stop.\n")

    # Wait for processes to finish
    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nStopping servers...")
    for p in processes:
        p.send_signal(signal.SIGINT)
    sys.exit(0)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure Node.js/npm and uvicorn are installed and accessible in PATH.")
    sys.exit(1)
