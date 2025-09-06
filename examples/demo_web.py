"""
Web interface demo for Scene2Sim.
"""
import sys
from pathlib import Path

# Add scene2sim to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene2sim.web.app import run_server

def main():
    print("Starting Scene2Sim Web Interface")
    print("=" * 40)
    print("Open your browser to: http://localhost:8000")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        run_server(host="localhost", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
