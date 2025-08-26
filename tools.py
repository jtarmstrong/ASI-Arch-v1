import sys
import os

# Add pipeline to path without importing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))

def read_code_file():
    from tools.tools import read_code_file as _read_code_file
    return _read_code_file()

def write_code_file(content, program_name=None):
    from tools.tools import write_code_file as _write_code_file
    return _write_code_file(content, program_name)

def run_training_script(name, script_path):
    from tools.tools import run_training_script as _run_training_script
    return _run_training_script(name, script_path)

def run_plot_script(script_path):
    from tools.tools import run_plot_script as _run_plot_script
    return _run_plot_script(script_path)

async def get_all_tools():
    from tools.tools import get_all_tools as _get_all_tools
    return await _get_all_tools()

# Export the functions
__all__ = ['read_code_file', 'write_code_file', 'run_training_script', 'run_plot_script', 'get_all_tools']
