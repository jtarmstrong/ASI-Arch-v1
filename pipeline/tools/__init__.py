import os
from pathlib import Path

def write_code_file(program_name: str, content: str):
    """Write content to a code file - direct implementation"""
    try:
        # Handle potential double .py extension
        if program_name.endswith('.py'):
            source_file = f"programs/{program_name}"
        else:
            source_file = f"programs/{program_name}.py"
        
        # Ensure programs directory exists
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        
        # Write the file directly
        with open(source_file, 'w') as f:
            f.write(content)
        
        # Verify file was written
        if os.path.exists(source_file) and os.path.getsize(source_file) > 0:
            return {
                'success': True,
                'message': f'Successfully wrote to {source_file}',
                'file_path': source_file,
                'file_size': os.path.getsize(source_file)
            }
        else:
            return {
                'success': False,
                'error': f'File {source_file} was not created or is empty'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Add missing attributes that the pipeline expects
write_code_file.name = "write_code_file"
write_code_file.description = "Write content to a code file"

def read_code_file():
    """Read a code file - direct implementation"""
    try:
        from config import Config
        source_file = Config.SOURCE_FILE
        with open(source_file, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Add missing attributes
read_code_file.name = "read_code_file"
read_code_file.description = "Read a code file and return its contents"

# Import other tools that might not have the same issues
from .tools import (
    run_training_script,
    run_plot_script, 
    run_rag,
    get_all_tools,
    validate_planner_output
)

__all__ = [
    'read_code_file',
    'write_code_file',
    'run_training_script', 
    'run_plot_script',
    'run_rag',
    'get_all_tools',
    'validate_planner_output'
]
