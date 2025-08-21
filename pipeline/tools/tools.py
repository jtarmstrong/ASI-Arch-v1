import subprocess
from typing import Any, Dict

from agents import function_tool
from config import Config


#OLD
#def read_code_file() -> Dict[str, Any]:
#    """Read a code file and return its contents."""
#    source_file = Config.SOURCE_FILE
#    try:
#        with open(source_file, 'r') as f:
#            content = f.read()
#        return {
#            'success': True,
#            'content': content
#        }
#    except Exception as e:
#        return {
#            'success': False,
#            'error': str(e)
#        }

@function_tool
def write_code_file(content: str, program_name: str = None) -> Dict[str, Any]:
    """Write content to a code file."""
    if program_name:
        source_file = f"programs/{program_name}.py"
    else:
        source_file = Config.SOURCE_FILE
    
    try:
        # Ensure programs directory exists
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        
        with open(source_file, 'w') as f:
            f.write(content)
        return {
            'success': True,
            'message': f'Successfully wrote to {source_file}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }





@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents."""
    try:
        with open(file_path, 'r') as f:
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


@function_tool
def write_code_file(content: str) -> Dict[str, Any]:
    """Write content to a code file."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'w') as f:
            f.write(content)
        return {
            'success': True,
            'message': f'Successfully write'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def run_training_script(name: str, script_path: str) -> Dict[str, Any]:
    """Run the training script and return its output."""
    try:
        subprocess.run(['bash', script_path, name], 
                      capture_output=True, 
                      text=True,
                      check=True)
        return {
            'success': True,
            'error': 'Training script executed successfully'
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': e.stderr
        }


@function_tool
def run_plot_script(script_path: str) -> Dict[str, Any]:
    """Run the plotting script."""
    try:
        result = subprocess.run(['python', script_path],
                              capture_output=True,
                              text=True,
                              check=True)
        return {
            'success': True,
            'output': result.stdout,
            'error': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'output': e.stdout,
            'error': e.stderr
        }


async def get_all_tools(*args, **kwargs):
    """Returns all available tools for agents"""
    return [read_code_file, write_code_file]



def run_rag(query: str) -> Dict[str, Any]:
    """Run RAG and return the results."""
    try:
        import requests
        
        response = requests.post(
            f'{Config.RAG}/search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': query,
                'k': 3, 
                'similarity_threshold': 0.5
            }
        )
        
        response.raise_for_status()
        results = response.json()
        
        return {
            'success': True,
            'results': results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
