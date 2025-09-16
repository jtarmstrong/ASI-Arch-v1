import sys; sys.path.append("/tmp")
import os
import subprocess
import ast
import re
import shutil
from typing import Any, Dict
from datetime import datetime
from pathlib import Path
from agents import function_tool
from config import Config

def clean_generated_code(content: str) -> str:
  #  exec(open("/tmp/debug_cleaning.py").read())
    """Remove markdown artifacts and ensure proper Python file structure."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip markdown code fences
        if stripped.startswith('```'):
            continue
            
        # Skip standalone language identifiers (like "python", "py", etc.)
        if stripped.lower() in ['python', 'py'] and i < 10:  # Check more lines, case insensitive
            continue
            
        # Skip empty lines at the beginning
        if not stripped and not cleaned_lines:
            continue
        
        # Clean markdown formatting from the line
        cleaned_line = clean_markdown_formatting(line)
        
        # Fix malformed future imports
        cleaned_line = fix_future_imports(cleaned_line)
        
        # Fix encoding declarations
        cleaned_line = fix_encoding_declaration(cleaned_line)
        
        # Fix missing operators
        cleaned_line = fix_missing_operators(cleaned_line)
        
        cleaned_lines.append(cleaned_line)
    
    # Ensure we don't have trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def fix_missing_operators(line: str) -> str:
    """Fix missing operators in mathematical expressions."""
    # Fix missing multiplication operator - variable followed by bracket/identifier
    line = re.sub(r'(\w)\s+(\w+\[)', r'\1 * \2', line)
    
    # Handle function calls: variable space function_name(
    line = re.sub(r'(\w)\s+(\w+\()', r'\1 * \2', line)
    
    # Also handle cases like: tensor   other_tensor
    line = re.sub(r'(\w)\s+(\w+)(?=\s*[,\)\]\n]|$)', r'\1 * \2', line)
    
    # Fix missing multiplication with parentheses: expr   (something)
    line = re.sub(r'(\w)\s+(\()', r'\1 * \2', line)
    
    return line

def fix_encoding_declaration(line: str) -> str:
    """Fix malformed encoding declarations."""
    # Fix -- style encoding to -*- style
    if re.match(r'#\s*--\s*coding[:=]\s*utf-8\s*--', line):
        return '# -*- coding: utf-8 -*-'
    
    # Also handle other common malformations
    if re.match(r'#\s*coding[:=]\s*utf-8', line) and '-*-' not in line:
        return '# -*- coding: utf-8 -*-'
        
    return line

def fix_future_imports(line: str) -> str:
    """Fix malformed future import statements."""
    # Fix markdown bold asterisks in future imports (most common issue)
    line = re.sub(r'from\s+\*\*future\*\*\s+import', 'from __future__ import', line)
    
    # Fix missing double underscores in future imports
    line = re.sub(r'from\s+future\s+import', 'from __future__ import', line)
    
    # Also handle other common future import malformations
    line = re.sub(r'from\s+_future_\s+import', 'from __future__ import', line)
    line = re.sub(r'from\s+__future\s+import', 'from __future__ import', line)
    line = re.sub(r'from\s+future__\s+import', 'from __future__ import', line)
    
    return line

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown bold, italic, and other formatting from a line."""
    # Remove markdown bold (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove markdown italic (*text*)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove any stray markdown asterisks that might be left
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)
    
    return text

@function_tool
def read_code_file() -> Dict[str, Any]:
    """Read a code file and return its contents."""
    source_file = Config.SOURCE_FILE
    try:
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
def write_code_file(program_name: str, content: str) -> Dict[str, Any]:
    """Write content to a code file.
    
    Args:
        program_name: Name of the program file (without .py extension)
        content: Python code content to write
    """
    # Handle potential double .py extension
    if program_name.endswith('.py'):
        source_file = f"programs/{program_name}"
    else:
        source_file = f"programs/{program_name}.py"
    
    try:
        # Clean markdown artifacts from content
        cleaned_content = clean_generated_code(content)
        
        # Ensure programs directory exists
        os.makedirs(os.path.dirname(source_file), exist_ok=True)
        
        with open(source_file, 'w') as f:
            f.write(cleaned_content)
            
        # Verify file was written successfully
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

# VALIDATION TOOLS

@function_tool
def validate_python_syntax(content: str) -> Dict[str, Any]:
    """Validate Python syntax without saving to file."""
    try:
        # Clean content first
        cleaned_content = clean_generated_code(content)
        
        # Try to parse as AST
        ast.parse(cleaned_content)
        
        return {
            'success': True,
            'valid': True,
            'cleaned_content': cleaned_content,
            'message': 'Syntax is valid'
        }
    except SyntaxError as e:
        return {
            'success': True,
            'valid': False,
            'error': str(e),
            'line': e.lineno,
            'column': e.offset,
            'text': e.text
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@function_tool
def check_required_imports(content: str, required_imports: list = None) -> Dict[str, Any]:
    """Check if required imports are present and properly placed."""
    if required_imports is None:
        required_imports = ['torch', 'torch.nn', 'typing']
    
    lines = content.split('\n')
    found_imports = {}
    future_import_line = None
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for future imports
        if 'from __future__ import' in stripped:
            future_import_line = i
            
        # Check for required imports
        for req_import in required_imports:
            if f'import {req_import}' in stripped or f'from {req_import}' in stripped:
                found_imports[req_import] = i
    
    missing_imports = [imp for imp in required_imports if imp not in found_imports]
    
    return {
        'success': True,
        'found_imports': found_imports,
        'missing_imports': missing_imports,
        'future_import_line': future_import_line,
        'future_import_first': future_import_line == 1 or (future_import_line == 2 and '# -*-' in lines[0])
    }

@function_tool
def assess_generation_quality(content: str) -> Dict[str, Any]:
    """Quick quality assessment of generated code."""
    issues = []
    
    # Check for common AI generation artifacts
    if content.count('```') > 0:
        issues.append('Contains markdown code fences')
    if re.search(r'^\s*python\s*$', content, re.MULTILINE):
        issues.append('Contains standalone language identifier')
    if 'TODO' in content or 'FIXME' in content:
        issues.append('Contains TODO/FIXME comments')
    
    # Check structure
    has_class = bool(re.search(r'class\s+\w+', content))
    has_torch_compile = '@torch.compile' in content
    
    return {
        'success': True,
        'quality_score': max(0, 100 - len(issues) * 20),
        'issues': issues,
        'has_class': has_class,
        'has_torch_compile': has_torch_compile
    }

# BACKUP AND VERSIONING

@function_tool
def backup_working_version(program_name: str, version_label: str = None) -> Dict[str, Any]:
    """Backup a working version of the program."""
    try:
        source_file = f"programs/{program_name}.py"
        if not os.path.exists(source_file):
            return {'success': False, 'error': f'Source file {source_file} not found'}
        
        # Create backup directory
        backup_dir = Path("programs/backups")
        backup_dir.mkdir(exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if version_label:
            backup_name = f"{program_name}_{version_label}_{timestamp}.py"
        else:
            backup_name = f"{program_name}_backup_{timestamp}.py"
        
        backup_path = backup_dir / backup_name
        shutil.copy2(source_file, backup_path)
        
        return {
            'success': True,
            'backup_path': str(backup_path),
            'message': f'Backed up to {backup_name}'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@function_tool
def restore_backup(program_name: str, backup_filename: str = None) -> Dict[str, Any]:
    """Restore from a backup version."""
    try:
        backup_dir = Path("programs/backups")
        
        if backup_filename:
            backup_path = backup_dir / backup_filename
        else:
            # Find most recent backup
            backups = list(backup_dir.glob(f"{program_name}_*.py"))
            if not backups:
                return {'success': False, 'error': 'No backups found'}
            backup_path = max(backups, key=lambda p: p.stat().st_mtime)
        
        if not backup_path.exists():
            return {'success': False, 'error': f'Backup file {backup_path} not found'}
        
        source_file = f"programs/{program_name}.py"
        shutil.copy2(backup_path, source_file)
        
        return {
            'success': True,
            'restored_from': str(backup_path),
            'message': f'Restored {program_name}.py from {backup_path.name}'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# DEBUGGING HELPERS

@function_tool
def extract_error_details(error_text: str) -> Dict[str, Any]:
    """Parse error text to extract structured information."""
    error_info = {
        'error_type': None,
        'line_number': None,
        'column': None,
        'message': None,
        'suggestions': []
    }
    
    # Common error patterns
    patterns = {
        'syntax_error': r'SyntaxError: (.+)',
        'import_error': r'ImportError: (.+)',
        'module_not_found': r'ModuleNotFoundError: (.+)',
        'line_info': r'File "(.+)", line (\d+)',
        'future_import_error': r'from __future__ imports must occur at the beginning'
    }
    
    for pattern_name, pattern in patterns.items():
        match = re.search(pattern, error_text)
        if match:
            if pattern_name == 'line_info':
                error_info['line_number'] = int(match.group(2))
            elif pattern_name == 'future_import_error':
                error_info['error_type'] = 'future_import_placement'
                error_info['suggestions'].append('Move from __future__ import to line 1')
            else:
                error_info['error_type'] = pattern_name
                error_info['message'] = match.group(1)
    
    return {
        'success': True,
        'error_info': error_info,
        'full_text': error_text
    }

# FILE MANAGEMENT

@function_tool
def fix_double_extensions() -> Dict[str, Any]:
    """Fix files with double .py.py extensions."""
    try:
        programs_dir = Path("programs")
        fixed_files = []
        
        for py_file in programs_dir.glob("*.py.py"):
            # Rename to single .py extension
            new_name = py_file.with_suffix('')  # Remove the last .py
            py_file.rename(new_name)
            fixed_files.append(f"{py_file.name} -> {new_name.name}")
        
        return {
            'success': True,
            'fixed_files': fixed_files,
            'count': len(fixed_files)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@function_tool
def list_generated_programs() -> Dict[str, Any]:
    """List all generated programs with metadata."""
    try:
        programs_dir = Path("programs")
        if not programs_dir.exists():
            return {'success': True, 'programs': []}
        
        programs = []
        for py_file in programs_dir.glob("*.py"):
            stat = py_file.stat()
            programs.append({
                'name': py_file.stem,
                'filename': py_file.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'path': str(py_file)
            })
        
        return {
            'success': True,
            'programs': sorted(programs, key=lambda x: x['modified'], reverse=True)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@function_tool
def cleanup_failed_programs(older_than_hours: int = 24) -> Dict[str, Any]:
    """Clean up old failed program files."""
    try:
        programs_dir = Path("programs")
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        
        cleaned_files = []
        for py_file in programs_dir.glob("*.py"):
            if py_file.stat().st_mtime < cutoff_time:
                # Quick syntax check - if invalid, it's probably a failed generation
                try:
                    with open(py_file) as f:
                        content = f.read()
                    ast.parse(content)
                except:
                    # Invalid syntax, probably failed
                    py_file.unlink()
                    cleaned_files.append(py_file.name)
        
        return {
            'success': True,
            'cleaned_files': cleaned_files,
            'count': len(cleaned_files)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# CONTENT INSPECTION

@function_tool
def test_content_cleaning(file_path: str = "programs/generated_program.py") -> Dict[str, Any]:
    """Test content cleaning on a specific file without modifying it."""
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        cleaned_content = clean_generated_code(original_content)
        
        # Show first 10 lines of each
        orig_lines = original_content.split('\n')[:10]
        clean_lines = cleaned_content.split('\n')[:10]
        
        return {
            'success': True,
            'original_first_10': orig_lines,
            'cleaned_first_10': clean_lines,
            'cleaning_removed_lines': len(orig_lines) - len(clean_lines),
            'future_import_line_original': next((i+1 for i, line in enumerate(orig_lines) if '__future__' in line), None),
            'future_import_line_cleaned': next((i+1 for i, line in enumerate(clean_lines) if '__future__' in line), None)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@function_tool
def preview_file_structure(file_path: str, lines: int = 20) -> Dict[str, Any]:
    """Preview file structure without reading entire content."""
    try:
        with open(file_path, 'r') as f:
            preview_lines = []
            for i, line in enumerate(f):
                if i >= lines:
                    break
                preview_lines.append(f"{i+1:3d}: {line.rstrip()}")
        
        return {
            'success': True,
            'preview': '\n'.join(preview_lines),
            'total_lines_shown': len(preview_lines)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# EXISTING TOOLS

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
    return [
        # Core tools
        read_code_file, 
        write_code_file,
        
        # Validation tools
        validate_python_syntax,
        check_required_imports,
        assess_generation_quality,
        
        # Backup and versioning
        backup_working_version,
        restore_backup,
        
        # Debugging helpers
        extract_error_details,
        
        # File management
        list_generated_programs,
        cleanup_failed_programs,
        preview_file_structure,
        fix_double_extensions
    ]

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

@function_tool
def validate_planner_output(program_name: str) -> Dict[str, Any]:
    """Validate that planner actually created the expected file."""
    try:
        expected_file = f"../programs/{program_name}.py"
        
        if not os.path.exists(expected_file):
            return {
                'success': False,
                'error': f'Planner failed to create file: {expected_file}',
                'action_required': 'Planner must call write_code_file'
            }
        
        # Check file was created recently (within last 10 minutes)
        file_age = datetime.now().timestamp() - os.path.getmtime(expected_file)
        if file_age > 600:  # 10 minutes
            return {
                'success': False,
                'error': f'File {expected_file} is too old (created {file_age/60:.1f} minutes ago)',
                'action_required': 'Planner must create new implementation'
            }
        
        # Basic syntax check
        with open(expected_file) as f:
            content = f.read()
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            return {
                'success': False,
                'error': f'Generated file has syntax errors: {e}',
                'action_required': 'Planner must fix implementation'
            }
        
        return {
            'success': True,
            'message': f'File {expected_file} created successfully and validated',
            'file_size': len(content),
            'lines': len(content.splitlines())
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
