# Create a root-level tools.py that imports from your enhanced tools
# Root-level tools.py - imports from enhanced pipeline.tools
from pipeline.tools.tools import (
    read_code_file, 
    write_code_file, 
    run_training_script,
    run_plot_script,
    get_all_tools
)

# Make the enhanced functions available
from pipeline.tools.tools import (
    validate_python_syntax,
    check_required_imports,
    assess_generation_quality,
    backup_working_version,
    restore_backup,
    extract_error_details,
    list_generated_programs,
    cleanup_failed_programs,
    preview_file_structure,
    fix_double_extensions
)

__all__ = [
    'read_code_file', 'write_code_file', 'run_training_script', 
    'run_plot_script', 'get_all_tools'
]
