import os
from pathlib import Path

def setup_project_structure():
    # Define the directory structure
    # Folders are keys, lists are files within them
    structure = {
        ".": [
            "dashboard.py",
            "forensic_analyzer.py",
            "groq_ai.py",
            "pipeline_wrapper.py",
            "final.py",
            "run_new_test.py",
            "generate_report.py"
        ],
        "templates": ["index.html"],
        "static": ["style.css"],
        "outputs": []  # Empty directory
    }

    for folder, files in structure.items():
        # Create the directory if it doesn't exist
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = Path(folder) / file
            
            # Create the file if it doesn't exist to avoid overwriting
            if not file_path.exists():
                file_path.touch()
                print(f"Created: {file_path}")
            else:
                print(f"Skipped (already exists): {file_path}")

if __name__ == "__main__":
    setup_project_structure()
    print("\nProject structure initialized successfully.")