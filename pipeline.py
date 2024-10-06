import subprocess
import glob
import pathlib

PREPEND = 'notebooks/training/expert/no_preprocess'

# Get the list of notebook files
notebooks = glob.glob(f'{PREPEND}/*.ipynb')

for notebook in notebooks:
    try:
        notebook_path = pathlib.Path(notebook)
        output_folder = notebook_path.parent / 'output'

        # Create the output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Run papermill on each notebook
        result = subprocess.run(
            ['papermill', str(notebook_path.name), f"output/{notebook_path.name}"],
            cwd=notebook_path.parent,
            check=True,
            capture_output=True,
            text=True
        )

        # Print success message
        print(f"Successfully processed: {notebook_path.name}")

    except subprocess.CalledProcessError as e:
        # Improved error handling: provide detailed error output
        print(f"Error executing notebook {notebook_path.name}")
        print(f"Exit code: {e.returncode}")
        print(f"Error message: {e.stderr}")
