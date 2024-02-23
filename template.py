import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:  %(message)s:')

project_name = "AuxSearch"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",    
    "app.py",
    "main.py",
    "Dockerfile",
    "research/trials.ipynb"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, file_name = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating {filedir} for {file_name}")

    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath,'w') as f:
            pass
        logging.info(f"creating empty file:  {file_name}")

    else:
        logging.info(f"{file_name}")