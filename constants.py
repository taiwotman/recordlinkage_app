from pathlib import Path
import os

# UPLOAD_DIR = os.path.join(f"{Path(__file__).parents[0]}","frontend_rcl/upload_path")
UPLOAD_DIR = "./upload_path"
SIM_TABLE_PATH = os.path.join(UPLOAD_DIR, "similarity_table.parquet")
