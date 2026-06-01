import duckdb
import networkx as nx
import polars as pl
import pydantic
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))

print("Polars version:", pl.__version__)
print("DuckDB version:", duckdb.__version__)
print("Pydantic version:", pydantic.__version__)
print("NetworkX version:", nx.__version__)
