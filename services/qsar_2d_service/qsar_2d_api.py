# qsar_2d_api.py
# FastAPI wrapper for 2D QSAR microservice

from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent_2d_qsar_ai import run_qsar_2d

app = FastAPI()

class QSAR2DRequest(BaseModel):
    descriptor_csv: str
    docking_column: str
    drop_columns: list = None
    model_dict: dict = None
    n_splits: int = 5
    random_state: int = 42
    best_model_filename: str = "BestQSARModel_CV.pkl"
    output_folder: str = "QSAR_Performance"

@app.post("/run-qsar-2d/")
def run_qsar_2d_endpoint(payload: QSAR2DRequest):
    result = run_qsar_2d(**payload.dict())
    return result
