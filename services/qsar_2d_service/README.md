# QSAR 2D Service

This microservice provides a fully parameterized API for running 2D QSAR model training and evaluation.

## Example API Payload

Use the following JSON payload when calling the `/run-qsar-2d/` endpoint:

```json
{
  "descriptor_csv": "path/to/your/descriptors.csv",
  "docking_column": "pIC50",
  "drop_columns": ["Molecule_ID", "IC50nm"],
  "model_dict": {
    "RandomForest": {
      "class": "RandomForestRegressor",
      "params": {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42
      }
    },
    "XGBoost": {
      "class": "XGBRegressor",
      "params": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
      }
    },
    "SVR": {
      "class": "SVR",
      "params": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
      }
    }
  },
  "n_splits": 5,
  "random_state": 42,
  "best_model_filename": "BestQSARModel_CV.pkl",
  "output_folder": "QSAR_Performance"
}
```

Adjust the file paths, columns, models, and parameters as needed for your use case.
