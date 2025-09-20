#Importing Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import os

training_flag = True
data_prep_flag = False

def calculate_2d_descriptors_from_mol(molecule_folder, output_csv="2D_QSAR_Features_from_mol.csv"):
    """
    Calculates 2D descriptors for all MOL files in the given folder and saves them to a CSV.
    """
    # Look for files ending in .mol
    mol_files = [f for f in os.listdir(molecule_folder) if f.endswith('.mol')]
    all_descriptors = []

    for file_name in mol_files:
        file_path = os.path.join(molecule_folder, file_name)
        
        # Use MolFromMolFile for single .mol files
        mol = Chem.MolFromMolFile(file_path)

        if mol is None:
            print(f"Warning: Could not load molecule from {file_name}")
            continue

        descriptors = {
            # Replace .mol for the Molecule_ID
            'Molecule_ID': file_name.replace('.mol', ''),
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol)
        }
        all_descriptors.append(descriptors)

    descriptor_df = pd.DataFrame(all_descriptors)
    descriptor_df["Molecule_ID_num"] = descriptor_df["Molecule_ID"].str.extract(r"^(\d+)", expand=False).astype(int)

    descriptor_df = descriptor_df.sort_values(by="Molecule_ID_num", ascending=True).drop(columns=["Molecule_ID_num"])

    descriptor_df.to_csv(output_csv, index=False)

    print(f"✅ Successfully extracted 2D descriptors from .mol files! Saved to {output_csv}")
    print(descriptor_df.head())


def train_2d_qsar_models(
    descriptor_csv,
    docking_column,
    drop_columns=None,
    output_folder="QSAR_Performance",
    model_dict=None,
    n_splits=5,
    random_state=42,
    best_model_filename="BestQSARModel_CV.pkl"
):
    """
    Train and evaluate QSAR regression models with cross-validation.

    Parameters:
        descriptor_csv (str): Path to the descriptor CSV file.
        docking_scores (list or np.array): List of docking scores to add as target.
        name_column (str): Name of the column to drop (non-numeric).
        docking_column (str): Name for the docking score column.
        output_folder (str): Folder to save results and plots.
        model_dict (dict): Dictionary of models to train. If None, uses default models.
        n_splits (int): Number of CV folds.
        random_state (int): Random state for reproducibility.
        best_model_filename (str): Filename to save the best model.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import GridSearchCV

    df = pd.read_csv(descriptor_csv)
    y = df.pop(docking_column)
    if drop_columns is not None:
        X = df.drop(columns=drop_columns)
    else:
        X = df

    print(f"X is: {X.head()}")
    print(f"y is: {y.head()}")

    # 1. Feature selection using RandomForest feature importances
    base_rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    base_rf.fit(X, y)
    importances = base_rf.feature_importances_
    importance_threshold = np.percentile(importances, 40)  # Keep top 60% features
    selector = SelectFromModel(base_rf, threshold=importance_threshold, prefit=True)
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support(indices=True)]
    print(f"Selected features: {list(selected_features)}")


    # 2. Hyperparameter tuning for RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    rf = RandomForestRegressor(random_state=random_state)
    rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                                  scoring='r2', n_jobs=-1, verbose=1)
    rf_grid_search.fit(X_selected, y)
    best_rf = rf_grid_search.best_estimator_
    print(f"Best RandomForest params: {rf_grid_search.best_params_}")

    # 3. Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=random_state, verbosity=0)
    xgb_grid_search = GridSearchCV(xgb, xgb_param_grid, cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                                   scoring='r2', n_jobs=-1, verbose=1)
    xgb_grid_search.fit(X, y)
    best_xgb = xgb_grid_search.best_estimator_
    print(f"Best XGBoost params: {xgb_grid_search.best_params_}")

    # 4. Hyperparameter tuning for SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    svr_param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'svr__kernel': ['rbf', 'linear']
    }
    svr = make_pipeline(StandardScaler(), SVR())
    svr_grid_search = GridSearchCV(svr, svr_param_grid, cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                                   scoring='r2', n_jobs=-1, verbose=1)
    svr_grid_search.fit(X, y)
    best_svr = svr_grid_search.best_estimator_
    print(f"Best SVR params: {svr_grid_search.best_params_}")

    # Update model_dict to use tuned models
    if model_dict is None:
        model_dict = {
            "RandomForest": best_rf,
            "XGBoost": best_xgb,
            "SVR": best_svr
        }
    else:
        model_dict["RandomForest"] = best_rf
        model_dict["XGBoost"] = best_xgb
        model_dict["SVR"] = best_svr

    results = []
    os.makedirs(output_folder, exist_ok=True)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, model in model_dict.items():
        if name == "RandomForest":
            # Use selected features for RandomForest
            y_pred = cross_val_predict(model, X_selected, y, cv=cv)
        else:
            y_pred = cross_val_predict(model, X, y, cv=cv)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})

        plt.figure()
        plt.scatter(y, y_pred, color='royalblue', edgecolor='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel("Actual Docking Score")
        plt.ylabel("Predicted Docking Score")
        plt.title(f"{name} - Actual vs Predicted (CV)")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{name}_cv_actual_vs_predicted.png")
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_folder}/model_comparison_cv.csv", index=False)

    best_model_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
    if best_model_name == "RandomForest":
        best_model = model_dict[best_model_name]
        best_model.fit(X_selected, y)
    else:
        best_model = model_dict[best_model_name]
        best_model.fit(X, y)
    joblib.dump(best_model, best_model_filename)

    print("\n✅ Cross-validation completed.")
    print(f"🏆 Best model based on R²: {best_model_name}")


if __name__ == "__main__":
    if data_prep_flag == True:
        print("Starting data preparation...")
        calculate_2d_descriptors_from_mol(molecule_folder= r"C:\Users\ABC\Projects\A0005_Vinnu_Phd_Papers_Miscellaneous\A0001_QSAR\Quinazoline _3D_ener_min")

    if training_flag == True:
        print("Starting model training...")
        train_2d_qsar_models(
            descriptor_csv=r"C:\Users\ABC\Projects\A0005_Vinnu_Phd_Papers_Miscellaneous\A0001_QSAR\2D_QSAR_Quinazoline_train_test_data_cleaned.csv",
            drop_columns=["Molecule_ID","IC50nm"],
            docking_column="pIC50"
        )