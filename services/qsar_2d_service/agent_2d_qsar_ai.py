# agent_2d_qsar_ai.py
# Core logic for 2D QSAR, fully parameterized

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt

def run_qsar_2d(
    descriptor_csv,
    docking_column,
    drop_columns=None,
    model_dict=None,
    n_splits=5,
    random_state=42,
    best_model_filename="BestQSARModel_CV.pkl",
    output_folder="QSAR_Performance"
):
    df = pd.read_csv(descriptor_csv)
    y = df.pop(docking_column)
    if drop_columns is not None:
        X = df.drop(columns=drop_columns)
    else:
        X = df

    # Feature selection using RandomForest
    base_rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    base_rf.fit(X, y)
    importances = base_rf.feature_importances_
    importance_threshold = np.percentile(importances, 40)
    selector = SelectFromModel(base_rf, threshold=importance_threshold, prefit=True)
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support(indices=True)]

    # Model selection and hyperparameter tuning
    results = []
    os.makedirs(output_folder, exist_ok=True)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # If model_dict is None, use default models with grid search
    if model_dict is None:
        model_dict = {}
        # RandomForest
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None]
        }
        rf = RandomForestRegressor(random_state=random_state)
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_selected, y)
        model_dict['RandomForest'] = rf_grid.best_estimator_
        # XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb = XGBRegressor(random_state=random_state, verbosity=0)
        xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=cv, scoring='r2', n_jobs=-1)
        xgb_grid.fit(X, y)
        model_dict['XGBoost'] = xgb_grid.best_estimator_
        # SVR
        svr_param_grid = {
            'svr__C': [0.1, 1, 10],
            'svr__gamma': ['scale', 'auto', 0.01],
            'svr__kernel': ['rbf', 'linear']
        }
        svr = make_pipeline(StandardScaler(), SVR())
        svr_grid = GridSearchCV(svr, svr_param_grid, cv=cv, scoring='r2', n_jobs=-1)
        svr_grid.fit(X, y)
        model_dict['SVR'] = svr_grid.best_estimator_
    else:
        # If user provides model_dict, instantiate models with given params
        for name, model_info in model_dict.items():
            model_class = model_info['class']
            params = model_info.get('params', {})
            if name == 'RandomForest':
                model_dict[name] = RandomForestRegressor(**params)
            elif name == 'XGBoost':
                model_dict[name] = XGBRegressor(**params)
            elif name == 'SVR':
                model_dict[name] = make_pipeline(StandardScaler(), SVR(**params))

    for name, model in model_dict.items():
        if name == 'RandomForest':
            y_pred = cross_val_predict(model, X_selected, y, cv=cv)
        else:
            y_pred = cross_val_predict(model, X, y, cv=cv)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        plt.figure()
        plt.scatter(y, y_pred, color='royalblue', edgecolor='black')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name} - Actual vs Predicted (CV)')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{name}_cv_actual_vs_predicted.png")
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_folder}/model_comparison_cv.csv", index=False)
    best_model_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
    if best_model_name == 'RandomForest':
        best_model = model_dict[best_model_name]
        best_model.fit(X_selected, y)
    else:
        best_model = model_dict[best_model_name]
        best_model.fit(X, y)
    joblib.dump(best_model, best_model_filename)
    return {
        'results': results_df.to_dict(orient='records'),
        'best_model': best_model_name,
        'best_model_file': best_model_filename
    }
