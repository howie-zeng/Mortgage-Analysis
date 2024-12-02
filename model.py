import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ml_model:
    def __init__(self, model_name, prob_threshold=0.5, random_state=42):
        self.model_name = model_name
        self.prob_threshold = prob_threshold
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def _objective_rf(self, trial, X_train, y_train, X_test, y_test):
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 50, 600),
            'max_depth': trial.suggest_int("max_depth", 10, 50),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 10, 50),
            # 'min_weight_fraction_leaf': trial.suggest_float("min_weight_fraction_leaf", 0.0001, 0.05, log=True),
            'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 50, 400),
            'max_samples': trial.suggest_float("max_samples", 0.8, 1.0),
            'max_features': trial.suggest_float("max_features", 0.2, 0.4, log=True),
        }
        model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    def _objective_xgbt(self, trial, X_train, y_train, X_test, y_test):
        params = {
            'eta': trial.suggest_float('eta', 1e-2, 0.15, log=True),
            'gamma': trial.suggest_float('gamma', 1e-1, 1, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 50, 600),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
            'sampling_method': trial.suggest_categorical('sampling_method', ['uniform']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.8),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 0.8),
            'max_depth': trial.suggest_int('max_depth', 6, 25),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
            'lambda': trial.suggest_float('lambda', 5e-3, 2, log=True),
            'alpha': trial.suggest_float('alpha', 1e-2, 0.5, log=True),
            'tree_method': trial.suggest_categorical('tree_method', ['hist']),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise']),
        }
        model = xgb.XGBClassifier(**params, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)
    
    def _objective_logistics(self, trial, X_train, y_train, X_test, y_test):
        params = {
            'penalty': trial.suggest_categorical("penalty", ['l2']),
            'C': trial.suggest_float("C", 1e-5, 10, log=True),
            'solver': trial.suggest_categorical("solver", ['newton-cholesky'])
        }
        model = LogisticRegression(**params, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)

    def tune(self, X_train, y_train, X_test, y_test, n_trials=100):
        if self.model_name == "rf":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective_rf(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
        elif self.model_name == "xgbt":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective_xgbt(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
        elif self.model_name == 'logistics':
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective_logistics(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
        else: 
            raise ValueError("Model not supported")

        
        self.best_params = study.best_params
        print("Best parameters:", self.best_params)
        self.study = study
        return self.best_params

    def train(self, X_train, y_train):
        if self.model_name == "rf":
            self.model = RandomForestClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
        elif self.model_name == "xgbt":
            self.model = xgb.XGBClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
        elif self.model_name == 'logistics':
            self.model = LogisticRegression(**self.best_params, random_state=self.random_state, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        return self

    # def etestuate(self, X_test, y_test, verbose=True):
    #     y_pred_proba = self.model.predict_proba(X_test)[:, 1]
    #     y_pred = (y_pred_proba > self.prob_threshold).astype(int)

    #     # Calculate etestuation metrics
    #     accuracy = accuracy_score(y_test, y_pred)
    #     roc_auc = roc_auc_score(y_test, y_pred_proba)
    #     precision = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)

    #     if verbose:
    #         print(f"Accuracy: {accuracy:.4f}")
    #         print(f"ROC AUC: {roc_auc:.4f}")
    #         print(f"Precision: {precision:.4f}")
    #         print(f"Recall: {recall:.4f}")
    #         print(f"F1 Score: {f1:.4f}")

    #     return {"accuracy": accuracy, "roc_auc": roc_auc, "precision": precision, "recall": recall, "f1": f1}
