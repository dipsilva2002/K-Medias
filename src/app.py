from __future__ import annotations
import os
import argparse
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
FEATURES = ["Latitude", "Longitude", "MedInc"]

KMEANS_MODEL_PATH = "models/kmeans_pipeline.joblib"
SUPERVISED_MODEL_PATH = "models/supervised_model.joblib"
METRICS_PATH = "models/metrics.json"
CLUSTERS_FIG_PATH = "reports/figures/kmeans_clusters_train_test.png"

def ensure_dirs():
    for d in ["data/raw", "data/interim", "data/processed", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)

def load_housing_data(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/housing.csv", index=False)
    return df

def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in df.columns]
    if len(cols) != len(FEATURES):
        missing = set(FEATURES) - set(cols)
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    return df[cols].copy()

def save_processed(df: pd.DataFrame, name: str) -> str:
    os.makedirs("data/processed", exist_ok=True)
    path = os.path.join("data", "processed", name)
    df.to_csv(path, index=False)
    return path

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_kmeans_pipeline(n_clusters: int = 6) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

def assign_clusters(kmeans_pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    return kmeans_pipe.predict(X)

def plot_train_test_clusters(train_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str = CLUSTERS_FIG_PATH) -> None:
    plt.figure()
    plt.scatter(train_df["Longitude"], train_df["Latitude"], c=train_df["cluster"], s=8, alpha=0.6)
    plt.scatter(test_df["Longitude"], test_df["Latitude"], c=test_df["cluster"], s=15, alpha=0.85, marker="x")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("K-Means clusters (train) con puntos de test superpuestos")
    os.makedirs(os.path.dirname(save_path), exist_ok=True
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def build_supervised_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=42)

def evaluate_supervised(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

def run_all(n_clusters: int = 6):
    ensure_dirs()
    print("Cargando dataset...")
    df_raw = load_housing_data()
    print(f"Shape raw: {df_raw.shape}")
    print("Columnas:", list(df_raw.columns)[:10], "...")
    df = keep_relevant_columns(df_raw)
    print("Split train/test (80/20)...")
    train_df, test_df = split_data(df, test_size=0.2, random_state=42)
    print(f"Entrenando K-Means con {n_clusters} clusters...")
    kmeans_pipe = build_kmeans_pipeline(n_clusters=n_clusters)
    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values
    kmeans_pipe.fit(X_train)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["cluster"] = assign_clusters(kmeans_pipe, X_train).astype(int)
    test_df["cluster"] = assign_clusters(kmeans_pipe, X_test).astype(int)
    save_processed(train_df, "housing_train_with_clusters.csv")
    save_processed(test_df, "housing_test_with_clusters.csv")
    plot_train_test_clusters(train_df, test_df, CLUSTERS_FIG_PATH)
    print("Entrenando modelo supervisado...")
    sup_model = build_supervised_model()
    sup_model.fit(X_train, train_df["cluster"].values)
    print("Evaluando modelo supervisado...")
    sup_metrics = evaluate_supervised(sup_model, X_test, test_df["cluster"].values)
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans_pipe, KMEANS_MODEL_PATH)
    joblib.dump(sup_model, SUPERVISED_MODEL_PATH)
    kmeans_obj: KMeans = kmeans_pipe.named_steps["kmeans"]
    all_metrics = {
        "kmeans": {"n_clusters": int(n_clusters), "inertia": float(kmeans_obj.inertia_)},
        "supervised": sup_metrics
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Listo")
    print(KMEANS_MODEL_PATH)
    print(SUPERVISED_MODEL_PATH)
    print(METRICS_PATH)
    print(CLUSTERS_FIG_PATH)
    print("data/processed/housing_train_with_clusters.csv")
    print("data/processed/housing_test_with_clusters.csv")

def predict_cluster(values: List[float]):
    pipe: Pipeline = joblib.load(KMEANS_MODEL_PATH)
    pred = pipe.predict([values])[0]
    print(int(pred))

def predict_cluster_supervised(values: List[float]):
    model = joblib.load(SUPERVISED_MODEL_PATH)
    pred = model.predict([values])[0]
    print(int(pred))

def main():
    parser = argparse.ArgumentParser(description="K-Means + Modelo Supervisado (California Housing)")
    sub = parser.add_subparsers(dest="command")
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--n-clusters", type=int, default=6)
    p_pred_km = sub.add_parser("predict-cluster")
    p_pred_km.add_argument("--values", nargs="+", type=float, required=True)
    p_pred_sup = sub.add_parser("predict-supervised")
    p_pred_sup.add_argument("--values", nargs="+", type=float, required=True)
    args = parser.parse_args()
    if args.command == "run-all" or args.command is None:
        run_all(n_clusters=getattr(args, "n_clusters", 6))
    elif args.command == "predict-cluster":
        if len(args.values) != len(FEATURES):
            raise ValueError(f"Se esperaban {len(FEATURES)} valores: {FEATURES}")
        predict_cluster(args.values)
    elif args.command == "predict-supervised":
        if len(args.values) != len(FEATURES):
            raise ValueError(f"Se esperaban {len(FEATURES)} valores: {FEATURES}")
        predict_cluster_supervised(args.values)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
