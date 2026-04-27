from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rossmann_mlops.processing as processing_module
from rossmann_mlops.processing import run_pipeline as run_preprocessing
from rossmann_mlops.train_model import train_pipeline


def test_train_pipeline_smoke(tmp_path: Path, monkeypatch) -> None:
    # --- Setup raw data ---
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    train_df = pd.DataFrame({
    "Store": [1, 1, 1, 2, 2, 2, 1, 1, 2, 2],
    "DayOfWeek": [1, 2, 3, 1, 2, 3, 1, 2, 1, 2],
    "Date": [
        "2015-05-20", "2015-05-21", "2015-06-20",  # train split
        "2015-05-20", "2015-05-21", "2015-06-20",  # train split
        "2015-07-01", "2015-07-02",                 # val split (week >= 26)
        "2015-07-01", "2015-07-02",                 # val split (week >= 26)
    ],
    "Open": [1] * 10,
    "Promo": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "StateHoliday": ["0"] * 10,
    "SchoolHoliday": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Sales": [5000, 5200, 5500, 3000, 3200, 3600, 5100, 5300, 3100, 3300],
})
    store_df = pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "a"],
        "CompetitionDistance": [100.0, 200.0],
        "Promo2": [0, 1],
        "Promo2SinceWeek": [0, 1],
        "Promo2SinceYear": [0, 2013],
        "CompetitionOpenSinceMonth": [0, 3],
        "CompetitionOpenSinceYear": [0, 2010],
        "PromoInterval": ["", "Jan,Apr,Jul,Oct"],
    })

    train_df.to_csv(raw_dir / "train.csv", index=False)
    store_df.to_csv(raw_dir / "store.csv", index=False)
    train_df.drop(columns=["Sales"]).to_csv(raw_dir / "test.csv", index=False)

    # --- Patch DEFAULT_PATHS trỏ vào tmp_path ---
    monkeypatch.setattr(processing_module, "DEFAULT_PATHS", {
        "store_raw":   str(raw_dir / "store.csv"),
        "train_raw":   str(raw_dir / "train.csv"),
        "test_raw":    str(raw_dir / "test.csv"),
        "train_final": str(processed_dir / "train_final.csv"),
        "val_final":   str(processed_dir / "val_final.csv"),
        "test_final":  str(processed_dir / "test_final.csv"),
    })

    # --- Chạy preprocessing ---
    run_preprocessing()

    # Verify processed files được tạo
    assert (processed_dir / "train_final.csv").exists(), "train_final.csv not created"
    assert (processed_dir / "val_final.csv").exists(),   "val_final.csv not created"

    # --- Config cho train_pipeline ---
    model_path                  = tmp_path / "models" / "model.joblib"
    metrics_path                = tmp_path / "metrics" / "metrics.json"
    official_model_config_path  = tmp_path / "configs" / "model_config.yaml"
    candidate_model_config_path = tmp_path / "models" / "model_config_candidate.yaml"

    config = {
        "paths": {
            "train_final_data":            str(processed_dir / "train_final.csv"),
            "val_final_data":              str(processed_dir / "val_final.csv"),
            "model_file":                  str(model_path),
            "metrics_file":                str(metrics_path),
            "model_config_file":           str(official_model_config_path),
            "model_config_candidate_file": str(candidate_model_config_path),
        },
        "training": {
            "production_train": False,
            "validation_start_date": "2015-06-01",
            "n_estimators": 10,
            "random_state": 42,
            "n_jobs": 1,
        },
    }

    # --- Chạy train ---
    result = train_pipeline(config)

    assert model_path.exists()
    assert metrics_path.exists()
    assert candidate_model_config_path.exists()
    assert not official_model_config_path.exists()
    assert result["model_config_overwritten"] is False
    assert result["metrics"]["rmse"] >= 0

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "mae" in saved_metrics
    assert "r2" in saved_metrics