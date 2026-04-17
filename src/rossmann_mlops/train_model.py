import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import yaml
import logging
import os
import matplotlib.pyplot as plt

# -----------------------------
# 1. Cấu hình Logging & Metrics
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rmspe(y_true, y_pred):
    """Tính toán RMSPE (Metric chính của Rossmann)"""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))

# -----------------------------
# 2. Feature Engineering Logic (Khớp với Notebook)
# -----------------------------
def apply_feature_engineering(train_set, val_set):
    logger.info("Đang tính toán các biến trung bình (Target Encoding)...")
    
    # Tính trung bình Sales_log theo Store, Thứ và Promo trên tập TRAIN
    store_dw_promo_avg = train_set.groupby(['Store', 'DayOfWeek', 'Promo'])['Sales_log'].mean().reset_index()
    store_dw_promo_avg.rename(columns={'Sales_log': 'Store_DW_Promo_Avg'}, inplace=True)

    # Tính trung bình Sales_log theo Tháng
    month_avg = train_set.groupby('Month')['Sales_log'].mean().reset_index()
    month_avg.rename(columns={'Sales_log': 'Month_Avg_Sales'}, inplace=True)

    # Merge vào các tập dữ liệu
    train_set = train_set.merge(store_dw_promo_avg, on=['Store', 'DayOfWeek', 'Promo'], how='left')
    train_set = train_set.merge(month_avg, on='Month', how='left')

    val_set = val_set.merge(store_dw_promo_avg, on=['Store', 'DayOfWeek', 'Promo'], how='left')
    val_set = val_set.merge(month_avg, on='Month', how='left')

    # Xử lý NaN bằng global mean của tập Train
    global_mean_train = train_set['Sales_log'].mean()
    val_set['Store_DW_Promo_Avg'] = val_set['Store_DW_Promo_Avg'].fillna(global_mean_train)
    val_set['Month_Avg_Sales'] = val_set['Month_Avg_Sales'].fillna(global_mean_train)

    return train_set, val_set

# -----------------------------
# 3. Luồng huấn luyện chính
# -----------------------------
def main(args):
    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Rossmann_Final_Training")

    # 1. Load data
    logger.info(f"Đang tải dữ liệu từ: {args.data}")
    df = pd.read_csv(args.data)
    
    # 2. Split data theo thời gian (6 tuần cuối 2015)
    val_condition = (df['Year'] == 2015) & (df['WeekOfYear'] >= 26)
    train_df = df[~val_condition].copy()
    val_df = df[val_condition].copy()

    # 3. Feature Engineering
    train_df, val_df = apply_feature_engineering(train_df, val_df)

    # 4. Chuẩn bị X, y (Huấn luyện trên Sales_log)
    drop_cols = ['Sales', 'Sales_log', 'Customers', 'Month', 'Promo2']
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['Sales_log']
    
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val_orig = np.exp(val_df['Sales_log']) # Giá trị gốc để tính RMSPE

    # 5. Khởi tạo Model (Mặc định dùng XGBoost theo best params của bạn)
    # Bạn có thể điều chỉnh params này theo kết quả Optuna/GridSearch
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_estimators=1000,
        max_depth=11,
        learning_rate=0.025,
        random_state=42
    )

    with mlflow.start_run(run_name="Production_Train_Logic"):
        logger.info("🚀 Đang huấn luyện mô hình...")
        model.fit(X_train, y_train)
        
        # 6. Dự đoán và tính toán Metric
        y_train_pred = np.exp(model.predict(X_train))
        y_val_pred = np.exp(model.predict(X_val))

        train_rmspe = rmspe(np.exp(y_train), y_train_pred)
        val_rmspe = rmspe(y_val_orig, y_val_pred)
        rmspe_gap = val_rmspe - train_rmspe

        # 7. Log MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("train_rmspe", train_rmspe)
        mlflow.log_metric("val_rmspe", val_rmspe)
        mlflow.log_metric("rmspe_gap", rmspe_gap)

        # 8. Xuất file model .joblib (Để gửi cho Phúc)
        os.makedirs(args.artifacts_dir, exist_ok=True)
        model_path = os.path.join(args.artifacts_dir, "rossmann_model.joblib")
        joblib.dump(model, model_path)
        
        # Log model lên MLflow
        mlflow.sklearn.log_model(model, "model")

        # 9. Lưu file Config .yaml
        config_path = '../configs/model_config.yaml'
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        model_config = {
            'project': 'Rossmann Store Sales',
            'best_model': {
                'name': 'XGBoost',
                'val_rmspe': float(val_rmspe),
                'params': model.get_params()
            },
            'features': {'input_columns': list(X_train.columns)}
        }
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f)

        logger.info(f"✅ Hoàn tất! Model: {model_path} | Val RMSPE: {val_rmspe:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/processed/train_final.csv")
    parser.add_argument("--artifacts-dir", type=str, default="../artifacts/models")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000")
    
    args = parser.parse_args()
    main(args)