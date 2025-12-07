# TAMSUI_train_models.py
# 訓練淡水版 XGBoost 模型（使用既有 CSV，不再產生假資料）

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from config import (
    TAMSUI_ORIGINAL_CSV,
    TAMSUI_GENERATED_DATA_CSV,
    TAMSUI_NON_SUSTAINABLE_ATTR_CSV,
    TAMSUI_NON_SUSTAINABLE_NON_ATTR_CSV,
    TAMSUI_XGB_MODEL1_PATH,
    TAMSUI_XGB_MODEL2_PATH,
    TAMSUI_PHTEST_MODEL_PATH,
    TAMSUI_NON_SUSTAINABLE_MODEL_PATH,
    TAMSUI_NON_SUSTAINABLE_NON_MODEL_PATH,
)


def ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


# ============ 模型 1：tamsui_XGboost_recommend1 ============

def train_tamsui_model1():
    """
    對應 tamsui_XGboost_recommend1：
    特徵：weather, gender, identity（全部 label encode，不做 one-hot）
    資料：TAMSUI_ORIGINAL_CSV
    """
    print("🧪 開始訓練淡水模型 1 (recommend1)...")

    Data = pd.read_csv(TAMSUI_ORIGINAL_CSV, encoding="utf-8-sig")

    df = pd.DataFrame(
        {
            "weather": Data["weather"],
            "gender": Data["gender"],
            "identity": Data["identity"],
            "label": Data["設置點"],
        }
    )

    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_label = LabelEncoder()

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    y = le_label.fit_transform(df["label"].values)

    X = df[["weather_enc", "gender_enc", "identity_enc"]].astype(np.float32).values

    num_classes = len(np.unique(y))
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        eval_metric="mlogloss",
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X, y)
    ensure_dir(TAMSUI_XGB_MODEL1_PATH)
    model.save_model(TAMSUI_XGB_MODEL1_PATH)
    print(f"✅ 淡水模型 1 已儲存到: {TAMSUI_XGB_MODEL1_PATH}")


# ============ 模型 2：tamsui_XGboost_recommend2 ============

def train_tamsui_model2():
    """
    對應 tamsui_XGboost_recommend2：
    特徵：weather, gender, identity, holiday, temperature, weight（再做 one-hot）
    資料：TAMSUI_ORIGINAL_CSV
    """
    print("🧪 開始訓練淡水模型 2 (recommend2)...")

    Data = pd.read_csv(TAMSUI_ORIGINAL_CSV, encoding="utf-8-sig")

    df = pd.DataFrame(
        {
            "weather": Data["weather"],
            "gender": Data["gender"],
            "identity": Data["identity"],
            "holiday": Data["holiday"],
            "temperature": pd.to_numeric(Data["temperature"], errors="coerce").fillna(0),
            "weight": pd.to_numeric(Data["weight"], errors="coerce").fillna(0),
            "label": Data["設置點"],
        }
    )

    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()
    le_label = LabelEncoder()

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(np.float32).values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = ohe.fit_transform(X_num)

    num_classes = len(np.unique(y))
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        eval_metric="mlogloss",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X, y)
    ensure_dir(TAMSUI_XGB_MODEL2_PATH)
    model.save_model(TAMSUI_XGB_MODEL2_PATH)
    print(f"✅ 淡水模型 2 已儲存到: {TAMSUI_XGB_MODEL2_PATH}")


# ============ 模型 3：tamsui_XGboost_recommend3 ============

def train_tamsui_ph_test_model():
    """
    對應 tamsui_XGboost_recommend3：
    特徵：weather, gender, identity, holiday, temperature, weight（one-hot）
    資料：TAMSUI_GENERATED_DATA_CSV
    """
    print("🧪 開始訓練淡水 PHTEST 模型 (recommend3)...")

    Data = pd.read_csv(TAMSUI_GENERATED_DATA_CSV, encoding="utf-8-sig")

    df = pd.DataFrame(
        {
            "weather": Data["weather"],
            "gender": Data["gender"],
            "identity": Data["identity"],
            "holiday": Data["holiday"],
            "temperature": pd.to_numeric(Data["temperature"], errors="coerce").fillna(0),
            "weight": pd.to_numeric(Data["weight"], errors="coerce").fillna(0),
            "label": Data["設置點"],
        }
    )

    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()
    le_label = LabelEncoder()

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(np.float32).values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = ohe.fit_transform(X_num)

    num_classes = len(np.unique(y))
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        eval_metric="mlogloss",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X, y)
    ensure_dir(TAMSUI_PHTEST_MODEL_PATH)
    model.save_model(TAMSUI_PHTEST_MODEL_PATH)
    print(f"✅ 淡水 PHTEST 模型已儲存到: {TAMSUI_PHTEST_MODEL_PATH}")


# ============ 分類模型：一般景點 / 一般餐廳 ============

def _train_classification_model(csv_path: str, model_path: str, desc: str):
    """
    共用：訓練 XGBoost 分類模型（一般景點 / 一般餐廳）
    特徵：weather, gender, identity, holiday, temperature, weight（one-hot）
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ {desc} 的 CSV 檔案不存在，跳過訓練: {csv_path}")
        return

    Data = pd.read_csv(csv_path, encoding="utf-8-sig")
    if Data.empty:
        print(f"⚠️ {desc} 的資料為空 (0 筆)，跳過訓練: {csv_path}")
        return

    print(f"🧪 開始訓練 {desc} 模型 ...")

    df = pd.DataFrame(
        {
            "weather": Data["weather"],
            "gender": Data["gender"],
            "identity": Data["identity"],
            "holiday": Data["holiday"],
            "temperature": pd.to_numeric(Data["temperature"], errors="coerce").fillna(0),
            "weight": pd.to_numeric(Data["weight"], errors="coerce").fillna(0),
            "label": Data["設置點"],
        }
    )

    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()
    le_label = LabelEncoder()

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(np.float32).values

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = ohe.fit_transform(X_num)

    num_classes = len(np.unique(y))
    model = XGBClassifier(
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        eval_metric="mlogloss",
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(X, y)
    ensure_dir(model_path)
    model.save_model(model_path)
    print(f"✅ {desc} 模型已儲存到: {model_path}")


def train_tamsui_classification_models():
    # 一般景點
    _train_classification_model(
        TAMSUI_NON_SUSTAINABLE_ATTR_CSV,
        TAMSUI_NON_SUSTAINABLE_MODEL_PATH,
        "淡水 一般景點",
    )

    # 一般餐廳（目前資料可能為空，會自動跳過）
    _train_classification_model(
        TAMSUI_NON_SUSTAINABLE_NON_ATTR_CSV,
        TAMSUI_NON_SUSTAINABLE_NON_MODEL_PATH,
        "淡水 一般餐廳",
    )


# ============ 主程式：一次訓練全部 ============

if __name__ == "__main__":
    train_tamsui_model1()
    train_tamsui_model2()
    train_tamsui_ph_test_model()
    train_tamsui_classification_models()

    print("🎉 所有淡水 XGBoost 模型訓練完成！（使用真實 CSV 資料）")
