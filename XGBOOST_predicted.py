import os
import numpy as np
import pandas as pd
from random import randrange, choice
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

import Now_weather  # 只在 __main__ 測試時使用

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

# ---------------------------
# 共用小工具
# ---------------------------
# ---- 路徑偵錯：看 Python 眼中路徑長怎樣 ----
print("TAMSUI_GENERATED_DATA_CSV =", repr(TAMSUI_GENERATED_DATA_CSV))
print("檔案是否存在？", os.path.exists(TAMSUI_GENERATED_DATA_CSV))

folder = os.path.dirname(TAMSUI_GENERATED_DATA_CSV)
print("資料夾內容 listing:", folder)
try:
    print(os.listdir(folder))
except Exception as e:
    print("列出資料夾失敗：", e)
# ---- 路徑偵錯結束 ----


def safe_onehot_transform(onehotencoder, input_data):
    try:
        final = onehotencoder.transform(input_data)
    except ValueError as e:
        print("❌ OneHotEncoder.transform() 出錯:", e, flush=True)
        print("🚨 輸入數據:", input_data, flush=True)
        raise e
    return final


def safe_label_transform(labelencoder, arr, default_value=0):
    try:
        arr_labelencode = labelencoder.transform(arr)
    except ValueError as e:
        print("❌ LabelEncoder.transform() 出錯:", e, flush=True)
        print("🚨 輸入 array:", arr, flush=True)
        print("⚠️ 使用預設值:", default_value, flush=True)
        arr_labelencode = np.array([default_value])
    return arr_labelencode


def check_and_set_defaults(**kwargs):
    """
    gender: '男', '女', '其他'
    identity: '學生', '非學生'
    holiday: '假日', '非假日'
    temperature: 數值
    weight: 1~5 喜好程度
    """
    defaults = {
        "gender": "其他",
        "identity": "學生",
        "holiday": "非假日",
        "temperature": 22,
        "weight": 3,
    }
    for key, default in defaults.items():
        if key in kwargs and (kwargs[key] is None or kwargs[key] == ""):
            kwargs[key] = default
    return kwargs


def safe_float(val):
    try:
        return float(val)
    except Exception as e:
        print("❌ safe_float: 轉換失敗，使用預設值 0.0，輸入值：", val, "錯誤：", e, flush=True)
        return 0.0


# ---------------------------
# 淡水版 XGBoost 預測函式
# ---------------------------


def tamsui_XGboost_recommend1(weather_arr, gender, identity):
    """
    推薦函式 1：
      特徵：weather, gender, identity
      資料：TAMSUI_ORIGINAL_CSV
      模型：TAMSUI_XGB_MODEL1_PATH
    """
    le_label = LabelEncoder()
    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()

    Data = pd.read_csv(TAMSUI_ORIGINAL_CSV, encoding="utf-8-sig")
    df = pd.DataFrame(
        {
            "weather": Data["weather"],
            "gender": Data["gender"],
            "identity": Data["identity"],
            "label": Data["設置點"],
        }
    )

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    y = le_label.fit_transform(df["label"].values)

    X = df[["weather_enc", "gender_enc", "identity_enc"]].astype(float).values

    weather_enc = safe_label_transform(le_weather, np.array(weather_arr), default_value=0)
    gender_enc = safe_label_transform(le_gender, np.array([gender]), default_value=0)
    identity_enc = safe_label_transform(le_identity, np.array([identity]), default_value=0)

    Value_arr = np.array(
        [safe_float(weather_enc[0]), safe_float(gender_enc[0]), safe_float(identity_enc[0])],
        dtype=float,
    )
    print("🚀 淡水 recommend1 輸入特徵:", Value_arr)

    loaded_model = XGBClassifier()
    loaded_model.load_model(TAMSUI_XGB_MODEL1_PATH)
    predicted = loaded_model.predict(Value_arr.reshape(1, -1))
    result = le_label.inverse_transform(predicted)
    return result[0]


def tamsui_XGboost_recommend2(
    weather_arr, gender, identity, holiday, temperature, weight, dont_go_here
):
    """
    推薦函式 2：
      特徵：weather, gender, identity, holiday, temperature, weight
      資料：TAMSUI_ORIGINAL_CSV（會排除 dont_go_here）
      模型：TAMSUI_XGB_MODEL2_PATH
    """
    params = check_and_set_defaults(
        gender=gender,
        identity=identity,
        holiday=holiday,
        temperature=temperature,
        weight=weight,
    )
    gender = params["gender"]
    identity = params["identity"]
    holiday = params["holiday"]
    temperature = safe_float(params["temperature"])
    weight = safe_float(params["weight"])

    le_label = LabelEncoder()
    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()

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

    # 排除不要去的地點
    df = df[~df["label"].isin(dont_go_here)]

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(float).values

    onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = onehotencoder.fit_transform(X_num)

    weather_enc = safe_label_transform(le_weather, np.array(weather_arr), default_value=0)
    gender_enc = safe_label_transform(le_gender, np.array([gender]), default_value=0)
    identity_enc = safe_label_transform(le_identity, np.array([identity]), default_value=0)
    holiday_enc = safe_label_transform(le_holiday, np.array([holiday]), default_value=0)

    Value_arr = np.array(
        [
            safe_float(weather_enc[0]),
            safe_float(gender_enc[0]),
            safe_float(identity_enc[0]),
            safe_float(holiday_enc[0]),
            temperature,
            weight,
        ],
        dtype=float,
    )

    final = onehotencoder.transform(np.atleast_2d(Value_arr))

    loaded_model = XGBClassifier()
    loaded_model.load_model(TAMSUI_XGB_MODEL2_PATH)
    predicted = loaded_model.predict(final)
    result = le_label.inverse_transform(predicted)
    return result[0]


def tamsui_XGboost_recommend3(weather_arr, gender, identity, holiday, temperature, weight):
    """
    推薦函式 3：
      特徵：weather, gender, identity, holiday, temperature, weight
      資料：TAMSUI_GENERATED_DATA_CSV
      模型：TAMSUI_PHTEST_MODEL_PATH
    """
    params = check_and_set_defaults(
        gender=gender,
        identity=identity,
        holiday=holiday,
        temperature=temperature,
        weight=weight,
    )
    gender = params["gender"]
    identity = params["identity"]
    holiday = params["holiday"]
    temperature = safe_float(params["temperature"])
    weight = safe_float(params["weight"])

    print("Now_weather 回傳值 (淡水): weather =", weather_arr, "temperature =", temperature)

    le_label = LabelEncoder()
    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()

    Data = pd.read_csv(TAMSUI_GENERATED_DATA_CSV, encoding="utf-8-sig")
    print("淡水 GENERATED CSV 檔案內容預覽：")
    print(Data.head())

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

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(float).values

    onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = onehotencoder.fit_transform(X_num)

    weather_enc = safe_label_transform(le_weather, np.array(weather_arr), default_value=0)
    gender_enc = safe_label_transform(le_gender, np.array([gender]), default_value=0)
    identity_enc = safe_label_transform(le_identity, np.array([identity]), default_value=0)
    holiday_enc = safe_label_transform(le_holiday, np.array([holiday]), default_value=0)

    Value_arr = np.array(
        [
            safe_float(weather_enc[0]),
            safe_float(gender_enc[0]),
            safe_float(identity_enc[0]),
            safe_float(holiday_enc[0]),
            temperature,
            weight,
        ],
        dtype=float,
    )
    print("🚀 淡水 recommend3 輸入特徵 (Value_arr):", Value_arr)

    input_data = np.atleast_2d(Value_arr)
    final = safe_onehot_transform(onehotencoder, input_data)

    loaded_model = XGBClassifier()
    loaded_model.load_model(TAMSUI_PHTEST_MODEL_PATH)
    predicted = loaded_model.predict(final)
    result = le_label.inverse_transform(predicted)
    print("✅ 淡水預測結果:", result[0])
    return result[0]


def tamsui_XGboost_classification(
    weather_arr, gender, identity, holiday, temperature, weight, arr_msg
):
    """
    淡水版分類函式：
      arr_msg:
        ['一般景點'] 或 ['一般餐廳']
      特徵：
        weather, gender, identity, holiday, temperature, weight
    """
    params = check_and_set_defaults(
        gender=gender,
        identity=identity,
        holiday=holiday,
        temperature=temperature,
        weight=weight,
    )
    gender = params["gender"]
    identity = params["identity"]
    holiday = params["holiday"]
    temperature = safe_float(params["temperature"])
    weight = safe_float(params["weight"])

    le_label = LabelEncoder()
    le_weather = LabelEncoder()
    le_gender = LabelEncoder()
    le_identity = LabelEncoder()
    le_holiday = LabelEncoder()

    if arr_msg == ["一般景點"]:
        Data = pd.read_csv(TAMSUI_NON_SUSTAINABLE_ATTR_CSV, encoding="utf-8-sig")
        model_path = TAMSUI_NON_SUSTAINABLE_MODEL_PATH
    elif arr_msg == ["一般餐廳"]:
        Data = pd.read_csv(TAMSUI_NON_SUSTAINABLE_NON_ATTR_CSV, encoding="utf-8-sig")
        model_path = TAMSUI_NON_SUSTAINABLE_NON_MODEL_PATH
    else:
        raise ValueError(f"arr_msg 只能是 ['一般景點'] 或 ['一般餐廳']，目前是：{arr_msg}")

    if Data.empty:
        raise ValueError("分類資料為空，請確認對應的 CSV 是否有資料。")

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

    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["identity_enc"] = le_identity.fit_transform(df["identity"])
    df["holiday_enc"] = le_holiday.fit_transform(df["holiday"])
    y = le_label.fit_transform(df["label"].values)

    X_num = df[
        ["weather_enc", "gender_enc", "identity_enc", "holiday_enc", "temperature", "weight"]
    ].astype(float).values

    onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = onehotencoder.fit_transform(X_num)

    weather_enc = safe_label_transform(le_weather, np.array(weather_arr), default_value=0)
    gender_enc = safe_label_transform(le_gender, np.array([gender]), default_value=0)
    identity_enc = safe_label_transform(le_identity, np.array([identity]), default_value=0)
    holiday_enc = safe_label_transform(le_holiday, np.array([holiday]), default_value=0)

    Value_arr = np.array(
        [
            safe_float(weather_enc[0]),
            safe_float(gender_enc[0]),
            safe_float(identity_enc[0]),
            safe_float(holiday_enc[0]),
            temperature,
            weight,
        ],
        dtype=float,
    )

    input_data = np.atleast_2d(Value_arr)
    final = onehotencoder.transform(input_data)

    loaded_model = XGBClassifier()
    loaded_model.load_model(model_path)
    predicted = loaded_model.predict(final)
    result = le_label.inverse_transform(predicted)
    return result[0]


# ---------------------------
# 測試：直接執行本檔案時
# ---------------------------
if __name__ == "__main__":
    weather = Now_weather.weather()
    arr = np.array([weather])

    gender = choice(["男", "女", "其他"])
    identity = choice(["學生", "非學生"])
    holiday = choice(["假日", "非假日"])
    temperature = Now_weather.temperature()
    weight = randrange(1, 6)  # 1~5

    print("淡水輸入參數:", arr, gender, identity, holiday, temperature, weight)

    try:
        print(
            "淡水推薦地點（recommend3）:",
            tamsui_XGboost_recommend3(
                arr, gender, identity, holiday, temperature, weight
            ),
        )
    except Exception as e:
        print("⚠️ 執行 recommend3 時發生錯誤:", e)

