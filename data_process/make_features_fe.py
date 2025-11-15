import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scipy import sparse
import joblib

# --------------------
# 1. 路径 & 读原始数据
# --------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

target_col = "loan_paid_back"
id_col = "id"

# --------------------
# 2. 构造新的数值特征
# --------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) 估算总负债额（approx yearly debt/payment）
    df["total_debt"] = df["annual_income"] * df["debt_to_income_ratio"]

    # 2) 贷款金额 / 收入比，防止极端值，做个剪裁
    df["loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"].clip(lower=1.0)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].clip(upper=5.0)

    # 3) 信用分 & 利率交互：信用分越低、利率越高 → 数值越大
    risk_score = (700 - df["credit_score"]).clip(lower=0)
    df["interest_x_risk"] = risk_score * df["interest_rate"]

    # 4) 信用分分箱 → 类别特征
    #   <600: bad, 600–659: fair, 660–719: good, >=720: excellent
    bins = [0, 600, 660, 720, 1000]
    labels = ["bad", "fair", "good", "excellent"]
    df["credit_score_bin"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False)

    return df

train_fe = add_engineered_features(train)
test_fe  = add_engineered_features(test)

print("Train with new features:", train_fe.shape)
print("Test with new features:", test_fe.shape)

# --------------------
# 3. 特征列列表（原始 + 新增）
# --------------------
num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    # 新增数值特征：
    "total_debt",
    "loan_to_income_ratio",
    "interest_x_risk",
]

cat_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
    # 新增类别特征：
    "credit_score_bin",
]

feature_cols = num_cols + cat_cols

# --------------------
# 4. 预处理器：数值标准化 + 类别 One-Hot
# --------------------
def build_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )
    return preprocessor

preprocessor = build_preprocessor(num_cols, cat_cols)

# --------------------
# 5. 划分 train / valid
# --------------------
X = train_fe[[id_col] + feature_cols].copy()
y = train_fe[target_col].astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape)

train_ids = X_train[id_col].to_numpy()
valid_ids = X_valid[id_col].to_numpy()
test_ids  = test_fe[id_col].to_numpy()

X_train_feat = X_train[feature_cols]
X_valid_feat = X_valid[feature_cols]
X_test_feat  = test_fe[feature_cols]

# --------------------
# 6. 拟合预处理器并转换
# --------------------
preprocessor.fit(X_train_feat)

X_train_proc = preprocessor.transform(X_train_feat)
X_valid_proc = preprocessor.transform(X_valid_feat)
X_test_proc  = preprocessor.transform(X_test_feat)

print("Processed shapes (with FE):",
      X_train_proc.shape, X_valid_proc.shape, X_test_proc.shape)

# --------------------
# 7. 保存到 processed_fe/ 目录
# --------------------
OUT_DIR = BASE_DIR / "processed_fe"
OUT_DIR.mkdir(exist_ok=True)

sparse.save_npz(OUT_DIR / "X_train_proc_fe.npz", X_train_proc)
sparse.save_npz(OUT_DIR / "X_valid_proc_fe.npz", X_valid_proc)
sparse.save_npz(OUT_DIR / "X_test_proc_fe.npz",  X_test_proc)

np.save(OUT_DIR / "y_train_fe.npy", y_train.to_numpy())
np.save(OUT_DIR / "y_valid_fe.npy", y_valid.to_numpy())

np.save(OUT_DIR / "train_ids_fe.npy", train_ids)
np.save(OUT_DIR / "valid_ids_fe.npy", valid_ids)
np.save(OUT_DIR / "test_ids_fe.npy",  test_ids)

joblib.dump(preprocessor, OUT_DIR / "preprocessor_fe.joblib")

print("All feature-engineered files saved under:", OUT_DIR.resolve())