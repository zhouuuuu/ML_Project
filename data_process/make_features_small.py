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
# 2. 构造 loan_to_income_ratio
# --------------------
def add_small_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 贷款金额 / 收入比，避免除 0，并剪裁一些极端值
    df["loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"].clip(lower=1.0)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].clip(upper=5.0)
    return df

train_s = add_small_features(train)
test_s  = add_small_features(test)

print("Train with loan_to_income_ratio:", train_s.shape)
print("Test with loan_to_income_ratio:", test_s.shape)

# --------------------
# 3. 只保留我们想要的特征
# --------------------
num_cols = [
    "debt_to_income_ratio",
    "loan_to_income_ratio",
    "credit_score",
]

cat_cols = [
    "loan_purpose",
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
X = train_s[[id_col] + feature_cols].copy()
y = train_s[target_col].astype(np.float32)

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
test_ids  = test_s[id_col].to_numpy()

X_train_feat = X_train[feature_cols]
X_valid_feat = X_valid[feature_cols]
X_test_feat  = test_s[feature_cols]

# --------------------
# 6. 拟合预处理器并转换
# --------------------
preprocessor.fit(X_train_feat)

X_train_proc = preprocessor.transform(X_train_feat)
X_valid_proc = preprocessor.transform(X_valid_feat)
X_test_proc  = preprocessor.transform(X_test_feat)

X_train_proc = sparse.csr_matrix(X_train_proc)
X_valid_proc = sparse.csr_matrix(X_valid_proc)
X_test_proc  = sparse.csr_matrix(X_test_proc)

print("Processed shapes (small feature set):",
      X_train_proc.shape, X_valid_proc.shape, X_test_proc.shape)

# --------------------
# 7. 保存到 processed_small/ 目录
# --------------------
OUT_DIR = BASE_DIR / "processed_small"
OUT_DIR.mkdir(exist_ok=True)

sparse.save_npz(OUT_DIR / "X_train_proc_small.npz", X_train_proc)
sparse.save_npz(OUT_DIR / "X_valid_proc_small.npz", X_valid_proc)
sparse.save_npz(OUT_DIR / "X_test_proc_small.npz",  X_test_proc)

np.save(OUT_DIR / "y_train_small.npy", y_train.to_numpy())
np.save(OUT_DIR / "y_valid_small.npy", y_valid.to_numpy())

np.save(OUT_DIR / "train_ids_small.npy", train_ids)
np.save(OUT_DIR / "valid_ids_small.npy", valid_ids)
np.save(OUT_DIR / "test_ids_small.npy",  test_ids)

joblib.dump(preprocessor, OUT_DIR / "preprocessor_small.joblib")

print("All small-feature files saved under:", OUT_DIR.resolve())