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
# 1. 路径 & 读数据
# --------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# --------------------
# 2. 列名配置
# --------------------
target_col = "loan_paid_back"
id_col = "id"

num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
]

cat_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]

feature_cols = num_cols + cat_cols

# --------------------
# 3. 构建预处理器
#    数值：标准化
#    类别：One-Hot
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
# 4. 划分训练 / 验证
# --------------------
X = train[[id_col] + feature_cols].copy()
y = train[target_col].astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape)

# 保留 id，后面保存用
train_ids = X_train[id_col].to_numpy()
valid_ids = X_valid[id_col].to_numpy()
test_ids  = test[id_col].to_numpy()

# 仅保留特征列用于预处理
X_train_feat = X_train[feature_cols]
X_valid_feat = X_valid[feature_cols]
X_test_feat  = test[feature_cols]

# --------------------
# 5. 拟合预处理器并转换
# --------------------
preprocessor.fit(X_train_feat)

X_train_proc = preprocessor.transform(X_train_feat)
X_valid_proc = preprocessor.transform(X_valid_feat)
X_test_proc  = preprocessor.transform(X_test_feat)

print("Processed shapes:",
      X_train_proc.shape, X_valid_proc.shape, X_test_proc.shape)

# --------------------
# 6. 保存到 processed/ 目录
# --------------------
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

# 特征矩阵（稀疏 npz）
sparse.save_npz(OUT_DIR / "X_train_proc.npz", X_train_proc)
sparse.save_npz(OUT_DIR / "X_valid_proc.npz", X_valid_proc)
sparse.save_npz(OUT_DIR / "X_test_proc.npz",  X_test_proc)

# 标签
np.save(OUT_DIR / "y_train.npy", y_train.to_numpy())
np.save(OUT_DIR / "y_valid.npy", y_valid.to_numpy())

# id
np.save(OUT_DIR / "train_ids.npy", train_ids)
np.save(OUT_DIR / "valid_ids.npy", valid_ids)
np.save(OUT_DIR / "test_ids.npy",  test_ids)

# 预处理器（可选，但建议保存）
joblib.dump(preprocessor, OUT_DIR / "preprocessor.joblib")

print("All processed files saved under:", OUT_DIR.resolve())