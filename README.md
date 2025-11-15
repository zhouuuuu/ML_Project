# ML_Project

# data processing
## structure
```text
data_process/
  data/                # Kaggle原始数据集
    train.csv          
    test.csv           
    sample_submission.csv
  processed/            # Baseline feature set (full original features)
    X_train_proc.npz
    X_valid_proc.npz
    X_test_proc.npz
    y_train.npy
    y_valid.npy
    train_ids.npy
    valid_ids.npy
    test_ids.npy
    preprocessor.joblib
  processed_fe/         # Baseline + engineered features
    X_train_proc_fe.npz
    X_valid_proc_fe.npz
    X_test_proc_fe.npz
    y_train_fe.npy
    y_valid_fe.npy
    train_ids_fe.npy
    valid_ids_fe.npy
    test_ids_fe.npy
    preprocessor_fe.joblib
  processed_small/      # Small core feature set (ablation)
    X_train_proc_small.npz
    X_valid_proc_small.npz
    X_test_proc_small.npz
    y_train_small.npy
    y_valid_small.npy
    train_ids_small.npy
    valid_ids_small.npy
    test_ids_small.npy
    preprocessor_small.joblib
  data_process.py        # Script: build full original set
  make_features_fe.py    # Script: build feature-engineered set
  make_features_small.py # Script: build small core feature set
  eda.ipynb              # eda处理，里面的结果可用于report
  usage.ipynb            # 用clf = LogisticRegression(max_iter=1000)跑了一下三种数据集，结果在内，可作参考
