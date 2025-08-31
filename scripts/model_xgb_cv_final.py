import os
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    # Progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None):
        # Fallback no-op tqdm
        return iterable if iterable is not None else range(total or 0)

# -----------------------------
# Utils
# -----------------------------

def read_events() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    sub_path = "data/raw/sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)
    return train, test, sub


def build_session_labels(train: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: user_id, user_session, session_value (target), event_time or similar optional
    cols = ["user_id", "user_session", "session_value"]
    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in train for labels: {missing}")
    labels = (
        train[["user_id", "user_session", "session_value"]]
        .drop_duplicates(subset=["user_id", "user_session"])  # session-level target
        .reset_index(drop=True)
    )
    return labels


def _load_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prefer v2 session features
    f2_tr = "data/processed/session_features_v2_train.csv"
    f2_te = "data/processed/session_features_v2_test.csv"
    f1_tr = "data/processed/session_features_v1_train.csv"
    f1_te = "data/processed/session_features_v1_test.csv"

    if os.path.exists(f2_tr) and os.path.exists(f2_te):
        feat_tr = pd.read_csv(f2_tr)
        feat_te = pd.read_csv(f2_te)
    elif os.path.exists(f1_tr) and os.path.exists(f1_te):
        feat_tr = pd.read_csv(f1_tr)
        feat_te = pd.read_csv(f1_te)
    else:
        raise FileNotFoundError("No processed features found (v2 or v1). Run feature scripts first.")

    # Required keys
    for df, name in [(feat_tr, "train features"), (feat_te, "test features")]:
        for c in ("user_id", "user_session"):
            if c not in df.columns:
                raise ValueError(f"Missing {c} in {name}")

    return feat_tr, feat_te


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["event_time", "timestamp", "time"]:
        if c in df.columns:
            return c
    return None


def last_items_per_session(events: pd.DataFrame) -> pd.DataFrame:
    time_col = _detect_time_col(events)
    cols = ["user_id", "user_session", "product_id", "category_id"]
    have = [c for c in cols if c in events.columns]
    if time_col and set(["user_id", "user_session"]).issubset(events.columns):
        events_sorted = events.sort_values(["user_id", "user_session", time_col])
    else:
        events_sorted = events

    last_rows = (
        events_sorted.groupby(["user_id", "user_session"], as_index=False)
        .tail(1)[[c for c in ["user_id", "user_session", "product_id", "category_id"] if c in events.columns]]
    )
    # Rename to last_*
    rename_map = {}
    if "product_id" in last_rows.columns:
        rename_map["product_id"] = "last_product_id"
    if "category_id" in last_rows.columns:
        rename_map["category_id"] = "last_category_id"
    last_rows = last_rows.rename(columns=rename_map)
    return last_rows


def add_user_backoff(train_df: pd.DataFrame, target_col: str) -> pd.Series:
    # Mean target per user_id on train fold
    user_means = train_df.groupby("user_id")[target_col].mean()
    return user_means


def xgb_params_base(seed: int, try_gpu: bool = True) -> dict:
    params = dict(
        n_estimators=30000,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.0,
        reg_lambda=1.2,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=os.cpu_count() or 8,
        eval_metric="rmse",
    )
    if try_gpu:
        params.update(dict(tree_method="gpu_hist", predictor="gpu_predictor"))
    else:
        params.update(dict(tree_method="hist"))
    return params


# -----------------------------
# Training
# -----------------------------

def run_cv_ensemble(seeds=(2025, 2027, 2031), n_splits=10, early_stopping=300) -> str:
    from xgboost import XGBRegressor

    t0 = time.time()
    train_ev, test_ev, sub = read_events()
    labels = build_session_labels(train_ev)

    feat_tr, feat_te = _load_features()

    # Last items for TE
    last_tr = last_items_per_session(train_ev)
    last_te = last_items_per_session(test_ev)

    # Merge labels + features + last ids
    df_tr = labels.merge(feat_tr, on=["user_id", "user_session"], how="left")
    df_tr = df_tr.merge(last_tr, on=["user_id", "user_session"], how="left")
    df_te = feat_te.merge(last_te, on=["user_id", "user_session"], how="left")

    # Keys
    key_cols = ["user_id", "user_session"]
    target = "session_value"

    # Fill missing last ids
    for c in ["last_product_id", "last_category_id"]:
        if c in df_tr.columns:
            df_tr[c] = df_tr[c].fillna(-1)
        if c in df_te.columns:
            df_te[c] = df_te[c].fillna(-1)

    # Target transform (+ optional winsorization before log1p)
    y_raw = df_tr[target].astype(float).values
    win_p = float(os.getenv("WINSORIZE_P", "0"))
    if 0.0 < win_p < 1.0:
        lo = np.nanpercentile(y_raw, (1.0 - win_p) * 100.0)
        hi = np.nanpercentile(y_raw, win_p * 100.0)
        y_raw = np.clip(y_raw, lo, hi)
        print(f"Winsorized target at p={win_p}: lo={lo:.4f}, hi={hi:.4f}")
    df_tr["y"] = np.log1p(y_raw)

    # Feature columns
    drop_cols = set(key_cols + [target, "y"])  # keep last_* for TE reference but not as raw unless helpful
    # We'll add TE numerics explicitly; we can keep last_* raw as well for model to learn id signal (high-cardinality trees handle it poorly), better to drop raw ids
    for c in ["last_product_id", "last_category_id"]:
        drop_cols.add(c)
    feature_cols = [c for c in df_tr.columns if c not in drop_cols]

    # Prepare containers
    oof = np.zeros(len(df_tr), dtype=float)
    test_preds_acc = np.zeros(len(df_te), dtype=float)

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error

    groups = df_tr["user_id"].values

    for si, seed in enumerate(seeds, 1):
        try_gpu = True
        params = xgb_params_base(seed, try_gpu=try_gpu)
        gkf = GroupKFold(n_splits=n_splits)
        fold = 0
        pbar = tqdm(total=n_splits, desc=f"Seed {seed}")
        for tr_idx, va_idx in gkf.split(df_tr, groups=groups, y=df_tr["y"].values):
            fold += 1

            tr_df = df_tr.iloc[tr_idx].copy()
            va_df = df_tr.iloc[va_idx].copy()

            # User backoff (on train fold only)
            user_mean_y = add_user_backoff(tr_df, target_col="y")
            tr_df["user_backoff_y"] = tr_df["user_id"].map(user_mean_y)
            va_df["user_backoff_y"] = va_df["user_id"].map(user_mean_y)
            te_df = df_te.copy()
            te_df["user_backoff_y"] = te_df["user_id"].map(user_mean_y)
            global_y_mean = tr_df["y"].mean()
            for d in (tr_df, va_df, te_df):
                d["user_backoff_y"] = d["user_backoff_y"].fillna(global_y_mean)

            # Target encoding for last ids (on log target)
            te_cols = []
            if "last_product_id" in tr_df.columns:
                prod_mean = tr_df.groupby("last_product_id")["y"].mean()
                tr_df["te_last_prod_y"] = tr_df["last_product_id"].map(prod_mean).fillna(global_y_mean)
                va_df["te_last_prod_y"] = va_df["last_product_id"].map(prod_mean).fillna(global_y_mean)
                te_df["te_last_prod_y"] = te_df["last_product_id"].map(prod_mean).fillna(global_y_mean)
                te_cols.append("te_last_prod_y")
            if "last_category_id" in tr_df.columns:
                cat_mean = tr_df.groupby("last_category_id")["y"].mean()
                tr_df["te_last_cat_y"] = tr_df["last_category_id"].map(cat_mean).fillna(global_y_mean)
                va_df["te_last_cat_y"] = va_df["last_category_id"].map(cat_mean).fillna(global_y_mean)
                te_df["te_last_cat_y"] = te_df["last_category_id"].map(cat_mean).fillna(global_y_mean)
                te_cols.append("te_last_cat_y")

            # Assemble features (+ backoff + TE)
            fold_features = feature_cols + ["user_backoff_y"] + te_cols

            # Ensure numeric-only matrices
            num_cols = [c for c in fold_features if np.issubdtype(tr_df[c].dtype, np.number)]
            if len(num_cols) != len(fold_features):
                missing = set(fold_features) - set(num_cols)
                if missing:
                    print(f"Dropping non-numeric columns from features: {sorted(list(missing))}")
            X_tr = tr_df[num_cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_tr = tr_df["y"].to_numpy(dtype=np.float32)
            X_va = va_df[num_cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_va = va_df["y"].to_numpy(dtype=np.float32)
            X_te = te_df[num_cols].fillna(0.0).to_numpy(dtype=np.float32)

            model = XGBRegressor(**params)

            # Try early stopping with eval_set; fallback if not supported
            used_es = True
            try:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    early_stopping_rounds=early_stopping,
                )
            except TypeError:
                # Older API signature
                used_es = False
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except Exception as e:
                # Possibly GPU not available; fallback to CPU and retry once
                if params.get("tree_method") == "gpu_hist":
                    params = xgb_params_base(seed, try_gpu=False)
                    model = XGBRegressor(**params)
                    try:
                        model.fit(
                            X_tr,
                            y_tr,
                            eval_set=[(X_va, y_va)],
                            verbose=False,
                            early_stopping_rounds=early_stopping,
                        )
                        used_es = True
                    except TypeError:
                        used_es = False
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                else:
                    raise e

            va_pred = model.predict(X_va)
            te_pred = model.predict(X_te)

            oof[va_idx] += va_pred
            test_preds_acc += te_pred

            rmse = mean_squared_error(y_va, va_pred, squared=False)
            print(f"Seed {seed} Fold {fold}/{n_splits} RMSE={rmse:.6f} ES={'Y' if used_es else 'N'} Trees={getattr(model, 'best_iteration', params['n_estimators'])}")
            try:
                pbar.update(1)
            except Exception:
                pass
        try:
            pbar.close()
        except Exception:
            pass

    # Average over seeds and folds
    S = len(seeds)
    F = n_splits
    # Each sample appears exactly once per seed in GroupKFold => divide by S
    oof = oof / float(S)

    test_preds = test_preds_acc / float(S * F)

    oof_rmse = mean_squared_error(df_tr["y"].values, oof, squared=False)
    print(f"OOF RMSE (log space): {oof_rmse:.6f}")

    # Inverse transform predictions
    test_pred_linear = np.expm1(test_preds)

    # Build submission in sample order
    sub_out = sub.copy()
    # Map by (user_id, user_session) -> prediction
    te_key = df_te[["user_id", "user_session"]].copy()
    te_key["pred"] = test_pred_linear

    merged = sub_out.merge(te_key, on=["user_id", "user_session"], how="left")
    # Fill NaNs with global mean of train target
    global_mean = df_tr[target].mean()
    merged["session_value"] = merged["pred"].fillna(global_mean)
    merged = merged[["user_id", "user_session", "session_value"]]

    os.makedirs("submissions", exist_ok=True)
    suffix = ""
    if 0.0 < win_p < 1.0:
        suffix = f"_win{win_p}"
    out_path = f"submissions/xgb_cv10_seeds3_v2_log_backoff_te{suffix}.csv"
    merged.to_csv(out_path, index=False)

    # Report
    print({"oof_rmse_log": oof_rmse, "submission": out_path, "n_rows": len(merged)})
    return out_path


if __name__ == "__main__":
    out = run_cv_ensemble()
    print("Wrote:", out)
