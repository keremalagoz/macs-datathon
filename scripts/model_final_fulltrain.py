import os
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(total or 0)

# -----------------------------
# Utils
# -----------------------------

def read_events() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv("data/raw/sample_submission.csv")
    return train, test, sub


def build_session_labels(train: pd.DataFrame) -> pd.DataFrame:
    cols = ["user_id", "user_session", "session_value"]
    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return train[cols].drop_duplicates(["user_id", "user_session"]).reset_index(drop=True)


def _load_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    f2_tr = "data/processed/session_features_v2_train.csv"
    f2_te = "data/processed/session_features_v2_test.csv"
    f1_tr = "data/processed/session_features_v1_train.csv"
    f1_te = "data/processed/session_features_v1_test.csv"
    if os.path.exists(f2_tr) and os.path.exists(f2_te):
        return pd.read_csv(f2_tr), pd.read_csv(f2_te)
    if os.path.exists(f1_tr) and os.path.exists(f1_te):
        return pd.read_csv(f1_tr), pd.read_csv(f1_te)
    raise FileNotFoundError("No processed features found (v2 or v1)")


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["event_time", "timestamp", "time"]:
        if c in df.columns:
            return c
    return None


def last_items_per_session(events: pd.DataFrame) -> pd.DataFrame:
    time_col = _detect_time_col(events)
    if time_col and set(["user_id", "user_session"]).issubset(events.columns):
        events = events.sort_values(["user_id", "user_session", time_col])
    last = (
        events.groupby(["user_id", "user_session"], as_index=False)
        .tail(1)[[c for c in ["user_id", "user_session", "product_id", "category_id"] if c in events.columns]]
    )
    rename = {}
    if "product_id" in last.columns:
        rename["product_id"] = "last_product_id"
    if "category_id" in last.columns:
        rename["category_id"] = "last_category_id"
    return last.rename(columns=rename)


def numeric_only(df: pd.DataFrame, cols) -> pd.DataFrame:
    out_cols = []
    for c in cols:
        if c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.uint8)
            if np.issubdtype(df[c].dtype, np.number):
                out_cols.append(c)
    if len(out_cols) != len(cols):
        missing = [c for c in cols if c not in out_cols]
        if missing:
            print(f"Dropping non-numeric: {missing}")
    return df[out_cols]


# -----------------------------
# Final full-train
# -----------------------------

def run_fulltrain() -> str:
    t0 = time.time()
    print("[1/6] Reading data...", flush=True)
    train_ev, test_ev, sub = read_events()
    labels = build_session_labels(train_ev)

    print("[2/6] Loading features...", flush=True)
    feat_tr, feat_te = _load_features()

    print("[3/6] Building last-item features...", flush=True)
    last_tr = last_items_per_session(train_ev)
    last_te = last_items_per_session(test_ev)

    print("[4/6] Merging...", flush=True)
    df_tr = labels.merge(feat_tr, on=["user_id", "user_session"], how="left")
    df_tr = df_tr.merge(last_tr, on=["user_id", "user_session"], how="left")
    df_te = feat_te.merge(last_te, on=["user_id", "user_session"], how="left")

    # Fill ids
    for c in ["last_product_id", "last_category_id"]:
        if c in df_tr.columns:
            df_tr[c] = df_tr[c].fillna(-1)
        if c in df_te.columns:
            df_te[c] = df_te[c].fillna(-1)

    # Target transform
    y = df_tr["session_value"].astype(float).values
    win_p = float(os.getenv("WINSORIZE_P", "0"))
    if 0.0 < win_p < 1.0:
        lo = np.nanpercentile(y, (1.0 - win_p) * 100.0)
        hi = np.nanpercentile(y, win_p * 100.0)
        y = np.clip(y, lo, hi)
        print(f"Winsorized y at p={win_p}: lo={lo:.4f}, hi={hi:.4f}")
    y_log = np.log1p(y)

    # Global user backoff and TE on full train (log target)
    user_mean = df_tr.groupby("user_id")["session_value"].mean()
    user_mean_log = df_tr.groupby("user_id")[y_log].mean() if False else df_tr.groupby("user_id").apply(lambda d: np.log1p(d["session_value"].astype(float))).groupby(level=0).mean()
    df_tr["user_backoff_y"] = df_tr["user_id"].map(user_mean_log)
    df_te["user_backoff_y"] = df_te["user_id"].map(user_mean_log).fillna(y_log.mean())

    te_cols = []
    global_ylog = y_log.mean()
    if "last_product_id" in df_tr.columns:
        pm = pd.Series(y_log, index=df_tr.index).groupby(df_tr["last_product_id"]).mean()
        df_tr["te_last_prod_y"] = df_tr["last_product_id"].map(pm).fillna(global_ylog)
        df_te["te_last_prod_y"] = df_te["last_product_id"].map(pm).fillna(global_ylog)
        te_cols.append("te_last_prod_y")
    if "last_category_id" in df_tr.columns:
        cm = pd.Series(y_log, index=df_tr.index).groupby(df_tr["last_category_id"]).mean()
        df_tr["te_last_cat_y"] = df_tr["last_category_id"].map(cm).fillna(global_ylog)
        df_te["te_last_cat_y"] = df_te["last_category_id"].map(cm).fillna(global_ylog)
        te_cols.append("te_last_cat_y")

    # Build features
    drop = set(["user_id", "user_session", "session_value", "last_product_id", "last_category_id"])  # drop raw ids
    base_features = [c for c in df_tr.columns if c not in drop]
    base_features += ["user_backoff_y"] + te_cols
    # Deduplicate
    base_features = list(dict.fromkeys(base_features))

    X_tr_df = numeric_only(df_tr, base_features).fillna(0.0)
    X_te_df = numeric_only(df_te, base_features).fillna(0.0)
    X_tr = X_tr_df.to_numpy(dtype=np.float32)
    X_te = X_te_df.to_numpy(dtype=np.float32)

    print(f"[5/6] Training model on {X_tr.shape[0]} rows, {X_tr.shape[1]} features...", flush=True)

    pred_log = None
    err = None

    # Try XGBoost native train with CPU hist by default; GPU if TRY_GPU=1 and available
    try_gpu = os.getenv("TRY_GPU", "0").strip() not in ("0","false","False","no","NO")

    try:
        import xgboost as xgb
        # Params
        n_estimators = int(os.getenv("N_ESTIMATORS", "30000"))
        lr = float(os.getenv("LEARNING_RATE", "0.03"))
        params = {
            'objective': 'reg:squarederror',
            'eta': lr,
            'max_depth': int(os.getenv("MAX_DEPTH", "9")),
            'min_child_weight': int(os.getenv("MIN_CHILD_WEIGHT", "6")),
            'subsample': float(os.getenv("SUBSAMPLE", "0.8")),
            'colsample_bytree': float(os.getenv("COLSAMPLE_BYTREE", "0.7")),
            'gamma': float(os.getenv("GAMMA", "0.0")),
            'lambda': float(os.getenv("REG_LAMBDA", "1.2")),
            'alpha': float(os.getenv("REG_ALPHA", "0.0")),
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist' if try_gpu else 'hist',
            'predictor': 'gpu_predictor' if try_gpu else 'auto',
            'nthread': os.cpu_count() or 8,
            'verbosity': 1,
        }
        # Build small validation via GroupShuffle by user
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=2025)
        groups = df_tr['user_id'].values
        tr_idx, va_idx = next(gss.split(X_tr, y_log, groups))
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y_log[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y_log[va_idx])
        dte = xgb.DMatrix(X_te)

        # Train with early stopping
        bst = xgb.train(params, dtr, num_boost_round=n_estimators,
                        evals=[(dtr,'train'), (dva,'valid')],
                        early_stopping_rounds=int(os.getenv("EARLY_STOP", "500")),
                        verbose_eval=False)
        best_it = getattr(bst, 'best_iteration', None)
        if best_it is None:
            best_it = n_estimators
        print(f"Best iteration: {best_it}")
        # Retrain on full train with best_it
        dfull = xgb.DMatrix(X_tr, label=y_log)
        bst_full = xgb.train(params, dfull, num_boost_round=best_it+1, verbose_eval=False)
        pred_log = bst_full.predict(dte)
    except Exception as e:
        err = e
        print(f"[WARN] XGBoost path failed: {e}; falling back to sklearn.")

    if pred_log is None:
        # Fallback: HistGradientBoostingRegressor
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=float(os.getenv("SK_LR", "0.04")),
            max_depth=int(os.getenv("SK_MAX_DEPTH", "None")) if os.getenv("SK_MAX_DEPTH") else None,
            max_leaf_nodes=int(os.getenv("SK_MAX_LEAVES", "31")),
            min_samples_leaf=int(os.getenv("SK_MIN_SAMPLES_LEAF", "20")),
            l2_regularization=float(os.getenv("SK_L2", "0.0")),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=int(os.getenv("SK_ES_ROUNDS", "100")),
            random_state=2025,
            verbose=1,
        )
        model.fit(X_tr, y_log)
        pred_log = model.predict(X_te)

    print("[6/6] Writing submission...", flush=True)
    pred = np.expm1(pred_log)

    # Build submission using user_session-only mapping (sample_submission has no user_id)
    pred_map = pd.Series(pred, index=df_te["user_session"]).to_dict()
    out = sub.copy()
    global_mean = df_tr["session_value"].mean()
    out["session_value"] = out["user_session"].map(pred_map).fillna(global_mean)
    out = out[["user_session", "session_value"]]

    os.makedirs("submissions", exist_ok=True)
    suffix = ""
    if 0.0 < float(os.getenv("WINSORIZE_P", "0")) < 1.0:
        suffix = f"_win{os.getenv('WINSORIZE_P')}"
    out_path = f"submissions/final_fulltrain_xgb_or_sk_v2_log_backoff_te{suffix}.csv"
    out.to_csv(out_path, index=False)

    print({"submission": out_path, "rows": len(out), "sec": round(time.time()-t0,2)})
    return out_path


if __name__ == "__main__":
    print("[0/6] Starting final full-train script", flush=True)
    out = run_fulltrain()
    print("Wrote:", out, flush=True)
