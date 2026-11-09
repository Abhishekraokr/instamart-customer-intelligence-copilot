import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def run_rfm_and_clustering():
    print("=== PART B: RFM AND CUSTOMER CLUSTERING ===")
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    completed = orders[orders["status"] == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])

    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)

    rfm_table = rfm[["customer_id", "recency", "frequency", "monetary"]].copy()

    print(f"Total customers with completed orders: {len(rfm_table)} (Expected: 1,175)")
    print(f"Recency: min={rfm_table['recency'].min()}, median={rfm_table['recency'].median()}, max={rfm_table['recency'].max()}")
    print(f"Frequency: mean={rfm_table['frequency'].mean():.2f}, max={rfm_table['frequency'].max()}")
    print(f"Monetary: mean={rfm_table['monetary'].mean():.2f}, max={rfm_table['monetary'].max():.2f}")

    scaler = StandardScaler()
    X_rfm = rfm_table[["recency", "frequency", "monetary"]].values
    X_scaled = scaler.fit_transform(X_rfm)

    print("\nK-Means Evaluation (K=2 to 8):")
    print("K\tInertia\t\tSilhouette")
    kmeans_results = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        inertia = km.inertia_
        kmeans_results[k] = (inertia, sil)
        print(f"{k}\t{inertia:.1f}\t\t{sil:.3f}")

    chosen_k = 4
    km_final = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm_table["segment_id"] = km_final.fit_predict(X_scaled)

    summary = rfm_table.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    print("\nSegment Profiles (K=4):")
    print(summary)

    premium_cluster_id = summary.sort_values("mean_monetary", ascending=False).iloc[0]["segment_id"]
    rfm_table["is_premium"] = (rfm_table["segment_id"] == premium_cluster_id)

    highest_freq_id = summary.sort_values("mean_frequency", ascending=False).iloc[0]["segment_id"]
    highest_rec_id = summary.sort_values("mean_recency", ascending=False).iloc[0]["segment_id"]

    def assign_name(row):
        sid = row["segment_id"]
        if sid == premium_cluster_id:
            return "Premium High-Spenders"
        elif sid == highest_freq_id:
            return "Frequent Loyalists"
        elif sid == highest_rec_id:
            return "At-Risk Dormant"
        else:
            return "Occasional Shoppers"

    rfm_table["segment_name"] = rfm_table.apply(assign_name, axis=1)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance Ratios: {var_ratio[0]:.3f}, {var_ratio[1]:.3f} (Total: {var_ratio.sum():.3f})")

    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "customer_segments.csv")
    rfm_table[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    return rfm_table

def run_lapse_modeling(rfm_table):
    print("\n=== PART B: LAPSE MODELING ===")
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    customers = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
    products = pd.read_csv(os.path.join("data", "clean", "products.csv"))

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    cutoff = pd.to_datetime("2025-09-01")

    orders_before = orders[orders["order_ts"] < cutoff].copy()
    completed_before = orders_before[orders_before["status"] == "completed"].copy()

    cohort_cids = sorted(completed_before["customer_id"].unique())
    print(f"Cohort size before cutoff: {len(cohort_cids)} (Expected: 1,156)")

    orders_after = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= pd.to_datetime("2025-09-30 23:59:59"))].copy()
    completed_after = orders_after[orders_after["status"] == "completed"].copy()
    active_after_cids = set(completed_after["customer_id"].unique())

    cat_map = dict(zip(products["product_id"], products["category"]))
    orders_before["category"] = orders_before["product_id"].map(cat_map)
    completed_before["category"] = completed_before["product_id"].map(cat_map)

    recency_s = completed_before.groupby("customer_id")["order_ts"].max().apply(lambda dt: (cutoff - dt).days)

    frequency_s = completed_before.groupby("customer_id")["order_id"].nunique()

    monetary_s = completed_before.groupby("customer_id")["order_amount"].sum()

    avg_discount_s = completed_before.groupby("customer_id")["discount_pct"].mean()

    total_orders_s = orders_before[orders_before["status"].isin(["completed", "cancelled"])].groupby("customer_id")["order_id"].nunique()
    cancelled_orders_s = orders_before[orders_before["status"] == "cancelled"].groupby("customer_id")["order_id"].nunique()
    cancellation_rate_s = (cancelled_orders_s / total_orders_s).fillna(0.0)

    avg_delivery_s = completed_before.groupby("customer_id")["delivery_minutes"].mean()

    def top_cat(s):
        modes = s.mode()
        return modes.iloc[0] if len(modes) > 0 else s.iloc[0]

    fav_cat_s = completed_before.groupby("customer_id")["category"].agg(top_cat)

    df_features = pd.DataFrame({"customer_id": cohort_cids})
    df_features["recency"] = df_features["customer_id"].map(recency_s)
    df_features["frequency"] = df_features["customer_id"].map(frequency_s)
    df_features["monetary"] = df_features["customer_id"].map(monetary_s)
    df_features["discount_pct"] = df_features["customer_id"].map(avg_discount_s)
    df_features["cancellation_rate"] = df_features["customer_id"].map(cancellation_rate_s).fillna(0.0)
    df_features["delivery_minutes"] = df_features["customer_id"].map(avg_delivery_s)
    df_features["favourite_category"] = df_features["customer_id"].map(fav_cat_s)

    cust_map = customers.set_index("customer_id")
    df_features["city"] = df_features["customer_id"].map(cust_map["city"])
    df_features["age"] = df_features["customer_id"].map(cust_map["age"])
    df_features["gender"] = df_features["customer_id"].map(cust_map["gender"])
    df_features["tenure_days"] = (cutoff - df_features["customer_id"].map(cust_map["signup_date"])).dt.days

    df_features["lapsed"] = df_features["customer_id"].apply(lambda cid: 0 if cid in active_after_cids else 1)

    print(f"Total cohort: {len(df_features)}")
    print(f"Lapsed count: {df_features['lapsed'].sum()} ({df_features['lapsed'].mean()*100:.2f}%) (Expected: 443, 38.32%)")
    assert len(df_features) == 1156, f"Expected 1156 cohort customers, got {len(df_features)}"
    assert df_features["lapsed"].sum() == 443, f"Expected 443 lapsed, got {df_features['lapsed'].sum()}"

    assert df_features.isna().sum().sum() == 0, "Missing values found in feature set!"

    X = df_features.drop(columns=["customer_id", "lapsed"])
    y = df_features["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    assert len(X_train) == 924 and len(X_test) == 232

    y_pred_base = np.zeros(len(y_test))
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, zero_division=0)
    print(f"Baseline (all 0s): Accuracy={base_acc*100:.1f}%, F1={base_f1:.4f}")

    numeric_cols = ["recency", "frequency", "monetary", "discount_pct", "cancellation_rate", "delivery_minutes", "age", "tenure_days"]
    nominal_cols = ["city", "gender", "favourite_category"]

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    learned_freq_mean = scaler.mean_[numeric_cols.index("frequency")]
    full_freq_mean = X["frequency"].mean()
    print(f"Learned mean of frequency (train only): {learned_freq_mean:.4f}")
    print(f"Full-cohort mean of frequency: {full_freq_mean:.4f}")

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe.fit(X_train[nominal_cols])

    X_train_num = scaler.transform(X_train[numeric_cols])
    X_train_cat = ohe.transform(X_train[nominal_cols])
    X_train_proc = np.hstack([X_train_num, X_train_cat])

    X_test_num = scaler.transform(X_test[numeric_cols])
    X_test_cat = ohe.transform(X_test[nominal_cols])
    X_test_proc = np.hstack([X_test_num, X_test_cat])

    cat_feature_names = ohe.get_feature_names_out(nominal_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    print("\nTuning Logistic Regression...")
    param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train_proc, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"Best LR Params: {grid_lr.best_params_}, Best CV F1: {grid_lr.best_score_:.4f}")

    print("\nTuning Random Forest...")
    param_grid_rf = {
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42), param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train_proc, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"Best RF Params: {grid_rf.best_params_}, Best CV F1: {grid_rf.best_score_:.4f}")

    y_pred_lr = best_lr.predict(X_test_proc)
    y_pred_rf = best_rf.predict(X_test_proc)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

    print("\nTest Evaluation:")
    print(f"Logistic Regression: Prec={prec_lr:.3f}, Rec={rec_lr:.3f}, F1={f1_lr:.3f}")
    print(f"LR Confusion Matrix:\n{cm_lr}")
    print(f"Random Forest: Prec={prec_rf:.3f}, Rec={rec_rf:.3f}, F1={f1_rf:.3f}")
    print(f"RF Confusion Matrix:\n{cm_rf}")

    print("\n--- Class-Balance Experiment ---")
    train_indices = X_train.index.values
    df_train = df_features.loc[train_indices].copy()
    train_lapsed_idx = df_train[df_train["lapsed"] == 1].index.values
    train_nonlapsed_idx = df_train[df_train["lapsed"] == 0].index.values
    n_majority = len(train_nonlapsed_idx)
    n_minority = len(train_lapsed_idx)

    rng = np.random.RandomState(42)
    oversample_lapsed = rng.choice(train_lapsed_idx, size=(n_majority - n_minority), replace=True)
    balanced_train_indices = np.concatenate([train_nonlapsed_idx, train_lapsed_idx, oversample_lapsed])

    X_train_bal = df_features.loc[balanced_train_indices].drop(columns=["customer_id", "lapsed"])
    y_train_bal = df_features.loc[balanced_train_indices]["lapsed"]

    X_train_bal_proc = np.hstack([scaler.transform(X_train_bal[numeric_cols]), ohe.transform(X_train_bal[nominal_cols])])

    lr_balanced = LogisticRegression(C=grid_lr.best_params_["C"], max_iter=1000, random_state=42)
    lr_balanced.fit(X_train_bal_proc, y_train_bal)
    y_pred_bal = lr_balanced.predict(X_test_proc)

    prec_bal = precision_score(y_test, y_pred_bal, zero_division=0)
    rec_bal = recall_score(y_test, y_pred_bal, zero_division=0)
    f1_bal = f1_score(y_test, y_pred_bal, zero_division=0)

    print(f"Balanced LR on Test: Prec={prec_bal:.3f}, Rec={rec_bal:.3f}, F1={f1_bal:.3f}")

    importances = best_rf.feature_importances_
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Most Important Features (Random Forest):")
    for feat, imp in feat_imp[:5]:
        print(f"  {feat}: {imp:.4f}")

    lr_coefs = dict(zip(all_feature_names, best_lr.coef_[0]))
    print("\nSample LR Feature Coefficients:")
    for feat in ["recency", "frequency", "monetary"]:
        print(f"  {feat}: {lr_coefs.get(feat, 0):.4f}")

    model_bundle = {
        "scaler": scaler,
        "ohe": ohe,
        "numeric_cols": numeric_cols,
        "nominal_cols": nominal_cols,
        "feature_names": all_feature_names,
        "model": best_rf,
    }
    model_path = os.path.join("outputs", "lapse_model.joblib")
    joblib.dump(model_bundle, model_path)
    print(f"\nSaved model bundle to {model_path}")

    reloaded = joblib.load(model_path)
    X_test_reloaded = np.hstack([
        reloaded["scaler"].transform(X_test[numeric_cols]),
        reloaded["ohe"].transform(X_test[nominal_cols])
    ])
    preds_reloaded = reloaded["model"].predict(X_test_reloaded)
    assert np.array_equal(y_pred_rf, preds_reloaded), "Model reloading failed to match predictions!"
    print("Reloaded model predictions match exactly.")

    test_cids = df_features.loc[X_test.index, "customer_id"].values
    test_probs = best_rf.predict_proba(X_test_proc)[:, 1]
    test_actuals = y_test.values

    df_test_res = pd.DataFrame({
        "customer_id": test_cids,
        "lapse_probability": test_probs.round(4),
        "actually_lapsed": test_actuals
    })

    prem_cids = set(rfm_table[rfm_table["is_premium"]]["customer_id"])
    df_prem_test = df_test_res[df_test_res["customer_id"].isin(prem_cids)].copy()

    df_prem_test = df_prem_test.sort_values("lapse_probability", ascending=False).reset_index(drop=True)
    top_retention = df_prem_test.head(10)
    retention_path = os.path.join("outputs", "retention_list.csv")
    top_retention.to_csv(retention_path, index=False)
    print(f"\nSaved Top 10 Premium Retention List to {retention_path}")
    print(top_retention)
    print(f"Truly lapsed among top retention candidates: {top_retention['actually_lapsed'].sum()} / {len(top_retention)}")

if __name__ == "__main__":
    rfm_table = run_rfm_and_clustering()
    run_lapse_modeling(rfm_table)

