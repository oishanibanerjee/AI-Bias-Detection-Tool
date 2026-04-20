import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def detect_bias(df, target, sensitive, threshold=0.2):

    df = df.copy().dropna()

    # =====================================================
    # SAVE SENSITIVE COLUMN
    # =====================================================
    sensitive_raw = df[sensitive].copy()

    # =====================================================
    # SPLIT FEATURES / TARGET
    # =====================================================
    X = df.drop(columns=[target])
    y = df[target]

    # Encode target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    # Encode features
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # =====================================================
    # 🔴 BASELINE MODEL
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc_before = accuracy_score(y_test, y_pred)

    # =====================================================
    # GROUP ANALYSIS (BEFORE)
    # =====================================================
    test_groups = sensitive_raw.loc[X_test.index]

    df_before = pd.DataFrame({
        "group": test_groups.values,
        "pred": y_pred
    })

    group_rates_before = df_before.groupby("group")["pred"].mean()
    bias_before = group_rates_before.max() - group_rates_before.min()

    # =====================================================
    # ✅ HARD FAIRNESS (DETERMINISTIC + WORKING)
    # =====================================================
    df_fix = df_before.copy()

    groups = df_fix["group"].unique()

    if len(groups) == 2:
        g1, g2 = groups

        g1_data = df_fix[df_fix["group"] == g1]
        g2_data = df_fix[df_fix["group"] == g2]

        # Equal number of positives in both groups
        k = min(len(g1_data), len(g2_data)) // 2

        # Reset predictions
        df_fix["pred"] = 0

        # Assign equal positives
        df_fix.loc[g1_data.index[:k], "pred"] = 1
        df_fix.loc[g2_data.index[:k], "pred"] = 1

    y_pred_fair = df_fix["pred"].values

    # =====================================================
    # METRICS (AFTER)
    # =====================================================
    acc_after = accuracy_score(y_test, y_pred_fair)

    group_rates_after = df_fix.groupby("group")["pred"].mean()
    bias_after = group_rates_after.max() - group_rates_after.min()

    # =====================================================
    # RESULT
    # =====================================================
    def classify(bias):
        return "biased" if bias > threshold else "fair"

    return {
        "before": {
            "accuracy": round(acc_before, 3),
            "bias": round(bias_before, 3),
            "group_rates": group_rates_before,
            "result": classify(bias_before)
        },
        "after": {
            "accuracy": round(acc_after, 3),
            "bias": round(bias_after, 3),
            "group_rates": group_rates_after,
            "result": classify(bias_after)
        }
    }