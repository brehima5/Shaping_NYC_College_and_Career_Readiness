"""
Shared data-loading, model-fitting, and prediction utilities.
All heavy work is cached so every Streamlit page can import these
without re-computing.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "sql" / "CID_database_clean.db"
CSV_DIR = PROJECT_ROOT / "data" / "csv"

# ── display constants ────────────────────────────────────────────────
FEATURE_DISPLAY = {
    "const":                              "Intercept (Baseline)",
    "economic_need_index":                "Economic Need Index",
    "log_temp_housing":                   "Housing Instability (log)",
    "teaching_environment_pct_positive":  "Teaching Environment",
    "eni_x_teach":                        "ENI × Teaching (Interaction)",
    "avg_student_attendance":             "Student Attendance",
    "student_support_pct":                "Student Support",
    "borough_Brooklyn":                   "Brooklyn (vs Bronx)",
    "borough_Manhattan":                  "Manhattan (vs Bronx)",
    "borough_Queens":                     "Queens (vs Bronx)",
    "borough_Staten Island":              "Staten Island (vs Bronx)",
}

SUBGROUP_COLORS = {
    "Asian": "#000000",
    "Black": "#D9D5BA",
    "Hispanic": "#716931",
    "White": "#1CD74B",
}

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]


# ── raw table loader ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data from database…")
def load_raw_tables():
    """Return the four DB tables + student-support from the CSV."""
    conn = sqlite3.connect(str(DB_PATH))
    dim_env  = pd.read_sql_query("SELECT * FROM dim_environment", conn)
    dim_loc  = pd.read_sql_query("SELECT * FROM dim_location", conn)
    dim_dem  = pd.read_sql_query("SELECT * FROM dim_demographic", conn)
    fact     = pd.read_sql_query("SELECT * FROM fact_school_outcomes", conn)
    conn.close()

    # student-support is only in the raw CSV
    env_csv = pd.read_csv(CSV_DIR / "env_dim.csv")
    env_csv = env_csv[["DBN", "Student Support - School Percent Positive"]].copy()
    env_csv.rename(
        columns={"Student Support - School Percent Positive": "student_support_pct"},
        inplace=True,
    )
    env_csv["student_support_pct"] = (
        env_csv["student_support_pct"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .apply(pd.to_numeric, errors="coerce")
        / 100.0
    )
    return dim_env, dim_loc, dim_dem, fact, env_csv


# ── beta-regression pipeline ────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting Beta Regression model…")
def fit_beta_model():
    """Replicate the notebook pipeline and return all model artifacts."""
    dim_env, dim_loc, _, _, env_csv = load_raw_tables()

    # merge
    model_df = (
        dim_env
        .merge(dim_loc[["DBN", "borough", "district"]], on="DBN", how="inner")
        .merge(env_csv, on="DBN", how="left")
    )

    cols_needed = [
        "district", "economic_need_index", "percent_temp_housing",
        "teaching_environment_pct_positive", "avg_student_attendance",
        "student_support_pct", "metric_value_4yr_ccr_all_students", "borough",
    ]
    numerical_cols = [
        "economic_need_index", "percent_temp_housing",
        "teaching_environment_pct_positive", "avg_student_attendance",
        "student_support_pct", "metric_value_4yr_ccr_all_students",
    ]

    # impute by district median
    for col in numerical_cols:
        model_df[col] = model_df[col].fillna(
            model_df.groupby("district")[col].transform("median")
        )
    model_df = model_df[cols_needed].dropna().copy()

    # feature engineering
    model_df["log_temp_housing"] = np.log(model_df["percent_temp_housing"] + 0.001)
    model_df["eni_x_teach"] = (
        model_df["economic_need_index"]
        * model_df["teaching_environment_pct_positive"]
    )

    borough_dummies = pd.get_dummies(
        model_df["borough"], prefix="borough", drop_first=True, dtype=float
    )
    model_df = pd.concat([model_df, borough_dummies], axis=1)

    n_total = len(model_df)
    model_df["ccr_prop"] = model_df["metric_value_4yr_ccr_all_students"] / 100
    # Smithson-Verkuilen squeeze
    model_df["ccr_prop"] = (model_df["ccr_prop"] * (n_total - 1) + 0.5) / n_total

    numerical_features = [
        "economic_need_index", "log_temp_housing",
        "teaching_environment_pct_positive", "eni_x_teach",
        "avg_student_attendance", "student_support_pct",
    ]
    borough_features = list(borough_dummies.columns)
    all_features = numerical_features + borough_features

    # split
    X = model_df[all_features].copy()
    y = model_df["ccr_prop"].values
    y_raw = model_df["metric_value_4yr_ccr_all_students"].values

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.20, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s  = X_test.copy()
    X_train_s[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_s[numerical_features]  = scaler.transform(X_test[numerical_features])

    X_train_c = sm.add_constant(X_train_s)
    X_test_c  = sm.add_constant(X_test_s)

    # fit
    model = BetaModel(y_train, X_train_c).fit(disp=False)

    y_pred_train = model.predict(X_train_c) * 100
    y_pred_test  = model.predict(X_test_c) * 100

    # metrics
    def _metrics(y_act, y_pred, label):
        res = y_act - y_pred
        mae  = np.mean(np.abs(res))
        mede = np.median(np.abs(res))
        rmse = np.sqrt(np.mean(res ** 2))
        mape = np.mean(np.abs(res / y_act)) * 100
        r, _ = pearsonr(y_act, y_pred)
        return dict(Set=label, MAE=round(mae, 2), MedianAE=round(mede, 2),
                    RMSE=round(rmse, 2), MAPE=round(mape, 2),
                    r=round(r, 4), r2=round(r ** 2, 4), N=len(y_act))

    train_m = _metrics(y_raw_train, y_pred_train, "Train")
    test_m  = _metrics(y_raw_test,  y_pred_test,  "Test")

    # coefficient table
    p_names = [n for n in model.params.index if n != "precision"]
    coef_df = pd.DataFrame({
        "Coefficient": [model.params[n] for n in p_names],
        "Std Error":   [model.bse[n]    for n in p_names],
        "z":           [model.tvalues[n] for n in p_names],
        "p":           [model.pvalues[n] for n in p_names],
    }, index=p_names)
    coef_df["sig"] = coef_df["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    )

    # feature ranges for sliders
    ranges = {}
    for f in ["economic_need_index", "percent_temp_housing",
              "teaching_environment_pct_positive", "avg_student_attendance",
              "student_support_pct"]:
        ranges[f] = dict(
            min=float(model_df[f].min()), max=float(model_df[f].max()),
            mean=float(model_df[f].mean()), median=float(model_df[f].median()),
        )

    return dict(
        model=model, scaler=scaler, coef_df=coef_df,
        train_metrics=train_m, test_metrics=test_m,
        model_df=model_df,
        numerical_features=numerical_features,
        borough_features=borough_features,
        all_features=all_features,
        X_test=X_test, y_raw_train=y_raw_train, y_raw_test=y_raw_test,
        y_pred_train=y_pred_train, y_pred_test=y_pred_test,
        param_names=p_names, feature_ranges=ranges,
        precision=float(model.params["precision"]),
    )


# ── single-school prediction ────────────────────────────────────────
def predict_ccr(art, eni, pct_temp, teaching, attendance, support, borough):
    """Return (predicted_ccr_pct, {feature: logit_contribution})."""
    model  = art["model"]
    scaler = art["scaler"]
    nf     = art["numerical_features"]
    bf     = art["borough_features"]

    log_temp    = np.log(pct_temp + 0.001)
    interaction = eni * teaching

    raw = np.array([[eni, log_temp, teaching, interaction, attendance, support]])
    scaled = scaler.transform(raw)[0]

    borough_vals = [1.0 if f"borough_{borough}" == b else 0.0 for b in bf]
    features = np.concatenate([[1.0], scaled, borough_vals])

    pred_prop = model.predict(features.reshape(1, -1))[0]
    pred_ccr  = pred_prop * 100

    # per-feature logit contributions
    params = model.params
    pn = [n for n in params.index if n != "precision"]
    contribs = {name: float(params[name] * features[i]) for i, name in enumerate(pn)}

    return pred_ccr, contribs


# ── subgroup dataset ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Building subgroup equity dataset…")
def build_subgroup_data():
    dim_env, dim_loc, dim_dem, fact, _ = load_raw_tables()

    sg = fact.copy()
    sg["ccr_pct"] = sg["ccr_rate"] * 100

    sg = sg.merge(
        dim_dem[["DBN", "Subgroup", "student_percent", "nearby_student_percent",
                 "pct_students_advanced_courses", "teacher_percent"]],
        on=["DBN", "Subgroup"], how="left",
    )

    env_cols = ["DBN", "economic_need_index", "percent_temp_housing",
                "teaching_environment_pct_positive", "avg_student_attendance",
                "metric_value_4yr_ccr_all_students"]
    sg = sg.merge(dim_env[env_cols], on="DBN", how="left")
    sg = sg.merge(dim_loc[["DBN", "borough"]], on="DBN", how="left")

    sg["ccr_status"] = np.where(
        sg["ccr_pct"].notna(), "reported",
        np.where(sg["n_count_ccr"].notna(), "suppressed", "no cohort"),
    )

    reported = sg[sg["ccr_pct"].notna()].copy()

    # within-school gaps (schools with ≥2 subgroups reporting)
    multi = reported.groupby("DBN").filter(lambda x: len(x) >= 2).copy()
    school_ccr = (
        multi.groupby("DBN")["metric_value_4yr_ccr_all_students"]
        .first()
        .rename("school_mean_ccr")
    )
    multi = multi.merge(school_ccr, on="DBN")
    multi["intra_school_gap"] = multi["ccr_pct"] - multi["school_mean_ccr"]

    return sg, reported, multi
