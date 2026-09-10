from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# --- INIT APP ---
web_simulator_hybrid_backend = Flask(__name__)

# --- LOAD TRAINED MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "new-trained-models")

SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

XGB_MODEL = xgb.Booster()
XGB_MODEL.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

arima_results = joblib.load(
    os.path.join(MODEL_DIR, "arima_model.pkl")
)

SEASONAL_PERIOD = 12

FEATURE_NAMES = [
    'Resid_Lag1',
    'Resid_Lag12',
    'Month',
    'Year',
    'Is_Outlier',
    'RollingMean3',
    'RollingStd3',
    'GDP_Lag1',
    'Inflation_Lag1'
]

# --- OUTLIER DATES ---
OUTLIER_DATES = pd.to_datetime([
    "2011-10-01", "2011-11-01", "2011-12-01",
    "2014-11-01", "2014-12-01",
    "2018-11-01", "2018-12-01",
    "2019-07-01",
    "2020-06-01",
    "2024-01-01", "2024-02-01"
])

# --- INVERSE TRANSFORM ---
def inverse_transform(forecast_stat, history_log):
    y_t_1 = history_log.iloc[-1]
    y_t_12 = history_log.iloc[-SEASONAL_PERIOD]
    y_t_13 = history_log.iloc[-SEASONAL_PERIOD - 1]
    pred_log = forecast_stat + y_t_1 + y_t_12 - y_t_13
    return np.exp(pred_log)

def interpret_dm_result(dm_stat, p_value, alpha=0.05):
    """
    Returns interpretation message and status for UI dashboard
    """

    if dm_stat > 0 and p_value < alpha:
        return {
            "status": "success",
            "message": "Hybrid model is significantly more accurate than ARIMA.",
            "icon": "✅"
        }

    elif dm_stat > 0 and p_value >= alpha:
        return {
            "status": "warning",
            "message": "Hybrid is better, but the difference is not statistically certain.",
            "icon": "⚖️"
        }

    elif dm_stat < 0 and p_value < alpha:
        return {
            "status": "danger",
            "message": "ARIMA is significantly better. Hybrid may be overfitting.",
            "icon": "❌"
        }

    else:
        return {
            "status": "neutral",
            "message": "Both models are performing similarly.",
            "icon": "🤝"
        }


def dm_test(e1, e2, loss="squared"):
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    T = len(e1)

    # 🔒 Minimum sample check
    if T < 5:
        return None, None, "Sample size too small for DM test (T < 5)"

    # Loss differential
    if loss == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        d = np.abs(e1) - np.abs(e2)

    d_mean = np.mean(d)

    # Newey-West variance
    def newey_west_var(d, lag):
        gamma0 = np.var(d, ddof=1)
        var = gamma0

        for j in range(1, lag + 1):
            if len(d[:-j]) > 1:
                gamma_j = np.cov(d[:-j], d[j:])[0, 1]
            else:
                gamma_j = 0

            weight = 1 - j / (lag + 1)
            var += 2 * weight * gamma_j

        return var

    lag = max(int(np.floor(T ** (1/3))), 1)
    var_d = newey_west_var(d, lag)

    # 🔒 Variance safety
    if var_d <= 0 or np.isnan(var_d):
        return None, None, "Variance unstable (cannot compute DM test)"

    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * stats.norm.sf(abs(dm_stat))

    return dm_stat, p_value, None

def hln_test(dm_stat, T, h=1):
    # 🔒 Only compute if DM is valid
    if dm_stat is None or T < 5:
        return None, None

    adjustment = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    hln_stat = dm_stat * adjustment
    p_value = 2 * stats.norm.sf(abs(hln_stat))

    return hln_stat, p_value

# --- ROUTES ---
@web_simulator_hybrid_backend.route('/')
def index():
    return render_template('web_final.html')

@web_simulator_hybrid_backend.route('/forecast', methods=['POST'])
def forecast():
    try:
        # ✅ RESET ARIMA EVERY REQUEST
        arima_results_local = joblib.load(
            os.path.join(MODEL_DIR, "arima_model.pkl")
        )

        file = request.files.get('csv_file')
        steps = int(request.form.get('steps', 3))

        if not file:
            return jsonify({"error": "Historical CSV file is required."})

        df = pd.read_csv(file)

        history_raw = df.iloc[:, -1]
        history_log = np.log(history_raw.replace(0, np.nan).dropna())

        # --- DOUBLE DIFFERENCING ---
        history_stat = []
        for i in range(SEASONAL_PERIOD + 1, len(history_log)):
            val = (
                history_log.iloc[i]
                - history_log.iloc[i - 1]
                - history_log.iloc[i - SEASONAL_PERIOD]
                + history_log.iloc[i - SEASONAL_PERIOD - 1]
            )
            history_stat.append(val)

        arima_preds, hybrid_preds, actuals, labels = [], [], [], []

        last_date = pd.to_datetime(df.iloc[-1, 0])

        for i in range(1, steps + 1):

            # ============================
            # INPUT HANDLING
            # ============================
            if request.form.get('inflation'):
                inf = float(request.form.get('inflation', 0))
                gdp = float(request.form.get('gdp', 0))
            else:
                inf = float(request.form.get(f'inflation_{i}', 0))
                gdp = float(request.form.get(f'gdp_{i}', 0))

            # ✅ NORMALIZE INPUTS (CRITICAL FIX)
            inf = inf / 100
            gdp = gdp / 100

            actual_val = request.form.get(f'actual_{i}')
            actuals.append(float(actual_val) if actual_val else None)
            labels.append(f"Month {i}")

            # ============================
            # ARIMA FORECAST
            # ============================
            forecast_series = arima_results_local.forecast(steps=1)
            arima_stat_pred = forecast_series.iloc[0]

            # Update ARIMA
            arima_results_local = arima_results_local.append(
                np.array([arima_stat_pred]), refit=False
            )

            # ============================
            # FEATURE ENGINEERING
            # ============================
            current_date = last_date + pd.DateOffset(months=i)
            month = current_date.month
            year = current_date.year
            is_outlier = int(current_date.normalize() in OUTLIER_DATES)

            resid_lag1 = history_stat[-1]
            resid_lag12 = history_stat[-12] if len(history_stat) >= 12 else 0

            rollingmean3 = history_log.iloc[-3:].mean()
            rollingstd3 = history_log.iloc[-3:].std()

            feature_values = [
                resid_lag1,
                resid_lag12,
                month,
                year,
                is_outlier,
                rollingmean3,
                rollingstd3,
                gdp,
                inf
            ]

            features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

            SCALER_COLS = [
                'Resid_Lag1',
                'Resid_Lag12',
                'GDP_Lag1',
                'Inflation_Lag1'
            ]

            features_df_scaled = features_df.copy()
            features_df_scaled[SCALER_COLS] = SCALER.transform(features_df[SCALER_COLS])

            dmatrix = xgb.DMatrix(
                features_df_scaled[FEATURE_NAMES],
                feature_names=FEATURE_NAMES
            )

            xgb_adj = XGB_MODEL.predict(dmatrix)[0]

            # ============================
            #  CONTROLLED HYBRID
            # ============================

            # 1. Clip extreme corrections
            xgb_adj = np.clip(xgb_adj, -0.3, 0.3)

            # 2. Adaptive weight (based on confidence)
            if abs(xgb_adj) < 0.05:
                alpha = 0.05   # very small correction
            elif abs(xgb_adj) < 0.15:
                alpha = 0.15
            else:
                alpha = 0.25

            # 3. Apply weighted correction
            hybrid_stat = arima_stat_pred + (alpha * xgb_adj)

            # 4. Safety fallback (VERY IMPORTANT)
            if abs(hybrid_stat - arima_stat_pred) > 0.2:
                hybrid_stat = arima_stat_pred

            # ============================
            # FINAL VALUES
            # ============================
            arima_final = inverse_transform(arima_stat_pred, history_log)
            hybrid_final = inverse_transform(hybrid_stat, history_log)

            arima_preds.append(round(float(arima_final), 2))
            hybrid_preds.append(round(float(hybrid_final), 2))

            # ============================
            # UPDATE HISTORY
            # ============================
            next_val = float(actual_val) if actual_val else hybrid_final
            next_log = np.log(next_val)

            new_stat = (
                next_log
                - history_log.iloc[-1]
                - history_log.iloc[-SEASONAL_PERIOD]
                + history_log.iloc[-SEASONAL_PERIOD - 1]
            )

            history_stat.append(new_stat)
            history_log = pd.concat([history_log, pd.Series([next_log])], ignore_index=True)

            # --- DISPLAY DETAILS IN BACKEND OUTPUT CELL ---
            print(f"\n--- DEBUG DETAILS: STEP {i} ---")
            print(f"1. ARIMA Stationary Forecast: {arima_stat_pred:.6f}")
            print(f"2. Scaled Features:\n{features_df_scaled[FEATURE_NAMES].to_string(index=False)}")
            print(f"3. XGBoost Residual Forecast: {xgb_adj:.6f}")
            print(f"4. Alpha: {alpha}")
            print(f"5. Combined Stationary Forecast: {hybrid_stat:.6f}")
            print(f"5. Final Hybrid Forecast (Price): {round(float(hybrid_final), 2)}")
            print("---------------------------------\n")

        # ============================
        # METRICS
        # ============================
        metrics = None
        valid_actuals = [a for a in actuals if a is not None]

        if len(valid_actuals) == steps:
            metrics = {
                "RMSE_ARIMA": f"{np.sqrt(mean_squared_error(valid_actuals, arima_preds)):.4f}",
                "MAPE_ARIMA": f"{mean_absolute_percentage_error(valid_actuals, arima_preds) * 100:.2f}%",
                "MAE_ARIMA": f"{mean_absolute_error(valid_actuals, arima_preds):.4f}",
                "MPE_ARIMA": f"{np.mean((np.array(valid_actuals) - np.array(arima_preds)) / np.array(valid_actuals)) * 100:.2f}%",
                "RMSE_HYBRID": f"{np.sqrt(mean_squared_error(valid_actuals, hybrid_preds)):.4f}",
                "MAPE_HYBRID": f"{mean_absolute_percentage_error(valid_actuals, hybrid_preds) * 100:.2f}%",
                "MAE_HYBRID": f"{mean_absolute_error(valid_actuals, hybrid_preds):.4f}",
                "MPE_HYBRID": f"{np.mean((np.array(valid_actuals) - np.array(hybrid_preds)) / np.array(valid_actuals)) * 100:.2f}%"
            }

        # DM TEST
        dm_test_result = None

        if len(valid_actuals) == steps:
            actual_arr = np.array(valid_actuals)
            arima_arr = np.array(arima_preds)
            hybrid_arr = np.array(hybrid_preds)

            arima_errors = actual_arr - arima_arr
            hybrid_errors = actual_arr - hybrid_arr

            dm_stat, dm_p, error_msg = dm_test(arima_errors, hybrid_errors)

            if error_msg:
                dm_test_result = {
                    "dm_stat": None,
                    "p_value": None,
                    "hln_stat": None,
                    "hln_p_value": None,
                    "interpretation": error_msg,
                    "status": "neutral",
                    "icon": "⚠️"
                }
            else:
                hln_stat, hln_p = hln_test(dm_stat, len(arima_errors))

                dm_result = interpret_dm_result(dm_stat, dm_p)

                dm_test_result = {
                    "dm_stat": round(float(dm_stat), 4),
                    "p_value": round(float(dm_p), 4),
                    "hln_stat": round(float(hln_stat), 4) if hln_stat is not None else None,
                    "hln_p_value": round(float(hln_p), 4) if hln_p is not None else None,
                    "interpretation": dm_result["message"],
                    "status": dm_result["status"],
                    "icon": dm_result["icon"]
                }

            # FINAL RETURN
        return jsonify({
            "labels": labels,
            "arima": arima_preds,
            "hybrid": hybrid_preds,
            "actual": actuals,
            "error_metrics": metrics,
            "dm_test": dm_test_result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# --- RUN SERVER ---
if __name__ == '__main__':
    web_simulator_hybrid_backend.run()