import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

alpha = 0.05

# ================= LOAD =================
df = pd.read_csv(
    "data_forecast.csv",
    delimiter=";"
)

df.columns = df.columns.str.strip()
df["Bulan"] = pd.to_datetime(df["Bulan"], dayfirst=True)

cols = [
    "EOM OS CTX",
    "EOM OS NEVER CTX",
    "EOM CNT VISIT SUCCESS",
    "EOM Value",
    "DISTANCE EOM AND CUT OFF",
    "Cut Off Value",
    "Cut Off OS CTX",
    "Cut Off OS NEVER CTX",
    "Cut Off VISIT SUCCESS",
    "Cut Off NM OS CTX",
    "Cut Off NM OS NEVER CTX",
    "EOM OS CTX MA(3)"
]

# ================= CLEANING =================
for col in cols:
    df[col] = (
        df[col].astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace(["", "-", "nan", "None"], np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

forecast_all = []

# ================= LOOP =================
for dh in df["DH"].unique():

    if str(dh).strip().lower() == "nasional":
        continue

    print(f"\n===== DH {dh} =====")

    df_dh = df[df["DH"] == dh].copy()
    df_dh = df_dh.sort_values("Bulan").set_index("Bulan")

    y = df_dh["EOM OS CTX"]
    X = df_dh[[
        "EOM OS NEVER CTX",
        "EOM CNT VISIT SUCCESS",
        "EOM Value",
        "DISTANCE EOM AND CUT OFF",
        "Cut Off Value",
        "Cut Off OS CTX",
        "Cut Off OS NEVER CTX",
        "Cut Off VISIT SUCCESS",
        "Cut Off NM OS CTX",
        "Cut Off NM OS NEVER CTX",
        "EOM OS CTX MA(3)"
    ]]

    # ================= LAG =================
    y_lag1 = y.shift(1).rename("EOM OS CTX_lag1")
    X_lag1 = X.shift(1).add_suffix("_lag1")

    data_model = pd.concat([y, y_lag1, X_lag1], axis=1)

    # force numeric + dropna
    data_model = data_model.apply(pd.to_numeric, errors="coerce").dropna()

    if len(data_model) < 10:
        print("Data terlalu sedikit")
        continue

    Y_model = data_model["EOM OS CTX"]
    X_model = sm.add_constant(data_model.drop(columns=["EOM OS CTX"]))

    # ================= BACKWARD ELIMINATION =================
    variables = X_model.columns.tolist()

    while True:
        model = sm.OLS(Y_model, X_model[variables]).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")

        if len(pvalues) == 0:
            break

        max_p = pvalues.max()

        if max_p > alpha:
            worst_var = pvalues.idxmax()
            print(f"Remove: {worst_var} (p={max_p:.4f})")
            variables.remove(worst_var)
        else:
            break

    model_final = sm.OLS(Y_model, X_model[variables]).fit()

    print("\nMODEL FINAL")
    print(model_final.summary())

    data_model["y_hat"] = model_final.predict(X_model[variables])

    # ================= METRICS =================
    mean_y = data_model["EOM OS CTX"].mean()
    rmse = np.sqrt(np.mean((data_model["EOM OS CTX"] - data_model["y_hat"])**2))
    rmse_pct = (rmse / mean_y) * 100
    mape = np.mean(np.abs((data_model["EOM OS CTX"] - data_model["y_hat"]) / data_model["EOM OS CTX"])) * 100

    print("MAPE:", mape)
    print("RMSE %:", rmse_pct)

    # ================= FORECAST =================
    next_index = df_dh.index[-1] + pd.offsets.MonthEnd(1)

    X_next = pd.DataFrame(index=[next_index], columns=variables)

    for col in variables:
        if col == "const":
            X_next.loc[next_index, col] = 1

        elif col == "EOM OS CTX_lag1":
            # 🔥 FIX PENTING
            X_next.loc[next_index, col] = y.iloc[-1]

        else:
            base_col = col.replace("_lag1", "")
            X_next.loc[next_index, col] = df_dh[base_col].iloc[-1]

    # align kolom
    X_next = X_next[variables]

    y_forecast = model_final.predict(X_next)

    forecast_all.append({
        "DH": dh,
        "Forecast": float(y_forecast.iloc[0])
    })

    # ================= PLOT =================
    plt.figure(figsize=(10, 5))

    plt.plot(df_dh.index, df_dh["EOM OS CTX"], marker="o", label="Aktual")
    plt.plot(data_model.index, data_model["y_hat"], linestyle="--", label="In-sample")

    plt.plot([df_dh.index[-1], next_index],
             [y.iloc[-1], y_forecast.iloc[0]],
             marker="o", linewidth=2, label="Forecast")

    plt.title(f"Forecast OS CTX - DH {dh}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ================= OUTPUT =================
forecast_df = pd.DataFrame(forecast_all)

print("\nForecast per DH:")
print(forecast_df)

if not forecast_df.empty:
    print("\nForecast Nasional:", forecast_df["Forecast"].sum())
else:
    print("Tidak ada forecast.")