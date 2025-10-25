# app.py  — Streamlit Regression App
# วิธีรัน:
#   pip install streamlit pandas scikit-learn numpy
#   streamlit run app.py
#
# ต้องมีไฟล์ advertising.csv อยู่โฟลเดอร์เดียวกับไฟล์นี้ และมีคอลัมน์เป้าหมายชื่อ "sales"

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Advertising Regression", layout="wide")
st.title("Linear Regression on advertising.csv")
st.caption("Step 1–5: read CSV → split X/y → split 70:30 → train Linear Regression → evaluate on test set")

# ---------- Step 1: Read advertising.csv ----------
@st.cache_data
def load_csv_from_local():
    csv_path = os.path.join(os.path.dirname(__file__), "advertising.csv")
    return pd.read_csv(csv_path, encoding="utf-8-sig")

try:
    df = load_csv_from_local()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'advertising.csv' ข้าง ๆ app.py — โปรดวางไฟล์ไว้โฟลเดอร์เดียวกันแล้วกด Rerun")
    st.stop()

st.subheader("ตัวอย่างข้อมูล (5 แถวแรก)")
st.dataframe(df.head(), use_container_width=True)

# ---------- Step 2: Split features (X) and target (y='sales') ----------
if "sales" not in df.columns:
    st.error("ไม่พบคอลัมน์ 'sales' ในไฟล์ CSV (ต้องสะกดตรงตัว)")
    st.stop()

df = df.dropna(subset=["sales"])
X = df.drop(columns=["sales"])
y = df["sales"]

# ทำให้ใช้งานได้กับทุกกรณี: เข้ารหัสหมวดหมู่ + เก็บตัวเลขเดิม
X = pd.get_dummies(X, drop_first=True)
# เติมค่าว่างเชิงตัวเลข (กันพัง)
X = X.fillna(X.median(numeric_only=True))

if X.shape[1] == 0:
    st.error("ไม่พบฟีเจอร์สำหรับฝึกโมเดล (หลังแปลงเป็นตัวเลขแล้วคอลัมน์หายหมด)")
    st.stop()

# ---------- Step 3: Train/Test split = 70:30 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------- Step 4: Build & Train Linear Regression ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- Step 5: Evaluate on the test set ----------
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

st.subheader("ผลการประเมิน (Test Set)")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f}")
c2.metric("RMSE", f"{rmse:.4f}")
c3.metric("R²", f"{r2:.4f}")

# แสดงค่าสัมประสิทธิ์ของโมเดล (เรียงตามขนาดสัมบูรณ์)
coef_df = (
    pd.DataFrame({"feature": X.columns, "coef": model.coef_})
    .assign(abs_coef=lambda d: d["coef"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns=["abs_coef"])
)
st.subheader("ค่าสัมประสิทธิ์ของโมเดล (Linear Coefficients)")
st.dataframe(coef_df, use_container_width=True)

# กราฟเปรียบเทียบค่าจริง vs ค่าพยากรณ์
st.subheader("Actual vs Predicted (Test Set)")
plot_df = pd.DataFrame(
    {"Actual": y_test.reset_index(drop=True), "Predicted": pd.Series(y_pred)}
)
st.line_chart(plot_df, use_container_width=True)
ing=False).to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")
