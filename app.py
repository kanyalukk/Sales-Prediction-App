# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Advertising Sales Regression", layout="wide")
st.title("Sales Prediction (Linear Regression)")
st.caption("อ่านไฟล์ advertising.csv จาก repo และประเมินโมเดลแบบ Linear Regression")

# ===== Step 1: อ่านข้อมูล =====
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "advertising.csv")
    # ลองอ่านด้วย utf-8-sig เพื่อกันปัญหา BOM/ภาษาไทย
    return pd.read_csv(csv_path, encoding="utf-8-sig")

try:
    df = load_data()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ advertising.csv ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

st.subheader("ตัวอย่างข้อมูล (ส่วนต้น)")
st.dataframe(df.head(), use_container_width=True)

# ===== Step 2: แยก X (features) และ y (target: 'sales') =====
if "sales" not in df.columns:
    st.error("ไม่พบคอลัมน์ 'sales' ในไฟล์ CSV กรุณาตรวจสอบชื่อคอลัมน์ให้ตรง")
    st.stop()

df = df.dropna(subset=["sales"])
X = df.drop(columns=["sales"])
y = df["sales"]

# เพื่อความปลอดภัย ถ้า X มีคอลัมน์ที่ไม่ใช่ตัวเลข ให้คัดเฉพาะตัวเลขก่อน
# (เบื้องต้นสำหรับมือใหม่; ภายหลังค่อยเพิ่มการเข้ารหัสหมวดหมู่)
X_num = X.select_dtypes(include=[np.number])
if X_num.shape[1] == 0:
    st.error("คอลัมน์ตัวทำนาย (X) ไม่มีตัวเลขเลย ต้องแปลงหมวดหมู่เป็นตัวเลขก่อน")
    st.stop()

# ===== Step 3: แบ่ง Train/Test 70:30 =====
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.30, random_state=42
)

# ===== Step 4: สร้างและฝึกโมเดล Linear Regression =====
model = LinearRegression()
model.fit(X_train, y_train)

# ===== Step 5: ประเมินผลบนชุดทดสอบ =====
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("ผลการประเมิน (Test Set)")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f}")
c2.metric("RMSE", f"{rmse:.4f}")
c3.metric("R²", f"{r2:.4f}")

# แสดงค่าสัมประสิทธิ์ (น้ำหนักของแต่ละตัวแปร)
coef_df = (
    pd.DataFrame({"feature": X_num.columns, "coef": model.coef_})
    .assign(abs_coef=lambda d: d["coef"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns=["abs_coef"])
)
st.subheader("ค่าสัมประสิทธิ์ของโมเดล (Linear Coefficients)")
st.dataframe(coef_df, use_container_width=True)

# กราฟเปรียบเทียบค่าจริง vs ค่าทำนาย (ดูแนวโน้ม)
st.subheader("Actual vs Predicted (Test Set)")
plot_df = pd.DataFrame(
    {"Actual": y_test.reset_index(drop=True), "Predicted": pd.Series(y_pred)}
)
st.line_chart(plot_df)
