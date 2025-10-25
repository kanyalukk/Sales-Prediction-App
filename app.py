# app.py — Streamlit app for loading a saved model and making predictions
# Inference steps:
# 1) Load the model from "model-reg-66130701925.pkl"
# 2) Create a new DataFrame (default: youtube=50, tiktok=50, instagram=50 if present)
# 3) Predict estimated "sales"

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction (Loaded Model)", layout="wide")
st.title("Sales Prediction from Saved Model")
st.caption("Step 1: Load model → Step 2: Create new DataFrame → Step 3: Predict 'sales'")

# ---------- Step 1: Load the model (and embedded info) ----------
@st.cache_resource
def load_saved_model():
    # หาไฟล์ .pkl ข้าง ๆ app.py ก่อน
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    pkl_path = os.path.join(base_dir, "model-reg-66130701925.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError("ไม่พบไฟล์ 'model-reg-66130701925.pkl' ข้าง ๆ app.py")

    obj = joblib.load(pkl_path)
    # รองรับทั้งกรณีที่บันทึกเป็น dict และกรณีที่เป็นโมเดลล้วน ๆ
    if isinstance(obj, dict):
        model = obj.get("model", None)
        feature_names = obj.get("feature_names", None)
        csv_df = obj.get("csv_df", None)
        if model is None:
            raise ValueError("ไฟล์ .pkl ไม่พบคีย์ 'model'")
        # เผื่อไม่มี feature_names ใน dict ให้ลองดึงจากโมเดล
        if feature_names is None:
            feature_names = getattr(model, "feature_names_in_", None)
    else:
        model = obj
        feature_names = getattr(model, "feature_names_in_", None)
        csv_df = None

    if feature_names is None:
        # กรณีเลวร้ายสุด ไม่มีชื่อคอลัมน์ ให้เดาเป็น feature_0..N-1
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            raise ValueError("ไม่สามารถระบุรายชื่อฟีเจอร์ที่โมเดลต้องการได้")
        feature_names = [f"feature_{i}" for i in range(n_features)]

    return model, list(feature_names), csv_df

try:
    model, feature_names, csv_df = load_saved_model()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

with st.expander("ดูข้อมูลที่ฝังมากับโมเดล (ถ้ามี)"):
    if isinstance(csv_df, pd.DataFrame):
        st.write("ตัวอย่างจาก DataFrame ที่ฝังไว้ในไฟล์โมเดล:")
        st.dataframe(csv_df.head(), use_container_width=True)
    else:
        st.info("ไม่มี DataFrame ฝังไว้ในไฟล์โมเดล")

st.markdown("**Features required by model:** " + ", ".join(feature_names))

# ---------- Step 2: Create a new DataFrame for prediction ----------
st.subheader("ใส่ค่าฟีเจอร์ที่ต้องการพยากรณ์")
st.caption("ค่าเริ่มต้นตั้งให้ youtube/tiktok/instagram = 50 ถ้าฟีเจอร์เหล่านี้อยู่ในโมเดล; ช่องอื่นเริ่มที่ 0")

defaults = {}
for name in feature_names:
    # ตั้งค่าเริ่มต้น 50 เฉพาะฟีเจอร์ยอดฮิตที่ระบุไว้
    if name.lower() in ("youtube", "tiktok", "instagram"):
        defaults[name] = 50.0
    else:
        defaults[name] = 0.0

# วางอินพุตในคอลัมน์เพื่อใช้ง่าย
cols = st.columns(min(4, len(feature_names)) or 1)
user_values = {}
for i, feat in enumerate(feature_names):
    with cols[i % len(cols)]:
        user_values[feat] = st.number_input(
            label=feat, min_value=float("-inf"), max_value=float("inf"),
            value=float(defaults[feat]), step=1.0
        )

# จัด DataFrame ให้ลำดับคอลัมน์ตรงกับ feature_names ของโมเดลเสมอ
X_new = pd.DataFrame([[user_values[f] for f in feature_names]], columns=feature_names)

# ---------- Step 3: Predict estimated "sales" ----------
st.subheader("ผลการพยากรณ์ (Predicted sales)")

if st.button("Predict"):
    try:
        y_pred = model.predict(X_new)
        pred_val = float(y_pred[0])
        st.success(f"Estimated sales: **{pred_val:.4f}**")
        with st.expander("ดูตารางอินพุตที่ใช้ในการพยากรณ์"):
            st.dataframe(X_new, use_container_width=True)
    except Exception as e:
        st.error(f"พยากรณ์ไม่สำเร็จ: {e}")

# ตัวอย่างปุ่มสร้าง DataFrame แบบเร็วตามคำสั่ง [50, 50, 50] สำหรับคอลัมน์ยอดนิยม
st.divider()
st.caption("Quick fill: ตั้งค่า youtube/tiktok/instagram = 50 (ถ้ามีในโมเดล) แล้วพยากรณ์ทันที")
if st.button("Quick Predict (50/50/50)"):
    for key in user_values:
        if key.lower() in ("youtube", "tiktok", "instagram"):
            user_values[key] = 50.0
    X_quick = pd.DataFrame([[user_values[f] for f in feature_names]], columns=feature_names)
    try:
        y_pred = model.predict(X_quick)
        pred_val = float(y_pred[0])
        st.success(f"[Quick] Estimated sales: **{pred_val:.4f}**")
        with st.expander("ดูตารางอินพุต (Quick)"):
            st.dataframe(X_quick, use_container_width=True)
    except Exception as e:
        st.error(f"พยากรณ์ไม่สำเร็จ (Quick): {e}")
