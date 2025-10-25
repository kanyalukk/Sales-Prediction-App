# app.py — Streamlit app: load saved model and predict "sales"
# Run:
#   pip install streamlit pandas scikit-learn joblib numpy
#   streamlit run app.py
#
# Needs: model-reg-66130701925.pkl in the same folder.

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction from Saved Model", layout="wide")
st.title("Sales Prediction from Saved Model")
st.caption("Step 1: Load model → Step 2: Create new DataFrame → Step 3: Predict 'sales'")

# ---------- Step 1: Load the model (and embedded info) ----------
@st.cache_resource
def load_saved_model():
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    pkl_path = os.path.join(base_dir, "model-reg-66130701925.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError("ไม่พบไฟล์ 'model-reg-66130701925.pkl' ข้าง ๆ app.py")

    obj = joblib.load(pkl_path)

    if isinstance(obj, dict):
        model = obj.get("model", None)
        feature_names = obj.get("feature_names", None)
        csv_df = obj.get("csv_df", None)
        if model is None:
            raise ValueError("ไฟล์ .pkl ไม่พบคีย์ 'model'")
        if feature_names is None:
            feature_names = getattr(model, "feature_names_in_", None)
    else:
        model = obj
        feature_names = getattr(model, "feature_names_in_", None)
        csv_df = None

    if feature_names is None:
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
        st.dataframe(csv_df.head(), use_container_width=True)
    else:
        st.info("ไม่มี DataFrame ฝังไว้ในไฟล์โมเดล")

st.markdown("**Features required by model:** " + ", ".join(feature_names))

# ---------- Step 2: Create a new DataFrame for prediction ----------
st.subheader("ใส่ค่าฟีเจอร์ที่ต้องการพยากรณ์")
st.caption("ค่าเริ่มต้นตั้งให้ youtube/tiktok/instagram = 50 ถ้ามีในโมเดล; ช่องอื่นเริ่มที่ 0")

defaults = {}
for name in feature_names:
    if name.lower() in ("youtube", "tiktok", "instagram"):
        defaults[name] = 50.0
    else:
        defaults[name] = 0.0

num_cols = max(1, min(4, len(feature_names)))
cols = st.columns(num_cols)

user_values = {}
for i, feat in enumerate(feature_names):
    with cols[i % num_cols]:
        # FIX: no ±inf bounds; keep it flexible but finite
        user_values[feat] = st.number_input(
            label=feat,
            value=float(defaults[feat]),
            step=1.0,
            format="%.6f",
        )

# DataFrame in the exact order expected by the model
X_new = pd.DataFrame([[user_values[f] for f in feature_names]], columns=feature_names)

# ---------- Step 3: Predict estimated "sales" ----------
st.subheader("ผลการพยากรณ์ (Predicted sales)")

if st.button("Predict"):
    try:
        y_pred = model.predict(X_new)
        st.success(f"Estimated sales: **{float(y_pred[0]):.4f}**")
        with st.expander("ดูตารางอินพุตที่ใช้ในการพยากรณ์"):
            st.dataframe(X_new, use_container_width=True)
    except Exception as e:
        st.error(f"พยากรณ์ไม่สำเร็จ: {e}")

st.divider()
st.caption("Quick fill: ตั้งค่า youtube/tiktok/instagram = 50 (ถ้ามีในโมเดล) แล้วพยากรณ์ทันที")
if st.button("Quick Predict (50/50/50)"):
    for k in user_values:
        if k.lower() in ("youtube", "tiktok", "instagram"):
            user_values[k] = 50.0
    X_quick = pd.DataFrame([[user_values[f] for f in feature_names]], columns=feature_names)
    try:
        y_pred = model.predict(X_quick)
        st.success(f"[Quick] Estimated sales: **{float(y_pred[0]):.4f}**")
        with st.expander("ดูตารางอินพุต (Quick)"):
            st.dataframe(X_quick, use_container_width=True)
    except Exception as e:
        st.error(f"พยากรณ์ไม่สำเร็จ (Quick): {e}")
