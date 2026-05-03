"""
╠══════════════════════════════════════════════════════════════════╣
║  HOW TO RUN                                                     ║
║    pip install streamlit joblib scikit-learn xgboost            ║
║               scipy numpy pandas matplotlib seaborn shap        ║
║    streamlit run har70_app.py                                   ║
║                                                                  ║
║  REQUIREMENTS                                                   ║
║    Run har70_pipeline_FINAL.py first to generate:               ║
║      ./har70_output/best_model.pkl                              ║
║      ./har70_output/scaler.pkl                                  ║
║      ./har70_output/label_encoder.pkl                           ║
║      ./har70_output/feature_names.pkl                           ║
║      ./har70_output/plots/*.png  (all 28 plots)                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.signal import welch

st.set_page_config(
    page_title="HAR70+ Activity Recognizer",
    page_icon="🏃", layout="wide",
    initial_sidebar_state="expanded"
)

OUTPUT_DIR  = "./har70_output"
SENSOR_COLS = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"]
WINDOW_SIZE = 100

ACTIVITY_NAMES = {
    1:"Walking", 3:"Shuffling", 4:"Stairs Up",
    5:"Stairs Down", 6:"Standing", 7:"Sitting", 8:"Lying"
}
ACTIVITY_ICONS = {
    "Walking":"🚶","Shuffling":"🦯","Stairs Up":"⬆️",
    "Stairs Down":"⬇️","Standing":"🧍","Sitting":"🪑","Lying":"🛌"
}
ACTIVITY_COLORS = {
    "Walking":"#2196F3","Shuffling":"#4CAF50","Stairs Up":"#FF5722",
    "Stairs Down":"#9C27B0","Standing":"#FF9800","Sitting":"#00BCD4","Lying":"#E91E63"
}

# Typical mean sensor values per activity (derived from real HAR70+ data)
# back_x/y/z: acceleration in g along down/left/forward axes
# thigh_x/y/z: acceleration in g along down/right/backward axes
ACTIVITY_PRESETS = {
    "Walking":    dict(back_x= 0.10,back_y=-0.10,back_z= 0.00,thigh_x= 0.50,thigh_y= 0.10,thigh_z=-0.20),
    "Shuffling":  dict(back_x= 0.05,back_y=-0.05,back_z= 0.00,thigh_x= 0.30,thigh_y= 0.05,thigh_z=-0.10),
    "Stairs Up":  dict(back_x= 0.20,back_y=-0.15,back_z= 0.10,thigh_x= 0.60,thigh_y= 0.20,thigh_z=-0.30),
    "Stairs Down":dict(back_x= 0.15,back_y=-0.20,back_z= 0.10,thigh_x= 0.55,thigh_y= 0.15,thigh_z=-0.25),
    "Standing":   dict(back_x= 0.00,back_y=-0.02,back_z= 1.00,thigh_x= 0.70,thigh_y= 0.05,thigh_z=-0.70),
    "Sitting":    dict(back_x= 0.00,back_y=-0.05,back_z= 0.98,thigh_x= 0.85,thigh_y= 0.10,thigh_z=-0.50),
    "Lying":      dict(back_x= 0.98,back_y= 0.05,back_z= 0.02,thigh_x= 0.95,thigh_y= 0.05,thigh_z=-0.02),
}

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1B3A6B,#2196F3);padding:2rem;
 border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center}
.main-header h1{font-size:2.2rem;margin:0;font-weight:700}
.main-header p{font-size:1rem;margin:.5rem 0 0;opacity:.85}

.metric-card{background:white;border-radius:10px;padding:1.2rem;text-align:center;
 box-shadow:0 2px 8px rgba(0,0,0,.08);border-left:4px solid #2196F3}
.metric-card h2{font-size:2rem;margin:0;color:#1B3A6B;font-weight:700}
.metric-card p{font-size:.85rem;margin:.3rem 0 0;color:#444} /* Darker gray */

.pred-box{border-radius:12px;padding:2rem;text-align:center;margin:1rem 0;color:white}
.pred-box h2{font-size:3rem;margin:0;color:white}
.pred-box h3{font-size:1.5rem;margin:.5rem 0 0;opacity:.9;color:white}

/* FIXED BOXES BELOW: Added explicit dark text colors */
.info-box{background:#f0f7ff; border-left:4px solid #2196F3;
 padding:1rem 1.2rem; border-radius:6px; margin:.5rem 0; color:#1B3A6B}

.warn-box{background:#fff8e1; border-left:4px solid #FF9800;
 padding:1rem 1.2rem; border-radius:6px; margin:.5rem 0; color:#856404}

.ok-box{background:#f1f8e9; border-left:4px solid #4CAF50;
 padding:1rem 1.2rem; border-radius:6px; margin:.5rem 0; color:#1b5e20}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        m  = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")
        sc = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
        le = joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")
        fn = joblib.load(f"{OUTPUT_DIR}/feature_names.pkl")
        return m, sc, le, fn, None
    except FileNotFoundError as e:
        return None,None,None,None,str(e)

model, scaler, le, feature_names, load_error = load_artifacts()

def extract_features(win_df):
    feats = {}
    for col in SENSOR_COLS:
        s = win_df[col].values
        feats[f"{col}_mean"]        = np.mean(s)
        feats[f"{col}_std"]         = np.std(s)
        feats[f"{col}_min"]         = np.min(s)
        feats[f"{col}_max"]         = np.max(s)
        feats[f"{col}_range"]       = np.ptp(s)
        feats[f"{col}_median"]      = np.median(s)
        feats[f"{col}_q25"]         = np.percentile(s,25)
        feats[f"{col}_q75"]         = np.percentile(s,75)
        feats[f"{col}_iqr"]         = np.percentile(s,75)-np.percentile(s,25)
        feats[f"{col}_skew"]        = skew(s)
        feats[f"{col}_kurt"]        = kurtosis(s)
        feats[f"{col}_energy"]      = np.sum(s**2)/len(s)
        feats[f"{col}_rms"]         = np.sqrt(np.mean(s**2))
        feats[f"{col}_mad"]         = np.mean(np.abs(s-np.mean(s)))
        feats[f"{col}_zcr"]         = int(((s[:-1]*s[1:])<0).sum())
        feats[f"{col}_var"]         = np.var(s)
        fr,psd = welch(s,fs=50,nperseg=min(50,len(s)))
        feats[f"{col}_psd_mean"]    = np.mean(psd)
        feats[f"{col}_psd_max"]     = np.max(psd)
        feats[f"{col}_dom_freq"]    = fr[np.argmax(psd)]
        pn = psd/(psd.sum()+1e-10)
        feats[f"{col}_spec_entropy"]= -np.sum(pn*np.log(pn+1e-10))
    bm = np.sqrt(win_df["back_x"]**2+win_df["back_y"]**2+win_df["back_z"]**2)
    tm = np.sqrt(win_df["thigh_x"]**2+win_df["thigh_y"]**2+win_df["thigh_z"]**2)
    feats["back_mag_mean"] =np.mean(bm); feats["thigh_mag_mean"]=np.mean(tm)
    feats["back_mag_std"]  =np.std(bm);  feats["thigh_mag_std"] =np.std(tm)
    feats["back_mag_max"]  =np.max(bm);  feats["thigh_mag_max"] =np.max(tm)
    feats["mag_diff_mean"] =np.mean(np.abs(bm-tm))
    feats["mag_corr"]      =np.corrcoef(bm,tm)[0,1]
    feats["back_xy_corr"]  =np.corrcoef(win_df["back_x"],win_df["back_y"])[0,1]
    feats["back_xz_corr"]  =np.corrcoef(win_df["back_x"],win_df["back_z"])[0,1]
    feats["back_yz_corr"]  =np.corrcoef(win_df["back_y"],win_df["back_z"])[0,1]
    feats["thigh_xy_corr"] =np.corrcoef(win_df["thigh_x"],win_df["thigh_y"])[0,1]
    feats["thigh_xz_corr"] =np.corrcoef(win_df["thigh_x"],win_df["thigh_z"])[0,1]
    feats["thigh_yz_corr"] =np.corrcoef(win_df["thigh_y"],win_df["thigh_z"])[0,1]
    v = np.array([feats.get(f,0.0) for f in feature_names])
    return np.nan_to_num(v,nan=0.0), feats

def build_window(means, noise, seed):
    """Build 100-sample window from user-supplied mean values + noise + oscillation."""
    np.random.seed(seed)
    t = np.linspace(0,2,WINDOW_SIZE)
    bz = abs(means["back_z"]); bx = abs(means["back_x"])
    # Decide oscillation: static posture vs dynamic
    if bz > 0.7 or bx > 0.7:
        freq, amp_b, amp_t = 0.0, 0.0, 0.0   # static
    else:
        tm_est = np.sqrt(means["thigh_x"]**2+means["thigh_y"]**2+means["thigh_z"]**2)
        if tm_est > 0.75:
            freq, amp_b, amp_t = 1.4, 0.40, 0.55  # stairs
        elif tm_est > 0.45:
            freq, amp_b, amp_t = 1.8, 0.30, 0.45  # walking
        else:
            freq, amp_b, amp_t = 1.0, 0.12, 0.18  # shuffling
    data = {}
    for i,col in enumerate(["back_x","back_y","back_z"]):
        osc = amp_b*np.sin(2*np.pi*freq*t+i*np.pi/3) if freq>0 else 0.0
        data[col] = means[col]+osc+np.random.normal(0,noise,WINDOW_SIZE)
    for i,col in enumerate(["thigh_x","thigh_y","thigh_z"]):
        osc = amp_t*np.sin(2*np.pi*freq*t+i*np.pi/3+np.pi/4) if freq>0 else 0.0
        data[col] = means[col]+osc+np.random.normal(0,noise,WINDOW_SIZE)
    return pd.DataFrame(data)

def predict(win_df):
    fv, fd = extract_features(win_df)
    fs = scaler.transform(fv.reshape(1,-1))
    pe = model.predict(fs)[0]
    pr = model.predict_proba(fs)[0]
    pl = le.inverse_transform([pe])[0]
    pn = ACTIVITY_NAMES[pl]
    cn = [ACTIVITY_NAMES[l] for l in le.classes_]
    return pn, pr, cn, fd

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏃 HAR70+ Project")
    st.markdown("**CS-245 Machine Learning**  \nNUST SEECS — Spring 2026")
    st.divider()
    page = st.radio("Navigation",[
        "🏠 Home & Overview","🔍 Live Prediction",
        "📊 Model Results","🖼️ All Plots",
        "🧠 SHAP Explainability","📁 Batch CSV Prediction",
    ], label_visibility="collapsed")
    st.divider()
    if load_error:
        st.error("⚠️ Model not loaded")
        st.caption("Run pipeline first:\n`python har70_pipeline_FINAL.py`")
    else:
        st.success("✅ Model loaded  |  XGBoost")
    st.divider()
    st.markdown("**Dataset:** HAR70+  \n**Subjects:** 18 (70–95 yrs)  \n"
                "**Classes:** 7 activities  \n**Features:** 134/window  \n"
                "**Best Acc:** 96.9%")

# ════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.markdown("""<div class="main-header">
        <h1>🏃 HAR70+ Activity Recognition</h1>
        <p>Human Activity Recognition for Older Adults ·
           CS-245 Machine Learning · NUST SEECS</p></div>""",
        unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,v,lbl in zip([c1,c2,c3,c4],
                          ["96.9%","0.9985","18","134"],
                          ["XGBoost Accuracy","AUC (OvR)","Subjects (70–95)","Features/Window"]):
        col.markdown(f'<div class="metric-card"><h2>{v}</h2><p>{lbl}</p></div>',
                     unsafe_allow_html=True)
    st.markdown("---")
    cl,cr = st.columns([1.2,1])
    with cl:
        st.subheader("📋 Project Overview")
        st.markdown("""
        Complete ML pipeline for **7-class HAR** on HAR70+ dataset.

        **Stages:** Load CSVs → EDA (6 plots) → Sliding-window feature
        engineering (134 features, 45K windows) → GroupShuffleSplit →
        StandardScaler → GroupKFold CV → SMOTE → RF · XGBoost · SVM →
        KMeans · DBSCAN → SHAP → 28 plots · saved models

        **Key design decisions:**
        - Subject-aware split prevents gait-identity leakage
        - SMOTE fixes 0.2% minority stair classes
        - SHAP explains per-prediction feature contributions
        - GroupKFold CV ensures honest generalisation estimate
        """)
    with cr:
        st.subheader("🎯 7 Activity Classes")
        for nm,pct,col,ico in [
            ("Walking","47.8%","#2196F3","🚶"),("Sitting","21.4%","#00BCD4","🪑"),
            ("Standing","18.5%","#FF9800","🧍"),("Lying","9.0%","#E91E63","🛌"),
            ("Shuffling","2.9%","#4CAF50","🦯"),("Stairs Up","0.2%","#FF5722","⬆️"),
            ("Stairs Down","0.2%","#9C27B0","⬇️"),]:
            bw=float(pct.replace("%",""))
            st.markdown(f"""<div style="margin:4px 0;display:flex;align-items:center;gap:10px">
                <span style="width:20px">{ico}</span>
                <span style="width:100px;font-size:.9rem">{nm}</span>
                <div style="flex:1;background:#f0f0f0;border-radius:4px;height:16px">
                    <div style="width:{min(bw*2,100)}%;background:{col};height:16px;border-radius:4px"></div>
                </div>
                <span style="width:45px;font-size:.85rem;color:#666">{pct}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🏗️ Architecture")
    st.code("""
  18 CSV files (501–518) → sort by subject+timestamp
  Sliding Window (100 samples / 2s, 50% overlap) → 45,157 windows, 134 features each
  GroupShuffleSplit 80/20 (subject-disjoint) → StandardScaler
  GroupKFold CV (5-fold, pre-SMOTE) → SMOTE (train only)
  ┌─────────────┬──────────────┬──────────────┐
  │ RandomForest│  XGBoost ✓   │  SVM (RBF)   │  Supervised
  └─────────────┴──────────────┴──────────────┘
  ┌──────────────────────┬─────────────────────┐
  │  KMeans K=7 (top-10) │  DBSCAN (top-10)    │  Unsupervised
  └──────────────────────┴─────────────────────┘
  SHAP TreeExplainer → beeswarm · per-class bar · waterfall
  → 28 plots · best_model.pkl · scaler.pkl
    """, language="")

# ════════════════════════════════════════════════════════════════
# LIVE PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Activity Prediction")
    if load_error:
        st.error("Model not loaded. Run `python har70_pipeline_FINAL.py` first.")
        st.stop()

    tab1, tab2 = st.tabs(["🎛️ Manual Sensor Input", "📤 Upload Window CSV"])

    # ── TAB 1: MANUAL INPUT ──────────────────────────────────────

    with tab1:
        st.markdown("""<div class="info-box">
        <b>Enter the mean acceleration values (in g) for both sensors.</b>
        The app builds a realistic 100-sample window around your values,
        extracts 134 features, and predicts the activity.
        Use <b>Load Preset</b> to auto-fill typical real-data values.
        </div>""", unsafe_allow_html=True)

        pc1, pc2, pc3 = st.columns([2,1,1])
        with pc1:
            preset = st.selectbox("Load Activity Preset",
                ["— enter manually —"]+list(ACTIVITY_PRESETS.keys()),
                format_func=lambda x: x if x=="— enter manually —"
                                      else f"{ACTIVITY_ICONS.get(x,'')} {x}")
        with pc2:
            noise = st.slider("Noise (g)", 0.005, 0.10, 0.02, 0.005,
                              help="Real sensors: ~0.01–0.03 g noise")
        with pc3:
            seed = st.number_input("Seed", 0, 9999, 42)

        d = (ACTIVITY_PRESETS[preset] if preset!="— enter manually —"
             else {c:0.0 for c in SENSOR_COLS})
        if preset != "— enter manually —":
            st.markdown(f"""<div class="ok-box">
            Preset loaded: <b>{ACTIVITY_ICONS.get(preset,'')} {preset}</b> —
            typical mean values from real HAR70+ recordings.
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── BACK SENSOR ────────────────────────────────────────
        st.markdown("### 🔵 Back Sensor (lower back)")
        st.caption("back_x = downward ≈1g lying on back | back_y = leftward ≈0 upright | back_z = forward ≈1g standing")
        b1,b2,b3 = st.columns(3)
        bx = b1.number_input("back_x  (down, g)", -2.0,2.0, float(round(d["back_x"],3)), 0.01, format="%.3f")
        by = b2.number_input("back_y  (left, g)", -2.0,2.0, float(round(d["back_y"],3)), 0.01, format="%.3f")
        bz = b3.number_input("back_z  (forward, g)", -2.0,2.0, float(round(d["back_z"],3)), 0.01, format="%.3f")
        bm = np.sqrt(bx**2+by**2+bz**2)
        ok = "✅ Plausible" if 0.7<bm<2.5 else "⚠️ Unusual (gravity=1g)"
        st.markdown(f"<div style='background:#1e2a3a;padding:.5rem 1rem;border-radius:8px;"
                    f"color:#90caf9;font-size:.85rem'>📐 Back magnitude: <b>{bm:.3f} g</b>"
                    f"&nbsp;&nbsp;{ok}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── THIGH SENSOR ───────────────────────────────────────
        st.markdown("### 🟢 Thigh Sensor (right front thigh)")
        st.caption("thigh_x = downward along thigh ≈0.9g sitting | thigh_y = rightward | thigh_z = backward along thigh")
        t1,t2,t3 = st.columns(3)
        tx = t1.number_input("thigh_x  (down, g)", -2.0,2.0, float(round(d["thigh_x"],3)), 0.01, format="%.3f")
        ty = t2.number_input("thigh_y  (right, g)", -2.0,2.0, float(round(d["thigh_y"],3)), 0.01, format="%.3f")
        tz = t3.number_input("thigh_z  (backward, g)", -2.0,2.0, float(round(d["thigh_z"],3)), 0.01, format="%.3f")
        tm_v = np.sqrt(tx**2+ty**2+tz**2)
        ok2  = "✅ Plausible" if 0.7<tm_v<2.5 else "⚠️ Unusual (gravity=1g)"
        st.markdown(f"<div style='background:#1a2e1a;padding:.5rem 1rem;border-radius:8px;"
                    f"color:#a5d6a7;font-size:.85rem'>📐 Thigh magnitude: <b>{tm_v:.3f} g</b>"
                    f"&nbsp;&nbsp;{ok2}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Reference table ─────────────────────────────────────
        with st.expander("📖 Reference — Typical values per activity"):
            ref = {nm: list(v.values()) for nm,v in ACTIVITY_PRESETS.items()}
            ref_df = pd.DataFrame(ref, index=SENSOR_COLS).T
            ref_df.index.name = "Activity"
            st.dataframe(ref_df.style.format("{:.3f}"), use_container_width=True)
            st.caption("Values are mean accelerations (g) from real HAR70+ recordings. "
                       "Dynamic activities (walking/stairs) show these as the DC offset; "
                       "the oscillation is added automatically by the app.")

        # ── PREDICT ─────────────────────────────────────────────
        if st.button("🚀 Predict Activity", type="primary", use_container_width=True):
            means = dict(back_x=bx,back_y=by,back_z=bz,
                         thigh_x=tx,thigh_y=ty,thigh_z=tz)
            win_df = build_window(means, noise, int(seed))
            pn, pr, cn, fd = predict(win_df)
            conf  = float(np.max(pr))*100
            color = ACTIVITY_COLORS.get(pn,"#2196F3")

            st.markdown(f"""<div class="pred-box"
                style="background:linear-gradient(135deg,{color},{color}99)">
                <h2>{ACTIVITY_ICONS.get(pn,'')} {pn}</h2>
                <h3>Confidence: {conf:.1f}%</h3></div>""",
                unsafe_allow_html=True)

            if conf >= 85:
                st.success(f"✅ High confidence — **{pn}** ({conf:.1f}%)")
            elif conf >= 60:
                st.warning(f"⚠️ Moderate confidence ({conf:.1f}%). "
                           "Try a different seed or adjust values slightly.")
            else:
                st.error(f"❌ Low confidence ({conf:.1f}%). "
                         "Values may be ambiguous. Check the reference table above.")

            cl2, cr2 = st.columns(2)

            with cl2:
                st.subheader("📈 Generated Signal (2 s)")
                fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,5),sharex=True)
                t_arr = np.linspace(0,2,WINDOW_SIZE)
                cols3 = ["#2196F3","#4CAF50","#FF5722"]
                for j,(cn2,lb) in enumerate(zip(["back_x","back_y","back_z"],["X","Y","Z"])):
                    ax1.plot(t_arr,win_df[cn2].values,color=cols3[j],label=f"Back-{lb}",lw=1.5)
                ax1.set_ylabel("Acc (g)"); ax1.legend(fontsize=8); ax1.set_title("Back Sensor")
                for j,(cn2,lb) in enumerate(zip(["thigh_x","thigh_y","thigh_z"],["X","Y","Z"])):
                    ax2.plot(t_arr,win_df[cn2].values,color=cols3[j],
                             label=f"Thigh-{lb}",lw=1.5,ls="--")
                ax2.set_ylabel("Acc (g)"); ax2.legend(fontsize=8)
                ax2.set_xlabel("Time (s)"); ax2.set_title("Thigh Sensor")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with cr2:
                st.subheader("📊 Class Probabilities")
                pdf = pd.DataFrame({"Activity":cn,"Prob (%)":pr*100}).sort_values("Prob (%)",ascending=True)
                fig,ax = plt.subplots(figsize=(8,5))
                bc = [ACTIVITY_COLORS.get(a,"#888") for a in pdf["Activity"]]
                bars = ax.barh(pdf["Activity"],pdf["Prob (%)"],color=bc,edgecolor="white",alpha=0.85)
                ax.set_xlabel("Probability (%)"); ax.set_xlim(0,110)
                ax.set_title("XGBoost Prediction Probabilities")
                for bar,val in zip(bars,pdf["Prob (%)"]):
                    if val>0.5:
                        ax.text(val+1,bar.get_y()+bar.get_height()/2,
                                f"{val:.1f}%",va="center",fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with st.expander("🔬 Extracted Features (key subset)"):
                kf = {k:v for k,v in fd.items()
                      if any(x in k for x in ["mean","std","rms","dom_freq","mag_corr","spec_entropy","zcr"])}
                st.dataframe(
                    pd.DataFrame(list(kf.items()),columns=["Feature","Value"])
                    .set_index("Feature").style.format({"Value":"{:.4f}"}),
                    height=300)

            with st.expander("🧪 Physics Check"):
                st.markdown(f"""
                | | Value | Expected |
                |---|---|---|
                | Back magnitude | {bm:.3f} g | 0.8–1.5 g |
                | Thigh magnitude | {tm_v:.3f} g | 0.7–1.4 g |
                | back\\_z (upright) | {bz:.3f} g | ≈1g standing/sitting, ≈0g lying |
                | back\\_x (horizontal) | {bx:.3f} g | ≈1g lying flat, ≈0g upright |
                | thigh\\_x (seat contact) | {tx:.3f} g | ≈0.85g sitting, ≈0.5g walking |
                """)

    # ── TAB 2: UPLOAD CSV ─────────────────────────────────────────
    with tab2:
        st.markdown("""<div class="info-box">
        Upload a CSV with ≥100 rows and columns:
        <b>back_x, back_y, back_z, thigh_x, thigh_y, thigh_z</b>.
        Extract any 100 consecutive rows from a real HAR70+ subject file.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Expected format (first 3 rows example):**")
        ex = pd.DataFrame({"back_x":[-0.999,-0.980,-0.950],"back_y":[-0.063,-0.079,-0.076],
                            "back_z":[0.141,0.141,0.141],"thigh_x":[-0.980,-0.961,-0.949],
                            "thigh_y":[-0.112,-0.122,-0.081],"thigh_z":[-0.048,-0.052,-0.067]})
        st.dataframe(ex, use_container_width=True)

        up = st.file_uploader("Upload window CSV", type=["csv"], key="t2")
        if up:
            try:
                df2 = pd.read_csv(up); df2.columns = df2.columns.str.strip()
                miss = [c for c in SENSOR_COLS if c not in df2.columns]
                if miss: st.error(f"Missing columns: {miss}")
                else:
                    w2 = df2[SENSOR_COLS].iloc[:WINDOW_SIZE].copy()
                    if len(w2)<WINDOW_SIZE:
                        while len(w2)<WINDOW_SIZE:
                            w2 = pd.concat([w2,w2],ignore_index=True)
                        w2 = w2.iloc[:WINDOW_SIZE]
                    pn,pr,cn,_ = predict(w2)
                    conf = float(np.max(pr))*100
                    color= ACTIVITY_COLORS.get(pn,"#2196F3")
                    st.markdown(f"""<div class="pred-box"
                        style="background:linear-gradient(135deg,{color},{color}99)">
                        <h2>{ACTIVITY_ICONS.get(pn,'')} {pn}</h2>
                        <h3>Confidence: {conf:.1f}%</h3></div>""",
                        unsafe_allow_html=True)
                    pdf2 = pd.DataFrame({"Activity":cn,"Prob (%)":pr*100}).sort_values("Prob (%)",ascending=False)
                    st.dataframe(pdf2.style.format({"Prob (%)":"{:.2f}"}),use_container_width=True)
                    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,5),sharex=True)
                    t2 = np.linspace(0,2,len(w2))
                    for cn2,lb in zip(["back_x","back_y","back_z"],["X","Y","Z"]):
                        ax1.plot(t2,w2[cn2].values,label=f"Back-{lb}",lw=1.2)
                    ax1.set_ylabel("Acc (g)"); ax1.legend(fontsize=8); ax1.set_title("Back Sensor")
                    for cn2,lb in zip(["thigh_x","thigh_y","thigh_z"],["X","Y","Z"]):
                        ax2.plot(t2,w2[cn2].values,label=f"Thigh-{lb}",lw=1.2,ls="--")
                    ax2.set_ylabel("Acc (g)"); ax2.legend(fontsize=8)
                    ax2.set_xlabel("Time (s)"); ax2.set_title("Thigh Sensor")
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Performance Results")
    st.subheader("🏆 Supervised Model Comparison")
    rdf = pd.DataFrame({
        "Model":["Random Forest","XGBoost","SVM (RBF)"],
        "Accuracy":[0.9616,0.9688,0.9444],
        "Weighted F1":[0.9582,0.9667,0.9539],
        "AUC (OvR)":[0.9979,0.9985,0.9979],
        "CV Mean F1":["0.9579","0.9645","—"],
        "CV Std":["0.0026","0.0013","—"],
    }).set_index("Model")
    def hl(s):
        try:
            n=pd.to_numeric(s,errors="coerce")
            return ["background-color:#c8e6c9;font-weight:bold" if v==n.max() else "" for v in n]
        except: return [""]*len(s)
    st.dataframe(rdf.style.apply(hl,subset=["Accuracy","Weighted F1","AUC (OvR)"]),
                 use_container_width=True)
    st.markdown("""<div class="ok-box">🏆 <b>Best Model: XGBoost</b> — 96.9% accuracy,
    F1=0.967, AUC=0.9985, CV=0.9645±0.0013.</div>""",unsafe_allow_html=True)

    st.subheader("📋 Per-Class F1 (XGBoost)")
    pc = pd.DataFrame({
        "Precision":[0.97,0.63,0.93,0.88,0.95,1.00,1.00],
        "Recall":   [0.98,0.42,0.68,0.70,0.96,1.00,1.00],
        "F1-Score": [0.98,0.51,0.79,0.78,0.96,1.00,1.00],
        "Support":  [4338,231,19,20,1676,1936,812],
    },index=["Walking","Shuffling","Stairs Up","Stairs Down","Standing","Sitting","Lying"])
    def cf1(v):
        if isinstance(v,float):
            if v>=0.95: return "background-color:#c8e6c9"
            if v>=0.70: return "background-color:#fff9c4"
            return "background-color:#ffcdd2"
        return ""
    st.dataframe(pc.style.applymap(cf1,subset=["Precision","Recall","F1-Score"])
                         .format({"Precision":"{:.2f}","Recall":"{:.2f}","F1-Score":"{:.2f}"}),
                 use_container_width=True)

    st.subheader("🔵 Unsupervised Clustering")
    st.dataframe(pd.DataFrame({
        "Method":["KMeans (K=7)","DBSCAN"],
        "Clusters":[7,23],"Silhouette":[0.161,0.428],
        "ARI":[0.473,0.557],"NMI":[0.657,"—"],"Noise %":["—","86.2%"],
    }).set_index("Method"),use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ALL PLOTS
# ════════════════════════════════════════════════════════════════
elif page == "🖼️ All Plots":
    st.title("🖼️ All 28 Visualisation Plots")
    pd_dir = f"{OUTPUT_DIR}/plots"
    pfiles = sorted(glob.glob(f"{pd_dir}/plot*.png"))
    if not pfiles:
        st.error("No plots found. Run `python har70_pipeline_FINAL.py` first.")
        st.stop()
    descs = {
        "plot01":"Class Distribution — bar and pie chart",
        "plot02":"Per-Subject Distribution — 18 subjects",
        "plot03":"Raw Signals — 2-sec windows per activity",
        "plot04":"Signal Boxplots — acceleration distributions",
        "plot05":"Correlation Heatmaps — per activity",
        "plot06":"Stats Heatmap — mean & std per axis",
        "plot07":"Feature Variance — top 30 features",
        "plot08":"t-SNE — 2D feature space projection",
        "plot09":"PCA Variance — explained variance curve",
        "plot10":"RF — confusion matrix + feature importance",
        "plot11":"RF ROC Curves — all 7 classes",
        "plot12":"RF Learning Curve — GroupKFold",
        "plot13":"XGBoost — confusion matrix + FI",
        "plot14":"XGBoost ROC Curves",
        "plot15":"SVM — confusion matrix + decision boundary",
        "plot16":"SVM ROC Curves",
        "plot17":"KMeans Elbow — K=2 to 11",
        "plot18":"KMeans — clusters vs true labels",
        "plot19":"Cluster-Activity mapping heatmap",
        "plot20":"DBSCAN — k-distance + clusters",
        "plot21":"Comparison Dashboard — all models",
        "plot22":"All Confusion Matrices side-by-side",
        "plot23":"Error Rates per class",
        "plot24":"Feature Importance RF vs XGBoost",
        "plot25":"Precision-Recall Curves (RF)",
        "plot26":"SHAP Beeswarm Summary",
        "plot27":"SHAP Per-Class Bar",
        "plot28":"SHAP Waterfall per class",
    }
    cats = {
        "📊 EDA (1–6)":pfiles[:6], "⚙️ Features (7–9)":pfiles[6:9],
        "🌲 RF (10–12)":pfiles[9:12], "⚡ XGBoost (13–14)":pfiles[12:14],
        "🔷 SVM (15–16)":pfiles[14:16], "🔵 Clustering (17–20)":pfiles[16:20],
        "📈 Comparison (21–25)":pfiles[20:25], "💡 SHAP (26–28)":pfiles[25:],
    }
    cat = st.selectbox("Category", list(cats.keys()))
    for f in cats[cat]:
        fn = os.path.basename(f); key = fn[:6]
        nm = fn[4:6].lstrip("0") or "0"
        with st.expander(f"Plot {nm} — {descs.get(key,'')}", expanded=True):
            st.image(f, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# SHAP
# ════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainability — XGBoost")
    st.markdown("""<div class="info-box"><b>SHAP</b> explains <i>why</i> the model
    makes each prediction. Positive SHAP = pushes toward class.
    Negative SHAP = pushes away.</div>""",unsafe_allow_html=True)
    pd_dir = f"{OUTPUT_DIR}/plots"
    for fn,title,desc in [
        ("plot26_shap_summary.png","SHAP Beeswarm Summary",
         "Each dot = one test sample. X = SHAP value. Color = feature value."),
        ("plot27_shap_per_class_bar.png","SHAP Per-Class Bar",
         "Top-10 features by mean |SHAP| for each activity class."),
        ("plot28_shap_waterfall_per_class.png","SHAP Waterfall per Class",
         "One correct prediction per class — red=up, blue=down."),
    ]:
        fp = f"{pd_dir}/{fn}"
        if os.path.exists(fp):
            st.subheader(f"📊 {title}"); st.markdown(f"*{desc}*")
            st.image(fp,use_container_width=True); st.divider()
        else: st.warning(f"{fn} not found.")
    st.subheader("💡 Key Insights")
    c1,c2 = st.columns(2)
    for col,items in zip([c1,c2],[
        [("🛌 Lying","back_x_mean≈1g — sensor lying flat"),
         ("🪑 Sitting","thigh_x_mean≈0.85g — thigh horizontal"),
         ("🧍 Standing","back_z_mean≈1g — upright, low std"),
         ("🚶 Walking","dom_freq≈1.8Hz, high RMS both sensors")],
        [("🦯 Shuffling","dom_freq≈1.0Hz, lower thigh_mag than walking"),
         ("⬆️ Stairs Up","thigh_mag_mean high, high energy + phase shift"),
         ("⬇️ Stairs Down","back_y_std high, low thigh_yz_corr"),
         ("🔑 Cross-sensor","mag_corr separates dynamic from static")]]):
        for nm,ins in items:
            col.markdown(f'<div class="info-box"><b>{nm}</b><br>{ins}</div>',
                         unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# BATCH CSV
# ════════════════════════════════════════════════════════════════
elif page == "📁 Batch CSV Prediction":
    st.title("📁 Batch Prediction from Subject CSV")
    if load_error:
        st.error("Model not loaded."); st.stop()
    st.markdown("""<div class="info-box">Upload any HAR70+ subject CSV (e.g. <b>501.csv</b>).
    The app slides a 100-sample window with 50% overlap across the whole recording.</div>""",
    unsafe_allow_html=True)
    up_b = st.file_uploader("Upload HAR70+ subject CSV",type=["csv"],key="batch")
    if up_b:
        dfb = pd.read_csv(up_b); dfb.columns = dfb.columns.str.strip()
        miss = [c for c in SENSOR_COLS if c not in dfb.columns]
        if miss: st.error(f"Missing: {miss}"); st.stop()
        nr = len(dfb); nw = (nr-WINDOW_SIZE)//50+1
        st.info(f"{nr:,} rows → {nw:,} windows available")
        mw = st.slider("Max windows",50,min(nw,2000),500,50)
        if st.button("▶️ Run Batch Prediction",type="primary"):
            preds,confs = [],[]
            prog = st.progress(0); stat = st.empty()
            step = max(1,nw//mw)
            for i,start in enumerate(range(0,nr-WINDOW_SIZE,50*step)):
                if i>=mw: break
                w = dfb[SENSOR_COLS].iloc[start:start+WINDOW_SIZE]
                if len(w)<WINDOW_SIZE: break
                pn,pr,_,_ = predict(w); preds.append(pn); confs.append(float(np.max(pr))*100)
                if i%50==0: prog.progress(min(i/mw,1.0)); stat.text(f"Window {i+1}/{mw}…")
            prog.progress(1.0); stat.text("✅ Done!")
            st.subheader("Prediction Timeline")
            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,6),sharex=True)
            for i,(p,c) in enumerate(zip(preds,confs)):
                ax1.bar(i,1,color=ACTIVITY_COLORS.get(p,"#888"),width=1,align="edge",alpha=0.85)
            from matplotlib.patches import Patch
            seen = list(dict.fromkeys(preds))
            ax1.legend(handles=[Patch(color=ACTIVITY_COLORS.get(a,"#888"),label=a) for a in seen],
                       fontsize=8,loc="upper right")
            ax1.set_yticks([]); ax1.set_title("Activity Prediction Timeline")
            ax2.plot(confs,color="#2196F3",lw=1,alpha=0.7)
            ax2.fill_between(range(len(confs)),confs,alpha=0.2,color="#2196F3")
            ax2.axhline(80,color="orange",ls="--",lw=1,label="80% threshold")
            ax2.set_ylim(0,105); ax2.set_title("Confidence %")
            ax2.set_xlabel("Window Index"); ax2.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            cl,cr = st.columns(2)
            from collections import Counter
            cnt = Counter(preds)
            with cl:
                fig,ax = plt.subplots(figsize=(6,5))
                ax.pie(cnt.values(),labels=cnt.keys(),autopct="%1.1f%%",
                       colors=[ACTIVITY_COLORS.get(a,"#888") for a in cnt],startangle=140)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with cr:
                sm = pd.DataFrame({
                    "Activity":list(cnt.keys()),"Windows":list(cnt.values()),
                    "Pct (%)": [v/len(preds)*100 for v in cnt.values()],
                    "Avg Conf":[np.mean([confs[i] for i,p in enumerate(preds) if p==a]) for a in cnt]
                }).sort_values("Windows",ascending=False)
                st.dataframe(sm.style.format({"Pct (%)":"{:.1f}","Avg Conf":"{:.1f}"}),
                             use_container_width=True)
                st.metric("Avg Confidence",f"{np.mean(confs):.1f}%")
                st.metric("Low Conf (<70%)",sum(1 for c in confs if c<70))

st.markdown("""---
<div style="text-align:center;color:#888;font-size:.8rem;padding:1rem 0">
HAR70+ · CS-245 Machine Learning · NUST SEECS · Spring 2026 ·
<a href="https://archive.ics.uci.edu/dataset/780/har70">UCI HAR70+ Dataset</a>
</div>""", unsafe_allow_html=True)