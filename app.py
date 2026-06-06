import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, precision_recall_curve,
    average_precision_score, classification_report,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Higgs Boson Classifier",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# GLOBAL PLOT STYLE
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#0b0f1e',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#1f2d45',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e2d42',
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
    'font.family':      'monospace',
    'figure.dpi':       110,
})

SIG  = '#22d3ee'
BG   = '#3b82f6'
ACC  = '#f472b6'
WARN = '#fb923c'

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0b0f1e;
    color: #cbd5e1;
}
.stApp { background-color: #0b0f1e; }

/* Header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #0b1a30 50%, #0f172a 100%);
    border: 1px solid #22d3ee22;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #22d3ee15, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #22d3ee;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-sub {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 6px;
    letter-spacing: 1px;
}

/* Metric cards */
.card {
    background: linear-gradient(135deg, #111827, #0f1e35);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.card-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #22d3ee;
}
.card-label {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 4px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #22d3ee;
    border-left: 3px solid #22d3ee;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}

/* Upload zone */
.upload-zone {
    background: #0f1e35;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    border-radius: 7px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0f2744 !important;
    color: #22d3ee !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0891b2, #1d4ed8);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 12px 0;
    width: 100%;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #22d3ee30;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080c18;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #22d3ee;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Info / warning boxes */
.info-box {
    background: #0f2744;
    border: 1px solid #22d3ee33;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 8px 0;
}
.warn-box {
    background: #1c1207;
    border: 1px solid #fb923c44;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 8px 0;
}

/* Badge */
.badge {
    display: inline-block;
    background: #0f2744;
    border: 1px solid #22d3ee44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: #22d3ee;
    margin: 2px;
}

div[data-testid="stSidebar"] * { color: #94a3b8 !important; }
div[data-testid="stSidebar"] h3 { color: #22d3ee !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
for key in ['trained', 'state', 'train_df', 'test_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚛️ HIGGS BOSON")
    st.markdown("### CLASSIFIER")
    st.markdown("---")

    st.markdown("### 📂 UPLOAD DATA")
    train_file = st.file_uploader("training.csv", type="csv", key="train_up",
                                   help="CSV with Label and Weight columns")
    test_file  = st.file_uploader("test.csv (optional)", type="csv", key="test_up")

    st.markdown("---")
    st.markdown("### ⚙️ SETTINGS")
    test_size    = st.slider("Validation split", 0.10, 0.30, 0.20, 0.05)
    shap_samples = st.slider("SHAP sample size", 200, 1000, 500, 100)

    st.markdown("### 🤖 MODELS")
    use_lr  = st.checkbox("Logistic Regression", value=True)
    use_xgb = st.checkbox("XGBoost",             value=True)
    use_lgb = st.checkbox("LightGBM",            value=True)

    st.markdown("---")
    train_btn = st.button("🚀  TRAIN MODELS")

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.65rem; color:#334155; line-height:1.8'>
⚛️ CERN Higgs Boson ML Challenge<br>
250K events · 30 features<br>
Signal vs Background classification<br><br>
Built with XGBoost · LightGBM · SHAP
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def parse_csv(file_bytes, name):
    return pd.read_csv(io.BytesIO(file_bytes))

if train_file:
    train_df = parse_csv(train_file.getvalue(), "train")
    st.session_state['train_df'] = train_df
else:
    train_df = st.session_state.get('train_df')

if test_file:
    test_df = parse_csv(test_file.getvalue(), "test")
    st.session_state['test_df'] = test_df
else:
    test_df = st.session_state.get('test_df')

# ══════════════════════════════════════════════════════════
# TRAIN FUNCTION
# ══════════════════════════════════════════════════════════
def run_training(df, test_size, use_lr, use_xgb, use_lgb):
    X = df.drop(columns=['Label', 'Weight'])
    y = df['Label'].map({'b': 0, 's': 1})
    w = df['Weight']

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=test_size, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_val_sc    = scaler.transform(X_val)

    results = {}

    if use_lr:
        with st.spinner("Training Logistic Regression..."):
            lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
            lr.fit(X_train_sc, y_train, sample_weight=w_train)
            proba = lr.predict_proba(X_val_sc)[:, 1]
            results['Logistic Regression'] = {
                'model': lr, 'proba': proba, 'type': 'linear',
                'auc':   roc_auc_score(y_val, proba, sample_weight=w_val),
                'acc':   accuracy_score(y_val, lr.predict(X_val_sc)),
            }

    if use_xgb:
        with st.spinner("Training XGBoost..."):
            xgb = XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='auc', random_state=42,
                verbosity=0, n_jobs=-1
            )
            xgb.fit(X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_val, y_val)], verbose=False)
            proba = xgb.predict_proba(X_val)[:, 1]
            results['XGBoost'] = {
                'model': xgb, 'proba': proba, 'type': 'tree',
                'auc':   roc_auc_score(y_val, proba, sample_weight=w_val),
                'acc':   accuracy_score(y_val, xgb.predict(X_val)),
            }

    if use_lgb:
        with st.spinner("Training LightGBM..."):
            lgb = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1
            )
            lgb.fit(X_train, y_train, sample_weight=w_train)
            proba = lgb.predict_proba(X_val)[:, 1]
            results['LightGBM'] = {
                'model': lgb, 'proba': proba, 'type': 'tree',
                'auc':   roc_auc_score(y_val, proba, sample_weight=w_val),
                'acc':   accuracy_score(y_val, lgb.predict(X_val)),
            }

    return {
        'results':       results,
        'X_train':       X_train,
        'X_val':         X_val,
        'y_train':       y_train,
        'y_val':         y_val,
        'w_train':       w_train,
        'w_val':         w_val,
        'scaler':        scaler,
        'feature_names': X.columns.tolist(),
    }

# Trigger training
if train_btn:
    if train_df is None:
        st.sidebar.error("Upload training.csv first.")
    elif not any([use_lr, use_xgb, use_lgb]):
        st.sidebar.error("Select at least one model.")
    else:
        state = run_training(train_df, test_size, use_lr, use_xgb, use_lgb)
        st.session_state['state']   = state
        st.session_state['trained'] = True

state = st.session_state.get('state')

# ══════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">⚛️ Higgs Boson Signal Classifier</div>
  <div class="hero-sub">CERN ML CHALLENGE &nbsp;·&nbsp; SIGNAL vs BACKGROUND &nbsp;·&nbsp; PRODUCTION ML PIPELINE</div>
  <div style="margin-top:14px">
    <span class="badge">XGBoost</span>
    <span class="badge">LightGBM</span>
    <span class="badge">Logistic Regression</span>
    <span class="badge">SHAP</span>
    <span class="badge">ROC-AUC</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LANDING — no data yet
# ══════════════════════════════════════════════════════════
if train_df is None:
    st.markdown("""
<div class="info-box">
👈 <strong>Upload your <code>training.csv</code> in the sidebar to begin.</strong><br><br>
The file must contain:<br>
&nbsp;&nbsp;• 30 physics features (DER_*, PRI_*)<br>
&nbsp;&nbsp;• <code>Label</code> column — <code>s</code> (signal) or <code>b</code> (background)<br>
&nbsp;&nbsp;• <code>Weight</code> column — event importance weight<br><br>
Dataset: <strong>CERN Higgs Boson ML Challenge</strong> (Kaggle) · 250,000 events
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="card">
  <div class="card-val">0.928</div>
  <div class="card-label">XGBoost ROC-AUC</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="card">
  <div class="card-val">250K</div>
  <div class="card-label">Training Events</div>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div class="card">
  <div class="card-val">30</div>
  <div class="card-label">Physics Features</div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  EDA",
    "🏋️  Train & Compare",
    "📈  Metrics",
    "🔍  SHAP",
    "🎯  Predict",
])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — EDA                                        ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    features = [c for c in train_df.columns if c not in ['Label', 'Weight']]
    sig_df   = train_df[train_df['Label'] == 's']
    bkg_df   = train_df[train_df['Label'] == 'b']

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{len(train_df):,}",              "Total Events"),
        (f"{len(features)}",                "Features"),
        (f"{(train_df['Label']=='s').sum():,}", "Signal (s)"),
        (f"{(train_df['Label']=='b').sum():,}", "Background (b)"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], cards):
        col.markdown(f'<div class="card"><div class="card-val">{val}</div><div class="card-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Class Balance</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        counts = train_df['Label'].value_counts()
        fig = go.Figure(go.Pie(
            values=[counts.get('b', 0), counts.get('s', 0)],
            labels=['Background', 'Signal'],
            hole=0.6,
            marker_colors=[BG, SIG],
            textfont_size=11,
        ))
        fig.update_layout(
            paper_bgcolor='#0b0f1e', plot_bgcolor='#0b0f1e',
            font_color='#94a3b8', showlegend=True,
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            legend=dict(bgcolor='#111827')
        )
        fig.add_annotation(text=f"<b>{len(train_df):,}</b><br>events",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        weighted = train_df.groupby('Label')['Weight'].sum().reset_index()
        weighted['Name'] = weighted['Label'].map({'s': 'Signal', 'b': 'Background'})
        fig2 = px.bar(weighted, x='Name', y='Weight',
                      color='Name',
                      color_discrete_map={'Signal': SIG, 'Background': BG},
                      text=weighted['Weight'].round(1))
        fig2.update_layout(
            paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
            font_color='#94a3b8', showlegend=False,
            margin=dict(t=10, b=10), height=280,
            xaxis_title='', yaxis_title='Total Weight'
        )
        fig2.update_traces(textposition='outside', textfont_color='#e2e8f0')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Feature Distribution</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select a feature:", features)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=bkg_df[selected], name='Background',
                                nbinsx=80, opacity=0.65, marker_color=BG))
    fig3.add_trace(go.Histogram(x=sig_df[selected], name='Signal',
                                nbinsx=80, opacity=0.65, marker_color=SIG))
    fig3.update_layout(
        barmode='overlay',
        paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
        font_color='#94a3b8', height=320,
        legend=dict(bgcolor='#111827'),
        xaxis_title=selected, yaxis_title='Count',
        margin=dict(t=10, b=30)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap (Top 15 Features)</div>', unsafe_allow_html=True)
    corr = train_df[features[:15]].corr()
    fig4, ax4 = plt.subplots(figsize=(13, 7))
    sns.heatmap(corr, ax=ax4, cmap='coolwarm', center=0,
                annot=False, linewidths=0.3,
                cbar_kws={'shrink': 0.7})
    ax4.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown('<div class="section-header">Dataset Statistics</div>', unsafe_allow_html=True)
    st.dataframe(train_df[features].describe().T.round(4), use_container_width=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — TRAIN & COMPARE                            ║
# ╚══════════════════════════════════════════════════════╝
with tab2:
    if state is None:
        st.markdown('<div class="warn-box">👈 Click <strong>🚀 TRAIN MODELS</strong> in the sidebar to begin.</div>',
                    unsafe_allow_html=True)
    else:
        results = state['results']
        y_val   = state['y_val']
        w_val   = state['w_val']

        # Summary table
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
        rows = sorted([
            {'Model': k, 'ROC-AUC': round(v['auc'], 4), 'Accuracy': round(v['acc'], 4)}
            for k, v in results.items()
        ], key=lambda x: x['ROC-AUC'], reverse=True)
        df_res = pd.DataFrame(rows)

        # Trophy for best
        best = df_res.iloc[0]['Model']
        st.markdown(f"**🏆 Best Model: `{best}`  —  AUC = {df_res.iloc[0]['ROC-AUC']:.4f}**")
        st.dataframe(
            df_res.style.highlight_max(subset=['ROC-AUC', 'Accuracy'],
                                       color='#0f2744').format({'ROC-AUC': '{:.4f}', 'Accuracy': '{:.4f}'}),
            use_container_width=True, hide_index=True
        )

        # ROC curves
        st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
        CMAP = {'Logistic Regression': '#64748b', 'XGBoost': SIG, 'LightGBM': ACC}
        fig_roc = go.Figure()
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_val, r['proba'], sample_weight=w_val)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f"{name}  AUC={r['auc']:.4f}",
                line=dict(color=CMAP.get(name, '#fff'), width=2.5)
            ))
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                          line=dict(color='#1e3a5f', dash='dash'))
        fig_roc.update_layout(
            paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
            font_color='#94a3b8', height=400,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(bgcolor='#111827'),
            margin=dict(t=10, b=40)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance (tree models)
        tree_models = {k: v for k, v in results.items() if v['type'] == 'tree'}
        if tree_models:
            st.markdown('<div class="section-header">Feature Importance (Top 15)</div>', unsafe_allow_html=True)
            cols_fi = st.columns(len(tree_models))
            for col, (name, r) in zip(cols_fi, tree_models.items()):
                imp = pd.DataFrame({
                    'Feature':    state['feature_names'],
                    'Importance': r['model'].feature_importances_
                }).nlargest(15, 'Importance').sort_values('Importance')
                fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h',
                                 color='Importance',
                                 color_continuous_scale=['#1e3a5f', SIG])
                fig_imp.update_layout(
                    title=dict(text=name, font=dict(color=SIG, size=12)),
                    paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
                    font_color='#94a3b8', coloraxis_showscale=False,
                    margin=dict(t=40, b=10, l=10, r=10), height=420
                )
                col.plotly_chart(fig_imp, use_container_width=True)

        # Save best model
        best_model  = results[best]['model']
        best_scaler = state['scaler']
        buf_model   = io.BytesIO()
        buf_scaler  = io.BytesIO()
        joblib.dump(best_model,  buf_model);  buf_model.seek(0)
        joblib.dump(best_scaler, buf_scaler); buf_scaler.seek(0)

        st.markdown('<div class="section-header">Download Trained Models</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        dl1.download_button("⬇️ Download best_model.pkl", buf_model,  "best_model.pkl",  "application/octet-stream")
        dl2.download_button("⬇️ Download scaler.pkl",     buf_scaler, "scaler.pkl",      "application/octet-stream")

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — METRICS DEEP DIVE                          ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    if state is None:
        st.markdown('<div class="warn-box">Train models first.</div>', unsafe_allow_html=True)
    else:
        results = state['results']
        y_val   = state['y_val']
        w_val   = state['w_val']

        model_sel = st.selectbox("Select model:", list(results.keys()))
        r         = results[model_sel]
        proba     = r['proba']

        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
        y_pred    = (proba >= threshold).astype(int)

        # Metric row
        p    = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val,    y_pred, zero_division=0)
        f1   = f1_score(y_val,        y_pred, zero_division=0)
        ap   = average_precision_score(y_val, proba, sample_weight=w_val)

        m1, m2, m3, m4 = st.columns(4)
        for col, val, label in zip(
            [m1, m2, m3, m4],
            [r['auc'], p, rec, f1],
            ['ROC-AUC', f'Precision @{threshold}', f'Recall @{threshold}', f'F1 @{threshold}']
        ):
            col.markdown(f'<div class="card"><div class="card-val">{val:.3f}</div><div class="card-label">{label}</div></div>',
                         unsafe_allow_html=True)

        st.markdown("")
        col_cm, col_pr = st.columns(2)

        with col_cm:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm  = confusion_matrix(y_val, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Background', 'Signal'],
                        yticklabels=['Background', 'Signal'],
                        ax=ax_cm, cbar=False,
                        annot_kws={'size': 14, 'color': 'white'})
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            plt.tight_layout()
            st.pyplot(fig_cm); plt.close()

        with col_pr:
            st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
            prec_curve, rec_curve, _ = precision_recall_curve(y_val, proba, sample_weight=w_val)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec_curve, y=prec_curve, mode='lines',
                line=dict(color=SIG, width=2),
                fill='tozeroy', fillcolor='rgba(34,211,238,0.08)',
                name=f'AP={ap:.3f}'
            ))
            fig_pr.update_layout(
                paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
                font_color='#94a3b8', height=300,
                xaxis_title='Recall', yaxis_title='Precision',
                margin=dict(t=10, b=30), legend=dict(bgcolor='#111827')
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # Classification report
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_val, y_pred,
                                       target_names=['Background', 'Signal'],
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        # Score distribution
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=proba[y_val == 0], name='Background',
            nbinsx=80, opacity=0.7, marker_color=BG))
        fig_score.add_trace(go.Histogram(
            x=proba[y_val == 1], name='Signal',
            nbinsx=80, opacity=0.7, marker_color=SIG))
        fig_score.add_vline(x=threshold, line_dash='dash',
                            line_color=WARN, annotation_text=f'threshold={threshold}')
        fig_score.update_layout(
            barmode='overlay',
            paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
            font_color='#94a3b8', height=320,
            xaxis_title='Predicted Probability', yaxis_title='Count',
            legend=dict(bgcolor='#111827'), margin=dict(t=10, b=30)
        )
        st.plotly_chart(fig_score, use_container_width=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 4 — SHAP                                       ║
# ╚══════════════════════════════════════════════════════╝
with tab4:
    if state is None:
        st.markdown('<div class="warn-box">Train models first.</div>', unsafe_allow_html=True)
    else:
        results = state['results']
        tree_m  = {k: v for k, v in results.items() if v['type'] == 'tree'}

        if not tree_m:
            st.markdown('<div class="warn-box">Enable XGBoost or LightGBM for SHAP analysis.</div>',
                        unsafe_allow_html=True)
        else:
            shap_model_sel = st.selectbox("Model for SHAP:", list(tree_m.keys()))
            shap_model     = tree_m[shap_model_sel]['model']

            X_shap = state['X_val'].sample(
                min(shap_samples, len(state['X_val'])), random_state=42
            )

            with st.spinner(f"Computing SHAP values on {len(X_shap)} samples..."):
                explainer   = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(X_shap)

            plot_type = st.radio("Plot type:", ["Beeswarm", "Bar (Mean |SHAP|)"], horizontal=True)

            fig_sh, ax_sh = plt.subplots(figsize=(11, 7))
            if plot_type == "Beeswarm":
                shap.summary_plot(shap_values, X_shap, show=False, plot_size=None)
            else:
                shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False, plot_size=None)
            plt.tight_layout()
            st.pyplot(fig_sh); plt.close()

            # SHAP table
            mean_shap = pd.DataFrame({
                'Feature':     state['feature_names'],
                'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
            }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)

            st.markdown('<div class="section-header">Top Features by SHAP</div>', unsafe_allow_html=True)
            st.dataframe(mean_shap.head(15).style.background_gradient(
                subset=['Mean |SHAP|'], cmap='Blues'
            ), use_container_width=True, hide_index=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 5 — PREDICT                                    ║
# ╚══════════════════════════════════════════════════════╝
with tab5:
    if state is None:
        st.markdown('<div class="warn-box">Train models first.</div>', unsafe_allow_html=True)
    else:
        results = state['results']
        st.markdown('<div class="section-header">Run Predictions on New Events</div>',
                    unsafe_allow_html=True)

        pred_src = st.radio("Data source:",
                            ["Upload new CSV", "Use uploaded test.csv"],
                            horizontal=True)
        model_choice = st.selectbox("Use model:", list(results.keys()))

        pred_df = None
        if pred_src == "Upload new CSV":
            pred_file = st.file_uploader("Upload events CSV", type="csv", key="pred_up")
            if pred_file:
                pred_df = pd.read_csv(pred_file)
        else:
            if test_df is not None:
                pred_df = test_df.copy()
            else:
                st.markdown('<div class="warn-box">No test.csv uploaded. Use the sidebar.</div>',
                            unsafe_allow_html=True)

        if pred_df is not None:
            feat_cols  = [c for c in state['feature_names'] if c in pred_df.columns]
            X_pred_raw = pred_df[feat_cols]

            r = results[model_choice]
            if r['type'] == 'linear':
                X_pred_in = state['scaler'].transform(X_pred_raw)
            else:
                X_pred_in = X_pred_raw.values

            probas = r['model'].predict_proba(X_pred_in)[:, 1]
            preds  = (probas >= 0.5).astype(int)

            out_df = pd.DataFrame({
                'EventId':          range(len(pred_df)),
                'Signal_Prob':      probas.round(5),
                'Prediction':       np.where(preds == 1, '🟢 Signal', '🔴 Background'),
            })

            # Summary stats
            p1, p2, p3 = st.columns(3)
            p1.markdown(f'<div class="card"><div class="card-val">{len(out_df):,}</div><div class="card-label">Events Scored</div></div>', unsafe_allow_html=True)
            p2.markdown(f'<div class="card"><div class="card-val">{preds.sum():,}</div><div class="card-label">Signal Predicted</div></div>', unsafe_allow_html=True)
            p3.markdown(f'<div class="card"><div class="card-val">{(1-preds).sum():,}</div><div class="card-label">Background Predicted</div></div>', unsafe_allow_html=True)

            st.markdown("")
            st.dataframe(out_df.head(100), use_container_width=True, hide_index=True)

            # Probability histogram
            fig_ph = go.Figure(go.Histogram(
                x=probas, nbinsx=80,
                marker_color=SIG, opacity=0.85
            ))
            fig_ph.add_vline(x=0.5, line_dash='dash', line_color=WARN,
                             annotation_text='threshold=0.5')
            fig_ph.update_layout(
                paper_bgcolor='#0b0f1e', plot_bgcolor='#111827',
                font_color='#94a3b8', height=300,
                xaxis_title='Signal Probability', yaxis_title='Count',
                margin=dict(t=10, b=30)
            )
            st.plotly_chart(fig_ph, use_container_width=True)

            # Download
            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️  Download Predictions CSV",
                csv_bytes, "predictions.csv", "text/csv"
            )
