# ── CONFIGURATION ────────────────────────────────────────────────
DATA_DIR                = "./har70plus"   # folder with 501.csv … 518.csv
OUTPUT_DIR              = "./har70_output"
WINDOW_SIZE             = 100             # 2 seconds @ 50 Hz
STEP_SIZE               = 50             # 50% overlap
RANDOM_STATE            = 42
N_CV_FOLDS              = 5
USE_SUBJECT_AWARE_SPLIT = True
USE_SMOTE               = True
# ────────────────────────────────────────────────────────────────

import os, warnings, glob, time
warnings.filterwarnings("ignore")
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import skew, kurtosis
from scipy.signal import welch

from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, f1_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import shap

# ── Global style ─────────────────────────────────────────────────
PALETTE = ["#2196F3","#4CAF50","#FF5722","#9C27B0",
           "#FF9800","#00BCD4","#E91E63"]
ACTIVITY_NAMES = {
    1:"Walking", 3:"Shuffling", 4:"Stairs Up",
    5:"Stairs Down", 6:"Standing", 7:"Sitting", 8:"Lying"
}
SENSOR_COLS = ["back_x","back_y","back_z","thigh_x","thigh_y","thigh_z"]
plt.rcParams.update({
    "figure.dpi":150, "font.size":10,
    "axes.spines.top":False, "axes.spines.right":False
})


# ════════════════════════════════════════════════════════════════
#  VISUALIZATION MANAGER
# ════════════════════════════════════════════════════════════════
class VisualizationManager:
    """All 28 plots centralised here. Keeps main() clean."""

    def __init__(self, out, palette, activity_names, class_names_enc):
        self.out = out
        self.P   = palette
        self.AN  = activity_names
        self.CN  = class_names_enc
        self.NC  = len(class_names_enc)

    def _save(self, name):
        path = f"{self.out}/plots/{name}"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"    → {path}")

    def _grid(self):
        nc = 4
        nr = (self.NC + nc - 1) // nc
        return nr, nc

    # ── EDA ──────────────────────────────────────────────────────
    def plot01_class_distribution(self, raw, LABELS):
        counts = raw["label"].value_counts().reindex(LABELS)
        names  = [self.AN[l] for l in counts.index]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("HAR70+ — Class Distribution (All Subjects)",
                     fontsize=14, fontweight="bold")
        axes[0].barh(names, counts.values, color=self.P, edgecolor="white")
        axes[0].set_xlabel("Sample Count")
        axes[0].set_title("Sample Count per Activity")
        for i, v in enumerate(counts.values):
            axes[0].text(v+500, i, f"{v:,}", va="center", fontsize=9)
        axes[1].pie(counts.values, labels=names, colors=self.P,
                    autopct="%1.1f%%", startangle=140, pctdistance=0.8)
        axes[1].set_title("Activity Proportions")
        plt.tight_layout()
        self._save("plot01_class_distribution.png")

    def plot02_per_subject(self, raw):
        subj_act = raw.groupby(["subject","label"]).size().unstack(fill_value=0)
        subj_act.columns = [self.AN[c] for c in subj_act.columns]
        fig, ax = plt.subplots(figsize=(16, 7))
        subj_act.plot(kind="bar", ax=ax, color=self.P, edgecolor="white", width=0.8)
        ax.set_title("Per-Subject Activity Sample Counts",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Subject ID"); ax.set_ylabel("Sample Count")
        ax.legend(bbox_to_anchor=(1.01,1), fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        self._save("plot02_per_subject_distribution.png")

    def plot03_raw_signals(self, raw, LABELS):
        fig, axes = plt.subplots(len(LABELS), 2, figsize=(16, 22))
        fig.suptitle("Raw Accelerometer Signals per Activity (2-sec window @ 50 Hz)",
                     fontsize=14, fontweight="bold")
        for row, act in enumerate(LABELS):
            subset = raw[raw["label"]==act].iloc[:100]
            t = np.arange(len(subset)) / 50
            for col, lbl in zip(["back_x","back_y","back_z"],
                                 ["Back-X","Back-Y","Back-Z"]):
                axes[row,0].plot(t, subset[col].values, alpha=0.8, label=lbl)
            for col, lbl in zip(["thigh_x","thigh_y","thigh_z"],
                                 ["Thigh-X","Thigh-Y","Thigh-Z"]):
                axes[row,1].plot(t, subset[col].values, alpha=0.8,
                                 label=lbl, linestyle="--")
            axes[row,0].set_ylabel(self.AN[act], fontsize=9, fontweight="bold")
            if row == 0:
                axes[row,0].set_title("Back Sensor", fontsize=10)
                axes[row,1].set_title("Thigh Sensor", fontsize=10)
                axes[row,0].legend(fontsize=7, ncol=3)
                axes[row,1].legend(fontsize=7, ncol=3)
        for ax in axes[-1,:]:
            ax.set_xlabel("Time (s)")
        plt.tight_layout()
        self._save("plot03_raw_signals.png")

    def plot04_boxplots(self, raw, LABELS):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Sensor Acceleration Distribution per Activity",
                     fontsize=14, fontweight="bold")
        CLASS_NAMES = [self.AN[l] for l in LABELS]
        for idx, col in enumerate(SENSOR_COLS):
            ax = axes[idx//3][idx%3]
            data = [raw[raw["label"]==act][col].values for act in LABELS]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=2))
            for patch, color in zip(bp["boxes"], self.P):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
            ax.set_title(col.replace("_"," ").title())
            ax.set_ylabel("Acceleration (g)")
        plt.tight_layout()
        self._save("plot04_signal_boxplots.png")

    def plot05_correlation_heatmaps(self, raw, LABELS):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle("Inter-Sensor Correlation per Activity",
                     fontsize=14, fontweight="bold")
        for idx, act in enumerate(LABELS):
            ax = axes[idx//4][idx%4]
            subset = raw[raw["label"]==act][SENSOR_COLS].sample(
                min(3000, (raw["label"]==act).sum()), random_state=RANDOM_STATE)
            sns.heatmap(subset.corr(), ax=ax, annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0, square=True, cbar=False,
                        annot_kws={"size":7},
                        xticklabels=[c.replace("_"," ") for c in SENSOR_COLS],
                        yticklabels=[c.replace("_"," ") for c in SENSOR_COLS])
            ax.set_title(self.AN[act], fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
        axes[1][3].axis("off")
        plt.tight_layout()
        self._save("plot05_correlation_heatmaps.png")

    def plot06_stats_heatmap(self, raw, LABELS):
        rows = []
        for act in LABELS:
            sub = raw[raw["label"]==act][SENSOR_COLS]
            row = {"Activity": self.AN[act]}
            for col in SENSOR_COLS:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"]  = sub[col].std()
            rows.append(row)
        stats_df = pd.DataFrame(rows).set_index("Activity")
        fig, ax = plt.subplots(figsize=(18, 5))
        sns.heatmap(stats_df, annot=True, fmt=".3f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax,
                    cbar_kws={"label":"Value"})
        ax.set_title("Mean & Std of Each Sensor Axis per Activity",
                     fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        self._save("plot06_stats_heatmap.png")

    def plot07_feature_variance(self, feat_df):
        feat_var = feat_df.var().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(16, 5))
        feat_var.head(30).plot(kind="bar", ax=ax,
                               color="steelblue", edgecolor="white")
        ax.set_title("Top 30 Features by Variance (pre-modelling)",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Variance"); ax.set_xlabel("Feature")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        self._save("plot07_feature_variance.png")

    def plot08_tsne(self, X_train_sc, y_train_orig):
        print("    Computing t-SNE (2D) — may take ~1 min …")
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train_sc), min(4000, len(X_train_sc)), replace=False)
        tsne   = TSNE(n_components=2, random_state=RANDOM_STATE,
                      perplexity=40, max_iter=1000)
        X_tsne = tsne.fit_transform(X_train_sc[idx])
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, name in enumerate(self.CN):
            mask = y_train_orig[idx] == i
            ax.scatter(X_tsne[mask,0], X_tsne[mask,1],
                       c=self.P[i%len(self.P)], label=name, alpha=0.6, s=15)
        ax.set_title("t-SNE Projection of Engineered Feature Space",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, markerscale=2)
        ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
        plt.tight_layout()
        self._save("plot08_tsne.png")

    def plot09_pca_variance(self, X_train_sc, feat_df):
        pca    = PCA(random_state=RANDOM_STATE).fit(X_train_sc)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95    = np.searchsorted(cumvar, 0.95) + 1
        n99    = np.searchsorted(cumvar, 0.99) + 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(range(1,31), pca.explained_variance_ratio_[:30],
                color="steelblue", edgecolor="white")
        ax1.set_title("Variance Explained by Each PC (Top 30)")
        ax1.set_xlabel("PC"); ax1.set_ylabel("Explained Variance Ratio")
        ax2.plot(range(1, len(cumvar)+1), cumvar, color="steelblue", lw=2)
        ax2.axhline(0.95, color="red",    ls="--", label=f"95% → {n95} PCs")
        ax2.axhline(0.99, color="orange", ls="--", label=f"99% → {n99} PCs")
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_xlabel("Number of PCs"); ax2.set_ylabel("Cumulative Variance")
        ax2.legend(); ax2.set_xlim(1, feat_df.shape[1])
        plt.tight_layout()
        self._save("plot09_pca_variance.png")

    # ── Supervised helpers ────────────────────────────────────────
    def _roc_grid(self, y_test_bin, y_prob, title, fname):
        nr, nc = self._grid()
        fig, axes = plt.subplots(nr, nc, figsize=(20, nr*5))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, name in enumerate(self.CN):
            fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_prob[:,i])
            ra = auc(fpr, tpr)
            axes[i].plot(fpr, tpr, color=self.P[i%len(self.P)],
                         lw=2, label=f"AUC={ra:.3f}")
            axes[i].plot([0,1],[0,1], "k--", lw=1)
            axes[i].fill_between(fpr, tpr, alpha=0.1,
                                 color=self.P[i%len(self.P)])
            axes[i].set_title(name, fontweight="bold")
            axes[i].set_xlabel("FPR"); axes[i].set_ylabel("TPR")
            axes[i].legend(fontsize=9)
        for j in range(self.NC, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save(fname)

    def _cm_fi_plot(self, y_test, y_pred, model, feat_df,
                    title, cmap, fi_color, fname):
        cm   = confusion_matrix(y_test, y_pred)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap=cmap, ax=ax1,
                    xticklabels=self.CN, yticklabels=self.CN,
                    linewidths=0.5, cbar_kws={"label":"Proportion"})
        ax1.set_title("Normalised Confusion Matrix")
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
        ax1.set_xticklabels(self.CN, rotation=30, ha="right", fontsize=8)
        fi = pd.Series(model.feature_importances_,
                       index=feat_df.columns).sort_values().tail(20)
        fi.plot(kind="barh", ax=ax2, color=fi_color, edgecolor="white")
        ax2.set_title("Top 20 Feature Importances")
        ax2.set_xlabel("Importance")
        plt.tight_layout()
        self._save(fname)
        return cm_n

    def plot10_rf_cm_fi(self, y_test, y_pred_rf, rf, feat_df):
        return self._cm_fi_plot(y_test, y_pred_rf, rf, feat_df,
            "Random Forest — Confusion Matrix + Feature Importance",
            "Blues", "steelblue", "plot10_rf_confusion_fi.png")

    def plot11_rf_roc(self, y_test_bin, y_prob_rf):
        self._roc_grid(y_test_bin, y_prob_rf,
            "Random Forest — ROC Curves (One-vs-Rest)", "plot11_rf_roc.png")

    def plot12_rf_learning_curve(self, rf, X_train_sc, y_train_orig, subjects_train):
        print("    Computing RF learning curve …")
        gkf = GroupKFold(n_splits=N_CV_FOLDS)
        splits = list(gkf.split(X_train_sc, y_train_orig, subjects_train))
        ts, tr, val = learning_curve(
            rf, X_train_sc, y_train_orig, cv=splits,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(ts, tr.mean(1)-tr.std(1), tr.mean(1)+tr.std(1),
                        alpha=0.15, color="blue")
        ax.fill_between(ts, val.mean(1)-val.std(1), val.mean(1)+val.std(1),
                        alpha=0.15, color="green")
        ax.plot(ts, tr.mean(1),  "o-", color="blue",  label="Train F1")
        ax.plot(ts, val.mean(1), "o-", color="green", label="Val F1 (GroupKFold)")
        ax.set_title("Random Forest — Learning Curve (GroupKFold)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Set Size"); ax.set_ylabel("Weighted F1")
        ax.legend(); ax.set_ylim(0, 1.05)
        plt.tight_layout()
        self._save("plot12_rf_learning_curve.png")

    def plot13_xgb_cm_fi(self, y_test, y_pred_xgb, xgb_model, feat_df):
        return self._cm_fi_plot(y_test, y_pred_xgb, xgb_model, feat_df,
            "XGBoost — Confusion Matrix + Feature Importance",
            "Greens", "seagreen", "plot13_xgb_confusion_fi.png")

    def plot14_xgb_roc(self, y_test_bin, y_prob_xgb):
        self._roc_grid(y_test_bin, y_prob_xgb,
            "XGBoost — ROC Curves (One-vs-Rest)", "plot14_xgb_roc.png")

    def plot15_svm_eval(self, y_test, y_pred_svm, X_train_sc, y_train_orig):
        cm   = confusion_matrix(y_test, y_pred_svm)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("SVM (RBF) — Confusion Matrix + 2D Decision Boundary",
                     fontsize=14, fontweight="bold")
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Purples", ax=ax1,
                    xticklabels=self.CN, yticklabels=self.CN,
                    linewidths=0.5, cbar_kws={"label":"Proportion"})
        ax1.set_title("Normalised Confusion Matrix")
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
        ax1.set_xticklabels(self.CN, rotation=30, ha="right", fontsize=8)
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        Xtr2 = pca2.fit_transform(X_train_sc)
        svm2 = SVC(kernel="rbf", C=10, gamma="scale",
                   class_weight="balanced", random_state=RANDOM_STATE)
        svm2.fit(Xtr2, y_train_orig)
        x1mn, x1mx = Xtr2[:,0].min()-1, Xtr2[:,0].max()+1
        x2mn, x2mx = Xtr2[:,1].min()-1, Xtr2[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x1mn,x1mx,300),
                              np.linspace(x2mn,x2mx,300))
        Z = svm2.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.18,
                     cmap=plt.cm.get_cmap("Set1", self.NC))
        for i, name in enumerate(self.CN):
            m = y_train_orig==i
            ax2.scatter(Xtr2[m,0], Xtr2[m,1], c=self.P[i%len(self.P)],
                        label=name, alpha=0.35, s=8)
        ax2.set_title("SVM Decision Regions (2-D PCA approx.)")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        ax2.legend(fontsize=7, markerscale=2)
        plt.tight_layout()
        self._save("plot15_svm_evaluation.png")
        return cm_n

    def plot16_svm_roc(self, y_test_bin, y_prob_svm):
        self._roc_grid(y_test_bin, y_prob_svm,
            "SVM (RBF) — ROC Curves (One-vs-Rest)", "plot16_svm_roc.png")

    # ── Unsupervised ──────────────────────────────────────────────
    def plot17_kmeans_elbow(self, inertias, silhouettes, K_range):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("KMeans — Choosing Optimal K", fontsize=14, fontweight="bold")
        ax1.plot(K_range, inertias, "o-", color="steelblue", lw=2)
        ax1.axvline(7, color="red", ls="--", label="K=7 (true #classes)")
        ax1.set_title("Elbow Curve"); ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")
        ax1.legend()
        ax2.plot(K_range, silhouettes, "o-", color="seagreen", lw=2)
        ax2.axvline(7, color="red", ls="--", label="K=7")
        ax2.set_title("Silhouette Score"); ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette")
        ax2.legend()
        plt.tight_layout()
        self._save("plot17_kmeans_elbow.png")

    def plot18_kmeans_clusters(self, X_vis, km_labels, centers_vis, y_train_orig):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("KMeans (K=7) — Cluster vs True Activity Labels",
                     fontsize=14, fontweight="bold")
        for i in range(7):
            m = km_labels==i
            ax1.scatter(X_vis[m,0], X_vis[m,1], c=self.P[i%len(self.P)],
                        label=f"Cluster {i+1}", alpha=0.4, s=10)
        ax1.scatter(centers_vis[:,0], centers_vis[:,1], c="black",
                    marker="X", s=200, zorder=5, label="Centroids")
        ax1.set_title("KMeans Clusters (2-D PCA)"); ax1.legend(fontsize=8)
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
        for i, name in enumerate(self.CN):
            m = y_train_orig==i
            ax2.scatter(X_vis[m,0], X_vis[m,1], c=self.P[i%len(self.P)],
                        label=name, alpha=0.4, s=10)
        ax2.set_title("True Activity Labels (2-D PCA)"); ax2.legend(fontsize=8)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        plt.tight_layout()
        self._save("plot18_kmeans_clusters.png")

    def plot19_cluster_activity_map(self, km_labels, y_train_orig):
        ca   = pd.crosstab(km_labels, y_train_orig)
        ca.columns = self.CN
        ca.index   = [f"C{i+1}" for i in range(7)]
        ca_n = ca.div(ca.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(ca_n, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label":"Proportion of Cluster"})
        ax.set_title("KMeans Cluster-to-Activity Mapping (row-normalised)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("True Activity"); ax.set_ylabel("Cluster")
        plt.tight_layout()
        self._save("plot19_cluster_activity_map.png")

    def plot20_dbscan(self, k_dists, X_db2, db_labels, n_clust_db, eps_val):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("DBSCAN — k-Distance & Cluster Visualisation",
                     fontsize=14, fontweight="bold")
        ax1.plot(range(len(k_dists)), k_dists, color="steelblue", lw=1.5)
        ax1.axhline(eps_val, color="red", ls="--", label=f"eps={eps_val}")
        ax1.set_title("k-Distance Graph (k=5) — eps Selection")
        ax1.set_xlabel("Points (sorted)"); ax1.set_ylabel("5th-NN Distance")
        ax1.legend()
        uniq = sorted(set(db_labels))
        tab  = plt.cm.tab10(np.linspace(0,1,max(len(uniq),2)))
        for idx2, lbl in enumerate(uniq):
            m2  = db_labels==lbl
            col = "grey" if lbl==-1 else tab[idx2%len(tab)]
            ax2.scatter(X_db2[m2,0], X_db2[m2,1], c=[col],
                        label="Noise" if lbl==-1 else f"Cluster {lbl}",
                        alpha=0.5 if lbl!=-1 else 0.2, s=10)
        ax2.set_title(f"DBSCAN Clusters (n={n_clust_db})  [2-D PCA]")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        ax2.legend(fontsize=7, markerscale=2)
        plt.tight_layout()
        self._save("plot20_dbscan.png")

    # ── Comparative ───────────────────────────────────────────────
    def plot21_dashboard(self, results, per_class_f1,
                         cv_rf, cv_xgb,
                         sil_km, ari_km, nmi_km, sil_db, ari_db):
        model_names = list(results.keys())
        fig = plt.figure(figsize=(22, 14))
        gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)
        fig.suptitle("HAR70+ — Complete Model Comparison Dashboard",
                     fontsize=16, fontweight="bold", y=1.01)
        ax1 = fig.add_subplot(gs[0,0])
        keys = ["Accuracy","F1_wtd","AUC"]
        lbls = ["Accuracy","Weighted F1","AUC (OvR)"]
        x = np.arange(len(keys)); w = 0.25
        for i,(mod,col) in enumerate(zip(model_names,
                                         ["#2196F3","#4CAF50","#FF5722"])):
            vals = [results[mod][k] for k in keys]
            bars = ax1.bar(x+i*w, vals, w, label=mod, color=col,
                           alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.004,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(x+w); ax1.set_xticklabels(lbls, fontsize=9)
        ax1.set_ylim(0,1.12); ax1.set_title("Overall Performance Metrics")
        ax1.set_ylabel("Score"); ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(gs[0,1:])
        x2  = np.arange(len(self.CN))
        for i,(mod,col) in enumerate(zip(model_names,
                                         ["#2196F3","#4CAF50","#FF5722"])):
            ax2.bar(x2+i*0.25, per_class_f1[mod], 0.25,
                    label=mod, color=col, alpha=0.85, edgecolor="white")
        ax2.set_xticks(x2+0.25)
        ax2.set_xticklabels(self.CN, rotation=30, ha="right", fontsize=9)
        ax2.set_ylim(0,1.12); ax2.set_title("Per-Class F1-Score Comparison")
        ax2.set_ylabel("F1-Score"); ax2.legend(fontsize=8)
        ax3 = fig.add_subplot(gs[1,0], polar=True)
        N_c    = len(self.CN)
        angles = [n/N_c*2*np.pi for n in range(N_c)] + [0]
        for mod, col in zip(model_names, ["#2196F3","#4CAF50","#FF5722"]):
            vals = list(per_class_f1[mod]) + [per_class_f1[mod].iloc[0]]
            ax3.plot(angles, vals, "o-", color=col, lw=2, label=mod)
            ax3.fill(angles, vals, alpha=0.08, color=col)
        ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(self.CN, size=8)
        ax3.set_ylim(0,1); ax3.set_title("Radar — Per-Class F1", pad=20)
        ax3.legend(loc="upper right", bbox_to_anchor=(1.4,1.15), fontsize=8)
        ax4 = fig.add_subplot(gs[1,1])
        bp   = ax4.boxplot([cv_rf, cv_xgb], patch_artist=True, notch=True,
                           medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp["boxes"], ["#2196F3","#4CAF50"]):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax4.set_xticklabels(["Random Forest","XGBoost"], fontsize=10)
        ax4.set_title("5-Fold GroupKFold CV Weighted F1")
        ax4.set_ylabel("Weighted F1")
        for i, cv in enumerate([cv_rf, cv_xgb], 1):
            ax4.text(i, cv.mean()+0.002, f"μ={cv.mean():.4f}",
                     ha="center", fontsize=9, fontweight="bold")
        ax5 = fig.add_subplot(gs[1,2])
        ud   = pd.DataFrame({
            "Silhouette": [sil_km, sil_db if not np.isnan(sil_db) else 0],
            "ARI":        [ari_km, ari_db if not np.isnan(ari_db) else 0],
            "NMI":        [nmi_km, 0],
        }, index=["KMeans (K=7)","DBSCAN"])
        ud.plot(kind="bar", ax=ax5, color=["#9C27B0","#FF9800","#00BCD4"],
                edgecolor="white", alpha=0.85)
        ax5.set_title("Unsupervised Clustering Metrics")
        ax5.set_ylabel("Score"); ax5.set_ylim(0,1.1)
        ax5.set_xticklabels(ud.index, rotation=0, fontsize=9)
        ax5.legend(fontsize=8)
        plt.savefig(f"{self.out}/plots/plot21_comparison_dashboard.png",
                    bbox_inches="tight", dpi=150)
        plt.close()
        print(f"    → {self.out}/plots/plot21_comparison_dashboard.png")

    def plot22_all_cm(self, cm_rf_n, cm_xgb_n, cm_svm_n):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle("Confusion Matrices — All 3 Models (Normalised)",
                     fontsize=14, fontweight="bold")
        for ax, cm, title, cmap in zip(
                axes,
                [cm_rf_n, cm_xgb_n, cm_svm_n],
                ["Random Forest","XGBoost","SVM (RBF)"],
                ["Blues","Greens","Purples"]):
            sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                        xticklabels=self.CN, yticklabels=self.CN,
                        linewidths=0.5, cbar_kws={"label":"Proportion"})
            ax.set_title(title, fontweight="bold", fontsize=12)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticklabels(self.CN, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(self.CN, fontsize=8)
        plt.tight_layout()
        self._save("plot22_all_confusion_matrices.png")

    def plot23_error_rates(self, y_test, preds_dict):
        fig, ax = plt.subplots(figsize=(13, 6))
        x_e = np.arange(len(self.CN))
        for i,(mod,col,pred) in enumerate(zip(
                preds_dict.keys(),
                ["#2196F3","#4CAF50","#FF5722"],
                preds_dict.values())):
            cr   = classification_report(y_test, pred,
                       target_names=self.CN, output_dict=True)
            errs = [1-cr[c]["recall"] for c in self.CN]
            ax.bar(x_e+i*0.25, errs, 0.25, label=mod,
                   color=col, alpha=0.85, edgecolor="white")
        ax.set_xticks(x_e+0.25)
        ax.set_xticklabels(self.CN, rotation=30, ha="right")
        ax.set_title("Per-Class Error Rate (1 − Recall) by Model",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Error Rate"); ax.legend(); ax.set_ylim(0,1)
        plt.tight_layout()
        self._save("plot23_error_rates.png")

    def plot24_fi_compare(self, rf, xgb_model, feat_df):
        top_rf  = pd.Series(rf.feature_importances_,
                            index=feat_df.columns).nlargest(20)
        top_xgb = pd.Series(xgb_model.feature_importances_,
                             index=feat_df.columns).nlargest(20)
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Feature Importance: RF vs XGBoost (Top 20)",
                     fontsize=14, fontweight="bold")
        top_rf.sort_values().plot(kind="barh", ax=ax1,
                                  color="#2196F3", edgecolor="white", alpha=0.85)
        ax1.set_title("Random Forest"); ax1.set_xlabel("Importance")
        top_xgb.sort_values().plot(kind="barh", ax=ax2,
                                   color="#4CAF50", edgecolor="white", alpha=0.85)
        ax2.set_title("XGBoost"); ax2.set_xlabel("Importance")
        plt.tight_layout()
        self._save("plot24_feature_importance_compare.png")

    def plot25_pr_curves(self, y_test_bin, y_prob_rf):
        nr, nc = self._grid()
        fig, axes = plt.subplots(nr, nc, figsize=(20, nr*5))
        fig.suptitle("Random Forest — Precision-Recall Curves (OvR)",
                     fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, name in enumerate(self.CN):
            prec, rec, _ = precision_recall_curve(y_test_bin[:,i], y_prob_rf[:,i])
            ap = average_precision_score(y_test_bin[:,i], y_prob_rf[:,i])
            axes[i].plot(rec, prec, color=self.P[i%len(self.P)],
                         lw=2, label=f"AP={ap:.3f}")
            axes[i].fill_between(rec, prec, alpha=0.1,
                                 color=self.P[i%len(self.P)])
            axes[i].set_title(name, fontweight="bold")
            axes[i].set_xlabel("Recall"); axes[i].set_ylabel("Precision")
            axes[i].set_ylim(0,1.05); axes[i].legend(fontsize=9)
        for j in range(self.NC, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save("plot25_rf_precision_recall.png")

    # ── SHAP ─────────────────────────────────────────────────────
    def plot26_shap_summary(self, shap_values, X_sample, feat_df):
        print("    Generating SHAP beeswarm summary …")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feat_df.columns.tolist(),
                          max_display=20, show=False)
        plt.title("SHAP Summary — XGBoost (Top 20 Features)",
                  fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        self._save("plot26_shap_summary.png")

    def plot27_shap_bar(self, shap_values, feat_df):
        print("    Generating SHAP per-class bar …")
        nr, nc = self._grid()
        fig, axes = plt.subplots(nr, nc, figsize=(22, nr*5))
        fig.suptitle("SHAP Mean |Value| per Class — XGBoost",
                     fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, name in enumerate(self.CN):
            sv_class = shap_values[:,:,i]
            mean_abs = np.abs(sv_class).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[-10:]
            axes[i].barh(
                [feat_df.columns[j] for j in top_idx],
                mean_abs[top_idx],
                color=self.P[i%len(self.P)], edgecolor="white", alpha=0.85)
            axes[i].set_title(name, fontweight="bold")
            axes[i].set_xlabel("Mean |SHAP|")
        for j in range(self.NC, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save("plot27_shap_per_class_bar.png")

    def plot28_shap_waterfall(self, explainer, shap_values,
                               X_sample, feat_df, y_test, y_pred_xgb):
        print("    Generating SHAP waterfall plots …")
        nr, nc = self._grid()
        fig, axes = plt.subplots(nr, nc, figsize=(22, nr*6))
        fig.suptitle("SHAP Waterfall — One Correct Prediction per Class",
                     fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, name in enumerate(self.CN):
            correct = np.where((y_test==i) & (y_pred_xgb==i))[0]
            if len(correct)==0 or correct[0]>=len(X_sample):
                axes[i].axis("off"); continue
            idx    = correct[0]
            sv     = shap_values[idx,:,i]
            base   = (explainer.expected_value[i]
                      if hasattr(explainer.expected_value,"__len__")
                      else explainer.expected_value)
            top_k  = 10
            top_idx = np.argsort(np.abs(sv))[-top_k:][::-1]
            colors  = ["#FF5722" if v>0 else "#2196F3" for v in sv[top_idx]]
            axes[i].barh(
                [feat_df.columns[j] for j in top_idx],
                sv[top_idx], color=colors, edgecolor="white", alpha=0.9)
            axes[i].axvline(0, color="black", lw=0.8)
            axes[i].set_title(f"{name}  (base={base:.3f})",
                              fontweight="bold", fontsize=9)
            axes[i].set_xlabel("SHAP value")
        for j in range(self.NC, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        self._save("plot28_shap_waterfall_per_class.png")


# ════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════
def extract_features(win):
    """134 statistical + frequency-domain features per 2-s window."""
    feats = {}
    for col in SENSOR_COLS:
        sig = win[col].values
        feats[f"{col}_mean"]         = np.mean(sig)
        feats[f"{col}_std"]          = np.std(sig)
        feats[f"{col}_min"]          = np.min(sig)
        feats[f"{col}_max"]          = np.max(sig)
        feats[f"{col}_range"]        = np.ptp(sig)
        feats[f"{col}_median"]       = np.median(sig)
        feats[f"{col}_q25"]          = np.percentile(sig, 25)
        feats[f"{col}_q75"]          = np.percentile(sig, 75)
        feats[f"{col}_iqr"]          = np.percentile(sig,75)-np.percentile(sig,25)
        feats[f"{col}_skew"]         = skew(sig)
        feats[f"{col}_kurt"]         = kurtosis(sig)
        feats[f"{col}_energy"]       = np.sum(sig**2)/len(sig)
        feats[f"{col}_rms"]          = np.sqrt(np.mean(sig**2))
        feats[f"{col}_mad"]          = np.mean(np.abs(sig-np.mean(sig)))
        feats[f"{col}_zcr"]          = ((sig[:-1]*sig[1:])<0).sum()
        feats[f"{col}_var"]          = np.var(sig)
        freqs, psd = welch(sig, fs=50, nperseg=min(50,len(sig)))
        feats[f"{col}_psd_mean"]     = np.mean(psd)
        feats[f"{col}_psd_max"]      = np.max(psd)
        feats[f"{col}_dom_freq"]     = freqs[np.argmax(psd)]
        psd_norm = psd/(psd.sum()+1e-10)
        feats[f"{col}_spec_entropy"] = -np.sum(psd_norm*np.log(psd_norm+1e-10))
    bm  = np.sqrt(win["back_x"]**2 +win["back_y"]**2 +win["back_z"]**2)
    tm  = np.sqrt(win["thigh_x"]**2+win["thigh_y"]**2+win["thigh_z"]**2)
    feats["back_mag_mean"]  = np.mean(bm)
    feats["thigh_mag_mean"] = np.mean(tm)
    feats["back_mag_std"]   = np.std(bm)
    feats["thigh_mag_std"]  = np.std(tm)
    feats["back_mag_max"]   = np.max(bm)
    feats["thigh_mag_max"]  = np.max(tm)
    feats["mag_diff_mean"]  = np.mean(np.abs(bm-tm))
    feats["mag_corr"]       = np.corrcoef(bm,tm)[0,1]
    feats["back_xy_corr"]   = np.corrcoef(win["back_x"], win["back_y"])[0,1]
    feats["back_xz_corr"]   = np.corrcoef(win["back_x"], win["back_z"])[0,1]
    feats["back_yz_corr"]   = np.corrcoef(win["back_y"], win["back_z"])[0,1]
    feats["thigh_xy_corr"]  = np.corrcoef(win["thigh_x"],win["thigh_y"])[0,1]
    feats["thigh_xz_corr"]  = np.corrcoef(win["thigh_x"],win["thigh_z"])[0,1]
    feats["thigh_yz_corr"]  = np.corrcoef(win["thigh_y"],win["thigh_z"])[0,1]
    return feats


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # ── STEP 1: LOAD ─────────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 1 — LOADING DATA")
    print("="*62)
    csv_files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in '{DATA_DIR}'.\n"
            "Set DATA_DIR at top of script to your har70plus/ folder.")
    dfs = []
    for f in csv_files:
        sid = int(os.path.splitext(os.path.basename(f))[0])
        df  = pd.read_csv(f); df.columns = df.columns.str.strip()
        df["subject"] = sid; dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    missing_c = [c for c in SENSOR_COLS+["label"] if c not in raw.columns]
    if missing_c: raise ValueError(f"Missing columns: {missing_c}")
    raw["label_name"] = raw["label"].map(ACTIVITY_NAMES)
    LABELS      = sorted(raw["label"].unique())
    CLASS_NAMES = [ACTIVITY_NAMES[l] for l in LABELS]
    print(f"  Subjects  : {raw['subject'].nunique()}")
    print(f"  Total rows: {len(raw):,}  |  Missing: {raw[SENSOR_COLS+['label']].isnull().sum().sum()}")
    print(f"  Activities: {[ACTIVITY_NAMES[l] for l in LABELS]}")
    raw.dropna(subset=SENSOR_COLS+["label"], inplace=True)
    raw.reset_index(drop=True, inplace=True)

    # ── STEP 2: EDA ───────────────────────────────────────────────
    print("\nSTEP 2 — EDA & VISUALISATIONS")
    vm_eda = VisualizationManager(OUTPUT_DIR, PALETTE, ACTIVITY_NAMES, CLASS_NAMES)
    vm_eda.plot01_class_distribution(raw, LABELS)
    vm_eda.plot02_per_subject(raw)
    vm_eda.plot03_raw_signals(raw, LABELS)
    vm_eda.plot04_boxplots(raw, LABELS)
    vm_eda.plot05_correlation_heatmaps(raw, LABELS)
    vm_eda.plot06_stats_heatmap(raw, LABELS)

    # ── STEP 3: FEATURE ENGINEERING ──────────────────────────────
    print("\n"+"="*62)
    print("STEP 3 — FEATURE ENGINEERING")
    print(f"  Window={WINDOW_SIZE} samples (2 s @ 50 Hz) | Step={STEP_SIZE} (50% overlap)")
    print("="*62)
    raw_sorted = raw.sort_values(["subject","timestamp"]).reset_index(drop=True)
    rows, y_all, subjects_all = [], [], []
    n = len(raw_sorted)
    print(f"  Processing {n:,} rows …")
    for start in range(0, n-WINDOW_SIZE, STEP_SIZE):
        win = raw_sorted.iloc[start:start+WINDOW_SIZE]
        if win["subject"].nunique() > 1: continue
        rows.append(extract_features(win))
        y_all.append(int(win["label"].mode()[0]))
        subjects_all.append(int(win["subject"].iloc[0]))
    feat_df      = pd.DataFrame(rows)
    y_all        = np.array(y_all)
    subjects_all = np.array(subjects_all)
    print(f"  Windows: {len(feat_df):,}  |  Features: {feat_df.shape[1]}")
    print(f"  NaN count: {feat_df.isnull().sum().sum()} → filled with 0")
    feat_df.fillna(0, inplace=True)
    print("  Class distribution:")
    for lbl, cnt in zip(*np.unique(y_all, return_counts=True)):
        print(f"    {ACTIVITY_NAMES[lbl]:<14}: {cnt:,}")
    vm_eda.plot07_feature_variance(feat_df)

    # ── STEP 4: PREPROCESSING ─────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 4 — PREPROCESSING [GroupShuffleSplit + StandardScaler + SMOTE]")
    print("="*62)
    le              = LabelEncoder()
    y_enc           = le.fit_transform(y_all)
    CLASS_NAMES_ENC = [ACTIVITY_NAMES[l] for l in le.classes_]
    N_CLASSES       = len(CLASS_NAMES_ENC)
    vm = VisualizationManager(OUTPUT_DIR, PALETTE, ACTIVITY_NAMES, CLASS_NAMES_ENC)
    X  = feat_df.values

    # Subject-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=subjects_all))
    X_train, X_test   = X[train_idx], X[test_idx]
    y_train_orig, y_test = y_enc[train_idx], y_enc[test_idx]
    subjects_train    = subjects_all[train_idx]
    print(f"  Train subjects: {np.unique(subjects_train).tolist()}")
    print(f"  Test  subjects: {np.unique(subjects_all[test_idx]).tolist()}")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Visualisations on pre-SMOTE data
    vm.plot08_tsne(X_train_sc, y_train_orig)
    vm.plot09_pca_variance(X_train_sc, feat_df)

    # GroupKFold splits (on pre-SMOTE for honest CV)
    gkf        = GroupKFold(n_splits=N_CV_FOLDS)
    gkf_splits = list(gkf.split(X_train_sc, y_train_orig, subjects_train))

    # SMOTE on training set only
    if USE_SMOTE:
        print("  Applying SMOTE on training set …")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train_orig)
        print(f"  Before: {X_train_sc.shape} → After: {X_train_bal.shape}")
    else:
        X_train_bal, y_train_bal = X_train_sc, y_train_orig

    print(f"  Test set (untouched): {X_test_sc.shape}")
    y_test_bin = label_binarize(y_test, classes=range(N_CLASSES))

    # ── STEP 5: RANDOM FOREST ─────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 5 — SUPERVISED MODEL 1: Random Forest")
    print("="*62)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE)
    print("  Running GroupKFold CV (pre-SMOTE, honest evaluation) …")
    cv_rf = cross_val_score(rf, X_train_sc, y_train_orig,
                             cv=gkf_splits, scoring="f1_weighted", n_jobs=-1)
    print("  Fitting final model on SMOTE-balanced data …")
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test_sc)
    y_prob_rf = rf.predict_proba(X_test_sc)
    acc_rf  = accuracy_score(y_test, y_pred_rf)
    f1_rf   = f1_score(y_test, y_pred_rf, average="weighted")
    auc_rf  = roc_auc_score(y_test, y_prob_rf, multi_class="ovr", average="weighted")
    print(f"  Accuracy={acc_rf:.4f}  F1={f1_rf:.4f}  AUC={auc_rf:.4f}")
    print(f"  GroupKFold CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES_ENC))
    cm_rf_n = vm.plot10_rf_cm_fi(y_test, y_pred_rf, rf, feat_df)
    vm.plot11_rf_roc(y_test_bin, y_prob_rf)
    vm.plot12_rf_learning_curve(rf, X_train_sc, y_train_orig, subjects_train)

    # ── STEP 6: XGBOOST ───────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 6 — SUPERVISED MODEL 2: XGBoost")
    print("="*62)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss",
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
    print("  Running GroupKFold CV (pre-SMOTE) …")
    cv_xgb = cross_val_score(xgb_model, X_train_sc, y_train_orig,
                              cv=gkf_splits, scoring="f1_weighted", n_jobs=-1)
    print("  Fitting final model on SMOTE-balanced data …")
    xgb_model.fit(X_train_bal, y_train_bal,
                  eval_set=[(X_test_sc, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test_sc)
    y_prob_xgb = xgb_model.predict_proba(X_test_sc)
    acc_xgb  = accuracy_score(y_test, y_pred_xgb)
    f1_xgb   = f1_score(y_test, y_pred_xgb, average="weighted")
    auc_xgb  = roc_auc_score(y_test, y_prob_xgb, multi_class="ovr", average="weighted")
    print(f"  Accuracy={acc_xgb:.4f}  F1={f1_xgb:.4f}  AUC={auc_xgb:.4f}")
    print(f"  GroupKFold CV: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=CLASS_NAMES_ENC))
    cm_xgb_n = vm.plot13_xgb_cm_fi(y_test, y_pred_xgb, xgb_model, feat_df)
    vm.plot14_xgb_roc(y_test_bin, y_prob_xgb)

    # Top-10 XGBoost features for unsupervised
    fi_xgb    = pd.Series(xgb_model.feature_importances_, index=feat_df.columns)
    top10_idx = [feat_df.columns.get_loc(f) for f in fi_xgb.nlargest(10).index]
    X_train_fs = X_train_sc[:, top10_idx]
    print(f"\n  Top-10 features (for clustering): {fi_xgb.nlargest(10).index.tolist()}")

    # ── STEP 7: SVM ───────────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 7 — SUPERVISED MODEL 3: SVM (RBF Kernel + PCA 50D)")
    print("="*62)
    pca_svm      = PCA(n_components=50, random_state=RANDOM_STATE)
    X_train_pca  = pca_svm.fit_transform(X_train_sc)
    X_test_pca   = pca_svm.transform(X_test_sc)
    X_train_pca_bal = pca_svm.transform(X_train_bal)
    print(f"  PCA variance retained: {pca_svm.explained_variance_ratio_.sum():.3f}")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight="balanced", random_state=RANDOM_STATE)
    print("  Fitting SVM on SMOTE-balanced PCA data …")
    svm.fit(X_train_pca_bal, y_train_bal)
    y_pred_svm = svm.predict(X_test_pca)
    y_prob_svm = svm.predict_proba(X_test_pca)
    acc_svm  = accuracy_score(y_test, y_pred_svm)
    f1_svm   = f1_score(y_test, y_pred_svm, average="weighted")
    auc_svm  = roc_auc_score(y_test, y_prob_svm, multi_class="ovr", average="weighted")
    print(f"  Accuracy={acc_svm:.4f}  F1={f1_svm:.4f}  AUC={auc_svm:.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES_ENC))
    cm_svm_n = vm.plot15_svm_eval(y_test, y_pred_svm, X_train_sc, y_train_orig)
    vm.plot16_svm_roc(y_test_bin, y_prob_svm)

    # ── STEP 8: KMEANS ────────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 8 — UNSUPERVISED: KMeans (top-10 features)")
    print("="*62)
    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    X_vis   = pca_vis.fit_transform(X_train_fs)
    inertias, silhouettes = [], []
    K_range = range(2, 12)
    print("  K | Inertia    | Silhouette")
    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        km.fit(X_train_fs)
        inertias.append(km.inertia_)
        s = silhouette_score(X_train_fs, km.labels_,
                              sample_size=min(3000,len(X_train_fs)),
                              random_state=RANDOM_STATE)
        silhouettes.append(s)
        print(f"  {k}  | {km.inertia_:10.0f} | {s:.4f}")
    vm.plot17_kmeans_elbow(inertias, silhouettes, K_range)
    km_best  = KMeans(n_clusters=7, init="k-means++", n_init=20,
                      random_state=RANDOM_STATE)
    km_best.fit(X_train_fs)
    km_labels    = km_best.labels_
    centers_vis  = pca_vis.transform(km_best.cluster_centers_)
    sil_km = silhouette_score(X_train_fs, km_labels,
                               sample_size=min(3000,len(X_train_fs)),
                               random_state=RANDOM_STATE)
    ari_km = adjusted_rand_score(y_train_orig, km_labels)
    nmi_km = normalized_mutual_info_score(y_train_orig, km_labels)
    print(f"\n  KMeans K=7 → Silhouette={sil_km:.4f} | ARI={ari_km:.4f} | NMI={nmi_km:.4f}")
    vm.plot18_kmeans_clusters(X_vis, km_labels, centers_vis, y_train_orig)
    vm.plot19_cluster_activity_map(km_labels, y_train_orig)

    # ── STEP 9: DBSCAN ────────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 9 — UNSUPERVISED: DBSCAN (top-10 features)")
    print("="*62)
    eps_val = 1.2
    n_db    = min(6000, len(X_train_fs))
    idx_db  = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train_fs), n_db, replace=False)
    X_db    = X_train_fs[idx_db]
    y_db    = y_train_orig[idx_db]
    nn = NearestNeighbors(n_neighbors=5).fit(X_db)
    dists, _ = nn.kneighbors(X_db)
    k_dists  = np.sort(dists[:,-1])
    db        = DBSCAN(eps=eps_val, min_samples=10, n_jobs=-1)
    db_labels = db.fit_predict(X_db)
    n_clust_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise_db = (db_labels==-1).sum()
    pca2_db = PCA(n_components=2, random_state=RANDOM_STATE)
    X_db2   = pca2_db.fit_transform(X_db)
    if n_clust_db > 1:
        valid  = db_labels!=-1
        sil_db = silhouette_score(X_db[valid], db_labels[valid],
                                   sample_size=min(2000,valid.sum()),
                                   random_state=RANDOM_STATE)
        ari_db = adjusted_rand_score(y_db[valid], db_labels[valid])
    else:
        sil_db, ari_db = float("nan"), float("nan")
    print(f"  Clusters: {n_clust_db} | Noise: {n_noise_db} ({n_noise_db/n_db*100:.1f}%)")
    print(f"  Silhouette={sil_db:.4f} | ARI={ari_db:.4f}")
    vm.plot20_dbscan(k_dists, X_db2, db_labels, n_clust_db, eps_val)

    # ── STEP 10: COMPARISON ───────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 10 — COMPARATIVE ANALYSIS")
    print("="*62)
    results = {
        "Random Forest": dict(Accuracy=acc_rf, F1_wtd=f1_rf,  AUC=auc_rf,
                              CV_mean=cv_rf.mean(),  CV_std=cv_rf.std()),
        "XGBoost":       dict(Accuracy=acc_xgb,F1_wtd=f1_xgb, AUC=auc_xgb,
                              CV_mean=cv_xgb.mean(), CV_std=cv_xgb.std()),
        "SVM (RBF)":     dict(Accuracy=acc_svm,F1_wtd=f1_svm, AUC=auc_svm,
                              CV_mean=None,          CV_std=None),
    }
    print(pd.DataFrame(results).T.to_string())
    cr_rf  = classification_report(y_test, y_pred_rf,
                target_names=CLASS_NAMES_ENC, output_dict=True)
    cr_xgb = classification_report(y_test, y_pred_xgb,
                target_names=CLASS_NAMES_ENC, output_dict=True)
    cr_svm = classification_report(y_test, y_pred_svm,
                target_names=CLASS_NAMES_ENC, output_dict=True)
    per_class_f1 = pd.DataFrame({
        "Random Forest": [cr_rf[c]["f1-score"]  for c in CLASS_NAMES_ENC],
        "XGBoost":       [cr_xgb[c]["f1-score"] for c in CLASS_NAMES_ENC],
        "SVM (RBF)":     [cr_svm[c]["f1-score"] for c in CLASS_NAMES_ENC],
    }, index=CLASS_NAMES_ENC)
    vm.plot21_dashboard(results, per_class_f1,
                        cv_rf, cv_xgb,
                        sil_km, ari_km, nmi_km, sil_db, ari_db)
    vm.plot22_all_cm(cm_rf_n, cm_xgb_n, cm_svm_n)
    vm.plot23_error_rates(y_test, {
        "Random Forest": y_pred_rf,
        "XGBoost":       y_pred_xgb,
        "SVM (RBF)":     y_pred_svm,
    })
    vm.plot24_fi_compare(rf, xgb_model, feat_df)
    vm.plot25_pr_curves(y_test_bin, y_prob_rf)

    # ── STEP 11: SHAP ─────────────────────────────────────────────
    print("\n"+"="*62)
    print("STEP 11 — SHAP EXPLAINABILITY (XGBoost)")
    print("="*62)
    n_shap  = min(500, len(X_test_sc))
    X_shap  = X_test_sc[:n_shap]
    y_shap  = y_test[:n_shap]
    yp_shap = y_pred_xgb[:n_shap]
    print(f"  Computing SHAP on {n_shap} test samples …")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)
    vm.plot26_shap_summary(shap_values, X_shap, feat_df)
    vm.plot27_shap_bar(shap_values, feat_df)
    vm.plot28_shap_waterfall(explainer, shap_values,
                              X_shap, feat_df, y_shap, yp_shap)

    # ── SAVE ARTIFACTS ────────────────────────────────────────────
    best = max(results, key=lambda m: results[m]["F1_wtd"])
    best_obj = {"Random Forest": rf, "XGBoost": xgb_model, "SVM (RBF)": svm}[best]
    joblib.dump(best_obj,          f"{OUTPUT_DIR}/best_model.pkl")
    joblib.dump(scaler,            f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(le,                f"{OUTPUT_DIR}/label_encoder.pkl")
    joblib.dump(feat_df.columns.tolist(), f"{OUTPUT_DIR}/feature_names.pkl")
    print(f"\n  Artifacts saved to {OUTPUT_DIR}/")
    print(f"    best_model.pkl  ({best})")
    print(f"    scaler.pkl")
    print(f"    label_encoder.pkl")
    print(f"    feature_names.pkl")

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n"+"="*62)
    print("FINAL RESULTS SUMMARY")
    print("="*62)
    print(f"\n  ✓ Best Model : {best}")
    for k, v in results[best].items():
        if v is not None: print(f"      {k:<12}: {v:.4f}")
    print(f"\n  KMeans K=7 → Silhouette={sil_km:.4f} | ARI={ari_km:.4f} | NMI={nmi_km:.4f}")
    print(f"  DBSCAN     → Silhouette={sil_db:.4f}  | ARI={ari_db:.4f}")
    print(f"\n  Total runtime: {elapsed/60:.1f} min")
    print(f"  28 plots saved to: {OUTPUT_DIR}/plots/")
    print("\n  PIPELINE COMPLETE ✓")


if __name__ == "__main__":
    main()
