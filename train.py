"""
train.py  –  Train and evaluate Naive Bayes, Logistic Regression, SVM
             on the Roman Urdu Sentiment dataset.
             Saves best model + vectorizer to models/
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from preprocess import preprocess_series

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/RomanUrduSentiment.csv"
MODEL_DIR   = "models"
PLOT_DIR    = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, header=None, encoding="utf-8", on_bad_lines="skip")
    df.columns = ["sentiment", "text"]
    df["sentiment"] = df["sentiment"].str.strip()
    df = df[df["sentiment"].isin(["pos", "neg"])]
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────
def prepare(df):
    print("Preprocessing text...")
    df["clean_text"] = preprocess_series(df["text"])
    df = df[df["clean_text"].str.strip() != ""]
    # Map labels to binary
    df["label"] = df["sentiment"].map({"pos": 1, "neg": 0})
    return df


# ── 3. Vectorise ──────────────────────────────────────────────────────────────
def vectorise(X_train, X_test):
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True
    )
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)
    return tfidf, X_tr, X_te


# ── 4. Train & evaluate all models ────────────────────────────────────────────
def train_all(X_tr, X_te, y_train, y_test):
    models = {
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "SVM":                 LinearSVC(C=1.0, max_iter=2000)
    }

    results = {}
    for name, clf in models.items():
        print(f"\n Training {name}...")
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)
        f1   = f1_score(y_test, preds)
        cm   = confusion_matrix(y_test, preds)

        results[name] = {
            "model":     clf,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "cm":        cm,
            "preds":     preds
        }

        print(f"   Accuracy : {acc*100:.2f}%")
        print(f"   Precision: {prec*100:.2f}%")
        print(f"   Recall   : {rec*100:.2f}%")
        print(f"   F1 Score : {f1*100:.2f}%")
        print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

    return results


# ── 5. Save plots ─────────────────────────────────────────────────────────────
def save_plots(results):
    names = list(results.keys())

    # -- Accuracy bar chart
    accs = [results[n]["accuracy"] * 100 for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/accuracy_comparison.png", dpi=150)
    plt.close()

    # -- Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, names):
        cm = results[name]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Neg", "Pos"],
                    yticklabels=["Neg", "Pos"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()

    # -- Metrics comparison
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name][m] * 100 for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Full Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/metrics_comparison.png", dpi=150)
    plt.close()

    print(f"\n Plots saved to '{PLOT_DIR}/'")


# ── 6. Save best model ────────────────────────────────────────────────────────
def save_best(results, tfidf):
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_clf  = results[best_name]["model"]
    print(f"\n Best model: {best_name}  ({results[best_name]['accuracy']*100:.2f}%)")

    joblib.dump(best_clf, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(tfidf,    f"{MODEL_DIR}/tfidf.pkl")

    # Also save all models individually
    name_map = {
        "Naive Bayes": "naive_bayes",
        "Logistic Regression": "logistic_regression",
        "SVM": "svm"
    }
    for name, clf_dict in results.items():
        joblib.dump(clf_dict["model"], f"{MODEL_DIR}/{name_map[name]}.pkl")

    print(f" Models saved to '{MODEL_DIR}/'")
    return best_name


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Roman Urdu Sentiment Analysis — Training")
    print("=" * 55)

    df = load_data()
    print(f"\nDataset: {len(df)} samples")
    print(df["sentiment"].value_counts().rename({"pos": "Positive", "neg": "Negative"}))

    df = prepare(df)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    tfidf, X_tr, X_te = vectorise(X_train, X_test)

    results = train_all(X_tr, X_te, y_train, y_test)

    save_plots(results)
    best = save_best(results, tfidf)

    print("\n" + "=" * 55)
    print(f"   Done!  Best model → {best}")
    print("   Run:  streamlit run app.py")
    print("=" * 55)
