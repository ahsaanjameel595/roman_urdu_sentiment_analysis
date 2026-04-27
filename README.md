# 🇵🇰 Roman Urdu Sentiment Analysis

A Machine Learning project that classifies **Roman Urdu** social media text as **Positive** or **Negative** using NLP techniques and three ML classifiers — deployed as an interactive web app with Streamlit.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 80.19% | 82.01% | 79.05% | 80.50% |
| **Logistic Regression** ✅ | **80.87%** | **83.27%** | **78.87%** | **81.01%** |
| SVM (LinearSVC) | 79.28% | 80.87% | 78.52% | 79.68% |

> ✅ **Best Model: Logistic Regression** with **80.87% accuracy**

---

## 📁 Project Structure

```
urdu-sentiment-analysis/
│
├── data/
│   └── RomanUrduSentiment.csv     # Kaggle dataset (11,000 samples)
│
├── models/
│   ├── best_model.pkl             # Best model (auto-selected)
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── tfidf.pkl                  # TF-IDF vectorizer
│
├── plots/
│   ├── accuracy_comparison.png
│   ├── metrics_comparison.png
│   └── confusion_matrices.png
│
├── preprocess.py                  # Text cleaning pipeline
├── train.py                       # Model training + evaluation
├── app.py                         # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Run

### 1. Clone / Download the project

```bash
git clone https://github.com/yourusername/urdu-sentiment-analysis.git
cd urdu-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `RomanUrduSentiment.csv` inside the `data/` folder.

> Dataset source: [Kaggle – Roman Urdu Sentiment Analysis](https://www.kaggle.com/datasets/owaisraza009/roman-urdu-dataset)

### 4. Train the models

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train Naive Bayes, Logistic Regression, and SVM
- Print accuracy, precision, recall, F1 for each model
- Save all models + TF-IDF vectorizer to `models/`
- Save evaluation plots to `plots/`

### 5. Launch the web app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔍 How It Works

### Text Preprocessing Pipeline (`preprocess.py`)

1. **Lowercase** — converts all text to lowercase
2. **Remove URLs, mentions, hashtags** — strips noise common in social media
3. **Remove numbers & special characters** — keeps only Roman alphabet letters
4. **Stopword removal** — filters out 60+ common Roman Urdu stopwords (ka, ki, hai, aur...)
5. **Short token removal** — drops tokens with fewer than 2 characters

### Feature Extraction

- **TF-IDF Vectorizer** with:
  - `max_features = 10,000`
  - `ngram_range = (1, 2)` — unigrams + bigrams
  - `sublinear_tf = True` — log-scale term frequency

### Models

| Model | Why used |
|---|---|
| Naive Bayes | Fast baseline, works well with sparse TF-IDF features |
| Logistic Regression | Strong linear classifier, best performer here |
| SVM (LinearSVC) | Effective in high-dimensional text spaces |

---

## 🖥️ Web App Features

- **Paste any Roman Urdu text** and get instant Positive / Negative prediction
- **Choose model** — switch between Naive Bayes, Logistic Regression, SVM, or Best
- **Example buttons** — try pre-loaded examples with one click
- **Cleaned text toggle** — see what the model actually processes
- **Training charts** — view confusion matrices and accuracy plots inside the app
- **Sidebar metrics** — all model scores at a glance

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

## 📚 Dataset

- **Source:** Kaggle — Roman Urdu Sentiment Dataset
- **Size:** ~11,000 samples
- **Labels:** Positive / Negative
- **Language:** Roman Urdu (Urdu written in Latin script)
- **Domain:** Social media comments, e-commerce reviews, public Facebook/Twitter posts

---

## 👤 Author

**Muhammad Ahsaan Jameel**
BS Computer Science — The Superior University, Lahore
[GitHub](https://github.com/ahsaanjameel595) | [LinkedIn](https://www.linkedin.com/in/ahsaan-jameel-6b8b48337)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
