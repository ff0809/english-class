import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import string

warnings.filterwarnings('ignore')

# ========== PROGRESS TRACKER ==========
class ProgressTracker:
    def __init__(self):
        self.start_time = None
        self.step_start_time = None
        self.step_name = ""

    def start_step(self, step_name):
        self.step_name = step_name
        self.step_start_time = time.time()
        if self.start_time is None:
            self.start_time = self.step_start_time
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"{'='*60}")
        return self

    def end_step(self):
        elapsed = time.time() - self.step_start_time
        total_elapsed = time.time() - self.start_time
        print(f"Complete in {elapsed:.2f} seconds")
        print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        return self

    def progress_update(self, current, total, message=""):
        percent = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        if message:
            print(f"\r[{bar}] {percent:.1f}% ({current}/{total}) - {message}", end='', flush=True)
        else:
            print(f"\r[{bar}] {percent:.1f}% ({current}/{total})", end='', flush=True)

        if current == total:
            print()

# ========== TEXT PROCESSING FUNCTIONS ==========
def simple_tokenize(text):
    """Simple tokenization function"""
    return [word for word in text.split() if word]

def remove_punctuation(text):
    """Remove punctuation"""
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

def simple_stemming(word):
    """Simple stemming"""
    suffixes = ['ing', 'ed', 's', 'es', 'ly', 'ment', 'ness', 'able', 'ible']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[:-len(suffix)]
            break
    return word

def preprocess_text(text):
    """Text preprocessing without NLTK"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_punctuation(text)
    tokens = simple_tokenize(text)
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [simple_stemming(token) for token in tokens]

    return ' '.join(tokens)

# ========== MAIN PROGRAM ==========
print("=" * 60)
print("MOVIE REVIEW SENTIMENT ANALYSIS PROJECT")
print("=" * 60)

progress = ProgressTracker()

# ========== 2. LOAD DATASET ==========
progress.start_step("Loading and exploring dataset")

try:
    print("Loading dataset...")
    chunksize = 10000
    chunks = []
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv("D:\english class\IMDB Dataset.csv",
                                           encoding='utf-8',
                                           chunksize=chunksize)):
        chunks.append(chunk)
        total_rows += len(chunk)
        progress.progress_update(i * chunksize + len(chunk), 50000,
                                f"Loaded {total_rows:,} rows...")

    df = pd.concat(chunks, ignore_index=True)

    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

progress.end_step()

print(f"\nChecking data quality...")
print(f"Missing values:\n{df.isnull().sum()}")
df = df.dropna()
print(f"Rows after cleaning: {len(df):,}")

# ========== 3. TEXT PREPROCESSING ==========
progress.start_step("Text preprocessing")

print("Starting text preprocessing...")
df['cleaned_review'] = ""
total_rows = len(df)

for i, idx in enumerate(df.index):
    df.at[idx, 'cleaned_review'] = preprocess_text(df.at[idx, 'review'])

    if (i + 1) % 1000 == 0 or (i + 1) == total_rows:
        progress.progress_update(i + 1, total_rows,
                                f"Processed {i + 1:,} of {total_rows:,} reviews")

print(f"\nSample original review: {df['review'].iloc[0][:50]}...")
print(f"Sample cleaned review: {df['cleaned_review'].iloc[0][:50]}...")

progress.end_step()

# ========== 4. EXPLORATORY DATA ANALYSIS ==========
progress.start_step("Exploratory data analysis")

sentiment_counts = df['sentiment'].value_counts()
print(f"Sentiment distribution:")
print(f"Positive: {sentiment_counts.get('positive', 0):,} reviews")
print(f"Negative: {sentiment_counts.get('negative', 0):,} reviews")
print(f"Ratio: {sentiment_counts.get('positive', 0)/len(df)*100:.1f}% positive")

df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
print(f"\nReview length statistics:")
print(f"Average: {df['review_length'].mean():.2f} words")
print(f"Minimum: {df['review_length'].min()} words")
print(f"Maximum: {df['review_length'].max()} words")
print(f"Median: {df['review_length'].median():.2f} words")

progress.end_step()

# ========== 5. FEATURE EXTRACTION ==========
progress.start_step("Feature extraction using TF-IDF")

print("Initializing TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.7,
    stop_words='english'
)

print("Transforming text to TF-IDF features...")
X = tfidf.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

print(f"Feature extraction completed!")
print(f"TF-IDF matrix shape: {X.shape}")
print(f"Number of features: {len(tfidf.get_feature_names_out())}")

progress.end_step()

# ========== 6. DATA SPLITTING ==========
progress.start_step("Splitting data into train and test sets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data split completed:")
print(f"Training set size: {X_train.shape[0]:,} samples")
print(f"Test set size: {X_test.shape[0]:,} samples")
print(f"Positive samples in training: {sum(y_train == 1):,}")
print(f"Negative samples in training: {sum(y_train == 0):,}")

progress.end_step()

# ========== 7. MODEL TRAINING ==========
progress.start_step("Training machine learning models")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', random_state=42, verbose=False)
}

results = {}

for i, (name, model) in enumerate(models.items(), 1):
    print(f"\n[{i}/{len(models)}] Training {name}...")
    model_start = time.time()

    print(f"Fitting model...")
    model.fit(X_train, y_train)
    training_time = time.time() - model_start
    print(f"Model trained in {training_time:.2f} seconds")

    print(f"Making predictions on test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': model,
        'predictions': y_pred,
        'training_time': training_time
    }

    print(f"{name} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

progress.end_step()

# ========== 8. VISUALIZATION ==========
progress.start_step("Creating visualizations")

print("Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sentiment Analysis Results', fontsize=16)

axes[0, 0].bar(['Positive', 'Negative'],
               [sentiment_counts.get('positive', 0), sentiment_counts.get('negative', 0)],
               color=['green', 'red'])
axes[0, 0].set_title('Sentiment Distribution')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')

model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

bars = axes[0, 1].bar(model_names, accuracies, color=['blue', 'orange', 'green'])
axes[0, 1].set_title('Model Accuracy Comparison')
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_ylim([0, 1.05])

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')

positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_review'])
wordcloud_pos = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=100).generate(positive_text)

axes[1, 0].imshow(wordcloud_pos, interpolation='bilinear')
axes[1, 0].set_title('Positive Reviews - Word Cloud')
axes[1, 0].axis('off')

negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_review'])
wordcloud_neg = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=100).generate(negative_text)

axes[1, 1].imshow(wordcloud_neg, interpolation='bilinear')
axes[1, 1].set_title('Negative Reviews - Word Cloud')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'sentiment_analysis_results.png'")

progress.end_step()

# ========== 9. CONFUSION MATRIX ==========
progress.start_step("Detailed analysis of best model")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
y_pred = results[best_model_name]['predictions']

print(f"Best model: {best_model_name}")
print(f"Best accuracy: {results[best_model_name]['accuracy']:.4f}")

print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")

print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, y_pred,
                            target_names=['Negative', 'Positive']))

progress.end_step()

# ========== 10. FEATURE IMPORTANCE ==========
if best_model_name == 'Random Forest':
    progress.start_step("Feature importance analysis")

    feature_names = tfidf.get_feature_names_out()
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Top 15 Important Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance.png'")

    progress.end_step()

# ========== 11. PREDICTION EXAMPLES ==========
progress.start_step("Example predictions on new reviews")

new_reviews = [
    "This movie was absolutely fantastic and thrilling!",
    "I was very disappointed with the poor acting.",
    "The cinematography was beautiful but story was weak.",
    "One of the best films of the year, highly recommended!"
]

print("Testing model on sample reviews:")
for i, review in enumerate(new_reviews, 1):
    cleaned = preprocess_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    print(f"Review {i}: '{review[:40]}...' -> Predicted: {sentiment.upper()}")

progress.end_step()

# ========== 12. FINAL SUMMARY ==========
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)

total_time = time.time() - progress.start_time
print(f"\nPROJECT SUMMARY")
print(f"{'-' * 40}")
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Total reviews analyzed: {len(df):,}")
print(f"Positive reviews: {sentiment_counts.get('positive', 0):,}")
print(f"Negative reviews: {sentiment_counts.get('negative', 0):,}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Testing samples: {X_test.shape[0]:,}")

print(f"\nMODEL PERFORMANCE")
print(f"{'-' * 40}")
for name, result in results.items():
    print(f"{name}:")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1-Score:  {result['f1']:.4f}")
    print(f"Training time: {result['training_time']:.2f}s")

print(f"\nBEST MODEL: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

print(f"\nOUTPUT FILES GENERATED:")
print(f"1. sentiment_analysis_results.png - Main visualizations")
print(f"2. confusion_matrix.png - Confusion matrix")
if best_model_name == 'Random Forest':
    print(f"3. feature_importance.png - Feature importance plot")

print(f"\nAll tasks completed successfully!")