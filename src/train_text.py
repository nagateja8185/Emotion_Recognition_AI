import os
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# --- AUTO-DETECT PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "text", "train.txt")
VAL_FILE   = os.path.join(BASE_DIR, "data", "text", "val.txt")
TEST_FILE  = os.path.join(BASE_DIR, "data", "text", "test.txt")
OUT_DIR    = os.path.join(BASE_DIR, "models", "text_emotion")

print("📄 TRAIN FILE:", TRAIN_FILE)
print("📄 VAL FILE  :", VAL_FILE)
print("📄 TEST FILE :", TEST_FILE)

def clean_text(text):
    """Enhanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common filler words
    filler_words = {'really', 'very', 'just', 'so', 'much', 'many', 'things', 'stuff'}
    words = [w for w in text.split() if w not in filler_words]
    return ' '.join(words)

def load_text_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")

    # Read using semicolon delimiter
    df = pd.read_csv(path, sep=";", header=None, names=["text", "emotion"], engine="python")

    # Clean up
    df = df.dropna(subset=["text", "emotion"])
    df["text"] = df["text"].apply(clean_text)
    df["emotion"] = df["emotion"].astype(str).str.strip()
    df = df[df["text"] != ""]
    
    # Balance dataset by undersampling majority classes
    print("\n📊 Original class distribution:")
    print(df['emotion'].value_counts())
    
    # Optional: You can add class balancing here if needed
    
    return df

def main():
    train_df = load_text_file(TRAIN_FILE)
    val_df = load_text_file(VAL_FILE)
    test_df = load_text_file(TEST_FILE)

    X_train, y_train = train_df["text"], train_df["emotion"]
    X_val, y_val = val_df["text"], val_df["emotion"]
    X_test, y_test = test_df["text"], test_df["emotion"]

    print(f"✅ Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")
    print("Detected classes:", sorted(set(y_train)))

    # Enhanced pipeline with better feature extraction and ensemble voting
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english'
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='multinomial'
        ))
    ])

    print("\n🚀 Training enhanced text emotion model...")
    pipeline.fit(X_train, y_train)
    
    # Validation performance
    preds_val = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, preds_val)
    print(f"\n📊 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print("\n📋 Validation Classification Report:")
    print(classification_report(y_val, preds_val))
    
    # Test performance
    preds_test = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, preds_test)
    print(f"\n🧪 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\n📋 Test Classification Report:")
    print(classification_report(y_test, preds_test))

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Text emotion model saved to {model_path}")
    
    # Save class information
    import json
    class_info = {
        'classes': sorted(list(set(y_train))),
        'num_classes': len(set(y_train)),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc)
    }
    info_path = os.path.join(OUT_DIR, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"✅ Model info saved to {info_path}")

if __name__ == "__main__":
    main()
