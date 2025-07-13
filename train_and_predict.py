import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def read_text_pair(folder, article_index):
    folder_name = f"article_{int(article_index):04d}"  # ← ここで article_0000 形式に
    base_path = os.path.join(folder, folder_name)
    with open(os.path.join(base_path, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(base_path, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()
    return file1, file2

# === Train ===
train_df = pd.read_csv("train.csv")

X, y = [], []

for idx, row in train_df.iterrows():
    file1, file2 = read_text_pair("train", row["id"])
    # モデルに2パターン学習させる（file1がrealか、file2がrealか）
    X.append(file1 + "[SEP]" + file2)
    y.append(1 if row["real_text_id"] == 1 else 0)
    
    X.append(file2 + "[SEP]" + file1)
    y.append(1 if row["real_text_id"] == 2 else 0)

vectorizer = TfidfVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# === Predict ===
test_dir = "test"
test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
results = []

for folder in test_folders:
    file1, file2 = read_text_pair(test_dir, folder.split("_")[1])
    pair1 = file1 + "[SEP]" + file2
    pair2 = file2 + "[SEP]" + file1
    vec1 = vectorizer.transform([pair1])
    vec2 = vectorizer.transform([pair2])
    prob1 = model.predict_proba(vec1)[0][1]
    prob2 = model.predict_proba(vec2)[0][1]
    real = 1 if prob1 > prob2 else 2
    results.append((folder, real))

submission_df = pd.DataFrame(results, columns=["id", "real_text_id"])
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved.")
