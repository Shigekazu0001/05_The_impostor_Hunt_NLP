import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import re

def read_text_pair(folder, article_index):
    """
    指定されたフォルダからfile_1.txtとfile_2.txtを読み込む関数
    """
    folder_name = f"article_{int(article_index):04d}"
    base_path = os.path.join(folder, folder_name)
    with open(os.path.join(base_path, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(base_path, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()
    return file1, file2

def extract_features(text):
    """
    テキストから統計的特徴量を抽出する関数
    人間とAIの文章の違いを数値化して捉えるために使用
    """
    # テキストの前処理（空白の除去）
    text = text.strip()
    if not text:
        return [0] * 7  # 空のテキストの場合はゼロを返す
    
    # 基本統計を計算
    words = text.split()  # 単語に分割
    word_count = len(words)  # 単語数
    char_count = len(text)  # 文字数
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]  # 文に分割
    sentence_count = len(sentences)  # 文の数
    
    # 平均文字長を計算（NaN回避のため条件分岐）
    if words:
        avg_word_length = np.mean([len(word) for word in words])  # 平均単語長
    else:
        avg_word_length = 0
    
    # 平均文長を計算
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count  # 1文あたりの平均単語数
    else:
        avg_sentence_length = 0
    
    # 語彙の多様性を計算（同じ単語を繰り返さない度合い）
    if word_count > 0:
        unique_words = len(set(word.lower() for word in words))  # ユニークな単語数
        lexical_diversity = unique_words / word_count  # 語彙の多様性
    else:
        unique_words = 0
        lexical_diversity = 0
    
    # 句読点の密度を計算
    if char_count > 0:
        punctuation_density = len(re.findall(r'[.!?,:;]', text)) / char_count
    else:
        punctuation_density = 0
    
    return [
        word_count, char_count, sentence_count,
        avg_word_length, avg_sentence_length,
        lexical_diversity, punctuation_density
    ]

def create_enhanced_features(file1, file2):
    """
    2つのファイルから拡張された特徴量を作成する関数
    テキスト特徴量（TF-IDF用）と統計的特徴量を両方作成
    """
    # TF-IDF用のテキスト（2つのファイルを[SEP]で結合）
    combined_text = file1 + " [SEP] " + file2
    
    # 各ファイルの統計的特徴量を取得
    features1 = extract_features(file1)  # file1の統計
    features2 = extract_features(file2)  # file2の統計
    
    # 差分特徴量（file1 - file2）を計算
    # 例：file1の単語数 - file2の単語数
    diff_features = [f1 - f2 for f1, f2 in zip(features1, features2)]
    
    # 比率特徴量（file1 / file2）を計算（0除算回避）
    ratio_features = []
    for f1, f2 in zip(features1, features2):
        if f2 != 0:
            ratio_features.append(f1 / f2)  # 通常の比率
        else:
            ratio_features.append(0 if f1 == 0 else 1)  # 0除算の場合の処理
    
    # 全ての特徴量を結合し、NaN値を0に置換
    all_features = features1 + features2 + diff_features + ratio_features
    all_features = [0 if np.isnan(f) or np.isinf(f) else f for f in all_features]
    
    return combined_text, all_features

# === 学習データの準備 ===
print("=== 学習データの準備を開始 ===")

# 訓練データを読み込み
train_df = pd.read_csv("train.csv")
print(f"📊 訓練データの形状: {train_df.shape}")
print(f"📊 正解ラベルの分布:")
print(train_df['real_text_id'].value_counts())

# 学習用のデータとラベルを格納するリスト
X_text = []  # TF-IDF用のテキストデータ
X_features = []  # 統計的特徴量
y = []  # 正解ラベル

print("\n各訓練データから特徴量を抽出中...")
for idx, row in train_df.iterrows():
    # 対応するファイルペアを読み込み
    file1, file2 = read_text_pair("train", row["id"])
    
    # パターン1: file1 + [SEP] + file2 の順序
    # この場合、file1が本物かどうかを学習
    combined_text1, features1 = create_enhanced_features(file1, file2)
    X_text.append(combined_text1)
    X_features.append(features1)
    y.append(1 if row["real_text_id"] == 1 else 0)  # file1が本物なら1、そうでなければ0
    
    # パターン2: file2 + [SEP] + file1 の順序
    # この場合、file2が本物かどうかを学習
    combined_text2, features2 = create_enhanced_features(file2, file1)
    X_text.append(combined_text2)
    X_features.append(features2)
    y.append(1 if row["real_text_id"] == 2 else 0)  # file2が本物なら1、そうでなければ0

print(f"📊 学習サンプル数: {len(X_text)} (元データの2倍)")
print(f"📊 ラベル分布: {pd.Series(y).value_counts()}")

# === TF-IDF特徴量の作成 ===
print("\n=== TF-IDF特徴量の作成 ===")

# TF-IDFベクトライザーを設定（特徴量数を削減）
vectorizer = TfidfVectorizer(
    max_features=2000,      # 特徴量数を5000から2000に削減
    min_df=2,               # 最低2回以上出現する単語のみ使用
    max_df=0.8,             # 80%以上の文書に出現する単語は除外
    ngram_range=(1, 2),     # 1-gram（単語）と2-gram（単語ペア）を使用
    stop_words='english'    # 英語のストップワード（the, a, anなど）を除去
)

# テキストをTF-IDFベクトルに変換
X_tfidf = vectorizer.fit_transform(X_text)
print(f"📊 TF-IDF特徴量の形状: {X_tfidf.shape}")

# === 統計的特徴量の準備 ===
print("\n=== 統計的特徴量の準備 ===")

# 統計的特徴量を配列に変換
X_features_array = np.array(X_features)
print(f"📊 統計的特徴量の形状: {X_features_array.shape}")

# NaN値のチェックと修正
if np.any(np.isnan(X_features_array)):
    print("⚠️  NaN値を検出しました。0に置換します。")
    X_features_array = np.nan_to_num(X_features_array, nan=0.0)

# 統計的特徴量の正規化
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features_array)
print("✅ 統計的特徴量を正規化しました")

# === 特徴量の結合 ===
print("\n=== 特徴量の結合 ===")

# TF-IDF特徴量と正規化された統計的特徴量を結合
from scipy.sparse import hstack
X_combined = hstack([X_tfidf, X_features_scaled])
print(f"📊 結合後の特徴量の形状: {X_combined.shape}")

# スパース行列内のNaN値チェック
if hasattr(X_combined, 'data') and np.any(np.isnan(X_combined.data)):
    print("⚠️  スパース行列内のNaN値を修正中...")
    X_combined = X_combined.toarray()
    X_combined = np.nan_to_num(X_combined, nan=0.0)
    
print(f"📊 最終的な特徴量行列: {X_combined.shape}")

# === モデルの学習 ===
print("\n=== モデルの学習 ===")

# ロジスティック回帰モデルを作成（収束問題を解決）
model = LogisticRegression(
    random_state=42, 
    max_iter=5000,          # 反復回数を増加
    solver='liblinear',     # 小さなデータセットに適したソルバー
    C=1.0,                  # 正則化パラメータ
    penalty='l2'            # L2正則化
)
model.fit(X_combined, y)
print("✅ モデルの学習が完了しました")

# === 交差検証による性能評価 ===
print("\n=== 交差検証による性能評価 ===")

# 5分割交差検証でモデルの性能を評価
cv_scores = cross_val_score(model, X_combined, y, cv=5)
print(f"📊 交差検証スコア: {cv_scores}")
print(f"📊 平均スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# === テストデータでの予測 ===
print("\n=== テストデータでの予測 ===")

test_dir = "test"
test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
print(f"📊 テストフォルダ数: {len(test_folders)}")

results = []

print("各テストデータを予測中...")
for i, folder in enumerate(test_folders):
    # フォルダ名からIDを抽出
    article_id = folder.split("_")[1]
    
    # テストファイルを読み込み
    file1, file2 = read_text_pair(test_dir, article_id)
    
    # 両方のパターンで特徴量を作成
    combined_text1, features1 = create_enhanced_features(file1, file2)  # file1が最初
    combined_text2, features2 = create_enhanced_features(file2, file1)  # file2が最初
    
    # TF-IDF変換
    tfidf1 = vectorizer.transform([combined_text1])
    tfidf2 = vectorizer.transform([combined_text2])
    
    # 統計的特徴量の正規化
    features1_scaled = scaler.transform(np.array([features1]))
    features2_scaled = scaler.transform(np.array([features2]))
    
    # 特徴量を結合
    X_test1 = hstack([tfidf1, features1_scaled])
    X_test2 = hstack([tfidf2, features2_scaled])
    
    # 予測確率を計算
    prob1 = model.predict_proba(X_test1)[0][1]  # file1が本物である確率
    prob2 = model.predict_proba(X_test2)[0][1]  # file2が本物である確率
    
    # より高い確率を持つ方を選択
    real = 1 if prob1 > prob2 else 2
    # IDを数値に変換して保存
    results.append((int(article_id), real))
    
    # 最初の5個は詳細を表示
    if i < 5:
        print(f"テスト {i}: file1確率={prob1:.4f}, file2確率={prob2:.4f}, 予測={real}")

# === 結果の保存 ===
print("\n=== 結果の保存 ===")

submission_df = pd.DataFrame(results, columns=["id", "real_text_id"])
print(f"📊 予測結果の分布:")
print(submission_df["real_text_id"].value_counts())

submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csvを保存しました")