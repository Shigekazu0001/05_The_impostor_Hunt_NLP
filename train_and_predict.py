import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# === データ読み込み関連の関数 ===
def read_text_pair(folder, article_index):
    """指定されたフォルダからfile_1.txtとfile_2.txtを読み込む関数"""
    folder_name = f"article_{int(article_index):04d}"
    base_path = os.path.join(folder, folder_name)
    with open(os.path.join(base_path, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(base_path, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()
    return file1, file2

def extract_features(text):
    """テキストから統計的特徴量を抽出する関数"""
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
    """2つのファイルから拡張された特徴量を作成する関数"""
    # TF-IDF用のテキスト（2つのファイルを[SEP]で結合）
    combined_text = file1 + " [SEP] " + file2

    # 各ファイルの統計的特徴量を取得
    features1 = extract_features(file1)  # file1の統計
    features2 = extract_features(file2)  # file2の統計

    # 差分特徴量（file1 - file2）を計算
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


# === 特徴量作成関連の関数 ===
def prepare_training_data(train_df):
    """訓練データから特徴量とラベルを作成する関数"""
    X_text = []      # TF-IDF用のテキストデータ
    X_features = []  # 統計的特徴量
    y = []           # 正解ラベル

    print("\n各訓練データから特徴量を抽出中...")
    for _, row in train_df.iterrows():
        # 対応するファイルペアを読み込み
        file1, file2 = read_text_pair("train", row["id"])

        # パターン1: file1 + [SEP] + file2 の順序
        combined_text1, features1 = create_enhanced_features(file1, file2)
        X_text.append(combined_text1)
        X_features.append(features1)
        y.append(1 if row["real_text_id"] == 1 else 0)

        # パターン2: file2 + [SEP] + file1 の順序
        combined_text2, features2 = create_enhanced_features(file2, file1)
        X_text.append(combined_text2)
        X_features.append(features2)
        y.append(1 if row["real_text_id"] == 2 else 0)

    print(f"📊 学習サンプル数: {len(X_text)} (元データの2倍)")
    print(f"📊 ラベル分布: {pd.Series(y).value_counts()}")
    
    return X_text, X_features, y

def create_tfidf_features(X_text):
    """TF-IDF特徴量を作成する関数"""
    print("\n=== TF-IDF特徴量の作成 ===")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"📊 TF-IDF特徴量の形状: {X_tfidf.shape}")
    return vectorizer, X_tfidf

def prepare_statistical_features(X_features):
    """統計的特徴量を準備する関数"""
    print("\n=== 統計的特徴量の準備 ===")
    X_features_array = np.array(X_features)
    
    if np.any(np.isnan(X_features_array)):
        print("⚠️  NaN値を検出しました。0に置換します。")
        X_features_array = np.nan_to_num(X_features_array, nan=0.0)
    
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features_array)
    print("✅ 統計的特徴量を正規化しました")
    return scaler, X_features_scaled

def combine_features(X_tfidf, X_features_scaled):
    """特徴量を結合する関数"""
    print("\n=== 特徴量の結合 ===")
    X_combined = hstack([X_tfidf, X_features_scaled])
    
    if hasattr(X_combined, 'data') and np.any(np.isnan(X_combined.data)):
        print("⚠️  スパース行列内のNaN値を修正中...")
        X_combined = X_combined.toarray()
        X_combined = np.nan_to_num(X_combined, nan=0.0)
    
    print(f"📊 最終的な特徴量行列: {X_combined.shape}")
    return X_combined

# === モデル関連の関数 ===
def train_model(X_combined, y):
    """モデルを学習する関数"""
    print("\n=== モデルの学習 ===")
    model = LogisticRegression(
        random_state=42,
        max_iter=5000,
        solver='liblinear',
        C=1.0,
        penalty='l2'
    )
    model.fit(X_combined, y)
    print("✅ モデルの学習が完了しました")
    return model

def evaluate_model(model, X_combined, y):
    """モデルの性能を評価する関数"""
    print("\n=== 交差検証による性能評価 ===")
    cv_scores = cross_val_score(model, X_combined, y, cv=5)
    print(f"📊 交差検証スコア: {cv_scores}")
    print(f"📊 平均スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return cv_scores

def predict_test_data(test_dir, vectorizer, scaler, model):
    """テストデータの予測を行う関数"""
    print("\n=== テストデータでの予測 ===")
    test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
    print(f"📊 テストフォルダ数: {len(test_folders)}")
    
    results = []
    for i, folder in enumerate(test_folders):
        article_id = folder.split("_")[1]
        file1, file2 = read_text_pair(test_dir, article_id)
        
        # 両方のパターンで特徴量を作成と予測
        prob1 = predict_single_pair(file1, file2, vectorizer, scaler, model)
        prob2 = predict_single_pair(file2, file1, vectorizer, scaler, model)
        
        # より高い確率を持つ方を選択
        real = 1 if prob1 > prob2 else 2
        results.append((int(article_id), real))
        
        if i < 5:
            print(f"テスト {i}: file1確率={prob1:.4f}, file2確率={prob2:.4f}, 予測={real}")
    
    return results

def predict_single_pair(text1, text2, vectorizer, scaler, model):
    """単一のテキストペアの予測確率を計算する関数"""
    combined_text, features = create_enhanced_features(text1, text2)
    tfidf = vectorizer.transform([combined_text])
    features_scaled = scaler.transform(np.array([features]))
    X_test = hstack([tfidf, features_scaled])
    return model.predict_proba(X_test)[0][1]

def save_results(results):
    """結果をCSVファイルに保存する関数"""
    print("\n=== 結果の保存 ===")
    submission_df = pd.DataFrame(results, columns=["id", "real_text_id"])
    print(f"📊 予測結果の分布:")
    print(submission_df["real_text_id"].value_counts())
    submission_df.to_csv("submission.csv", index=False)
    print("✅ submission.csvを保存しました")

def main():
    """メインの実行関数"""
    # データの準備
    print("=== 学習データの準備を開始 ===")
    train_df = pd.read_csv("train.csv")
    print(f"📊 訓練データの形状: {train_df.shape}")
    print(f"📊 正解ラベルの分布:")
    print(train_df['real_text_id'].value_counts())

    # 特徴量の作成
    X_text, X_features, y = prepare_training_data(train_df)
    vectorizer, X_tfidf = create_tfidf_features(X_text)
    scaler, X_features_scaled = prepare_statistical_features(X_features)
    X_combined = combine_features(X_tfidf, X_features_scaled)

    # モデルの学習と評価
    model = train_model(X_combined, y)
    cv_scores = evaluate_model(model, X_combined, y)

    # テストデータの予測と結果の保存
    results = predict_test_data("test", vectorizer, scaler, model)
    save_results(results)

if __name__ == "__main__":
    main()