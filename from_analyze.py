import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_insight_based_features(text):
    """データ分析結果に基づく最適化された特徴量 - 固定長25次元"""
    features = [0.0] * 25  # 固定長で初期化
    
    if not text.strip():
        return features
    
    # 基本統計
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences) if sentences else 1  # 0除算回避
    
    # 1. 長さ関連の特徴量（最も重要な差別化要因）
    features[0] = word_count
    features[1] = char_count
    features[2] = sentence_count
    
    # 2. 密度特徴量（AIは冗長な傾向）
    features[3] = word_count / sentence_count  # words_per_sentence
    features[4] = char_count / sentence_count  # chars_per_sentence
    
    # 3. 句読点の使用パターン（差が明確）
    periods = len(re.findall(r'\.', text))
    commas = len(re.findall(r',', text))
    parentheses = len(re.findall(r'[()]', text))
    quotes = len(re.findall(r'["\']', text))
    
    # 正規化された句読点密度
    features[5] = periods / char_count if char_count > 0 else 0
    features[6] = commas / char_count if char_count > 0 else 0
    features[7] = parentheses / char_count if char_count > 0 else 0
    features[8] = quotes / char_count if char_count > 0 else 0
    
    # 4. 文ごとの句読点使用率
    features[9] = periods / sentence_count
    features[10] = commas / sentence_count
    features[11] = parentheses / sentence_count
    features[12] = quotes / sentence_count
    
    # 5. 語彙の複雑さ（AIの特徴を捉える）
    alpha_words = [w for w in words if w.isalpha()]
    if alpha_words:
        features[13] = np.mean([len(w) for w in alpha_words])  # avg_word_length
        unique_words = len(set(w.lower() for w in alpha_words))
        features[14] = unique_words / len(alpha_words)  # lexical_diversity
    else:
        features[13] = 0
        features[14] = 0
    
    # 6. 文の長さの変動性（人間の方が不規則）
    if len(sentences) > 1:
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_length_std = np.std(sentence_lengths)
        sentence_length_mean = np.mean(sentence_lengths)
        features[15] = sentence_length_std
        features[16] = sentence_length_std / sentence_length_mean if sentence_length_mean > 0 else 0  # CV
    else:
        features[15] = 0
        features[16] = 0
    
    # 7. 特殊文字の使用
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:()\'"/-]', text))
    features[17] = special_chars
    
    # 8. 大文字使用パターン
    uppercase_words = len([w for w in words if w.isupper() and len(w) > 1])
    title_case_words = len([w for w in words if w.istitle()])
    features[18] = uppercase_words
    features[19] = title_case_words
    
    # 9. 数字の使用
    numbers = len(re.findall(r'\d+', text))
    features[20] = numbers
    
    # 10. その他の句読点
    semicolons = len(re.findall(r';', text))
    colons = len(re.findall(r':', text))
    exclamations = len(re.findall(r'!', text))
    questions = len(re.findall(r'\?', text))
    
    features[21] = semicolons
    features[22] = colons
    features[23] = exclamations
    features[24] = questions
    
    # NaN/Inf値の処理
    for i in range(len(features)):
        if np.isnan(features[i]) or np.isinf(features[i]):
            features[i] = 0.0
    
    return features

def create_comparative_features(file1, file2):
    """比較特徴量を重点的に作成"""
    # 基本特徴量（各25次元）
    features1 = extract_insight_based_features(file1)
    features2 = extract_insight_based_features(file2)
    
    # テキスト結合
    combined_text = file1 + " [SEP] " + file2
    
    # 重要な比較特徴量（10次元）
    comparative_features = [0.0] * 10
    
    # 1. 長さの比較（最重要）
    len1, len2 = len(file1.split()), len(file2.split())
    char1, char2 = len(file1), len(file2)
    sent1 = len([s for s in re.split(r'[.!?]+', file1) if s.strip()])
    sent2 = len([s for s in re.split(r'[.!?]+', file2) if s.strip()])
    
    comparative_features[0] = len1 / max(1, len2)  # word ratio
    comparative_features[1] = char1 / max(1, char2)  # char ratio
    comparative_features[2] = sent1 / max(1, sent2)  # sentence ratio
    
    # 2. 句読点使用の比較
    punct1 = len(re.findall(r'[.!?,:;()]', file1))
    punct2 = len(re.findall(r'[.!?,:;()]', file2))
    comparative_features[3] = punct1 / max(1, punct2)
    
    # 3. 語彙多様性の比較
    words1 = [w.lower() for w in file1.split() if w.isalpha()]
    words2 = [w.lower() for w in file2.split() if w.isalpha()]
    
    if words1 and words2:
        diversity1 = len(set(words1)) / len(words1)
        diversity2 = len(set(words2)) / len(words2)
        comparative_features[4] = diversity1 / max(0.001, diversity2)
    else:
        comparative_features[4] = 1.0
    
    # 4. 重要な差分特徴量（5次元）
    important_indices = [0, 1, 2, 3, 4]  # word_count, char_count, sentence_count, words_per_sentence, chars_per_sentence
    for i, idx in enumerate(important_indices):
        comparative_features[5 + i] = features1[idx] - features2[idx]
    
    # 差分特徴量（25次元）
    diff_features = [f1 - f2 for f1, f2 in zip(features1, features2)]
    
    # 比率特徴量（25次元）
    ratio_features = []
    for f1, f2 in zip(features1, features2):
        if f2 != 0:
            ratio = f1 / f2
        else:
            ratio = 1.0 if f1 == 0 else 2.0
        
        # 異常値クリッピング
        ratio = max(-10, min(10, ratio))
        ratio_features.append(ratio)
    
    # 全特徴量を結合 (25 + 25 + 10 + 25 + 25 = 110次元)
    all_features = features1 + features2 + comparative_features + diff_features + ratio_features
    
    # 最終的な異常値処理
    for i in range(len(all_features)):
        if np.isnan(all_features[i]) or np.isinf(all_features[i]):
            all_features[i] = 0.0
        else:
            all_features[i] = max(-100, min(100, all_features[i]))
    
    return combined_text, all_features

def read_text_pair(folder, article_index):
    """テキストペアを読み込み"""
    folder_name = f"article_{int(article_index):04d}"
    base_path = os.path.join(folder, folder_name)
    with open(os.path.join(base_path, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(base_path, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()
    return file1, file2

def prepare_insight_training_data(train_df):
    """分析結果に基づく訓練データ準備"""
    X_text = []
    X_features = []
    y = []

    print("分析結果に基づく特徴量を抽出中...")
    for idx, row in train_df.iterrows():
        try:
            file1, file2 = read_text_pair("train", row["id"])

            # パターン1: file1 vs file2
            combined_text1, features1 = create_comparative_features(file1, file2)
            X_text.append(combined_text1)
            X_features.append(features1)
            y.append(1 if row["real_text_id"] == 1 else 0)

            # パターン2: file2 vs file1
            combined_text2, features2 = create_comparative_features(file2, file1)
            X_text.append(combined_text2)
            X_features.append(features2)
            y.append(1 if row["real_text_id"] == 2 else 0)
            
        except Exception as e:
            print(f"エラー（ID: {row['id']}）: {e}")
            continue

    print(f"📊 特徴量次元数: {len(X_features[0]) if X_features else '未定義'}")
    print(f"📊 学習サンプル数: {len(X_text)}")
    
    # 特徴量の次元チェック
    if X_features:
        feature_lengths = [len(f) for f in X_features]
        print(f"📊 特徴量長の範囲: {min(feature_lengths)} - {max(feature_lengths)}")
        if min(feature_lengths) != max(feature_lengths):
            print("⚠️ 特徴量の次元が不一致です。修正中...")
            # 最短の長さに合わせる
            min_len = min(feature_lengths)
            X_features = [f[:min_len] for f in X_features]
            print(f"✅ 特徴量を{min_len}次元に統一しました")
    
    return X_text, X_features, y

def create_optimized_tfidf(X_text):
    """最適化されたTF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=2000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )
    
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"📊 TF-IDF特徴量: {X_tfidf.shape}")
    return vectorizer, X_tfidf

def train_insight_model(X_combined, y):
    """分析結果に基づくモデル訓練"""
    print("=== 最適化モデルの訓練 ===")
    
    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            C=0.5,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=5000
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_combined, y, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            print(f"{name}: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        except Exception as e:
            print(f"{name}でエラー: {e}")
            continue
    
    print(f"\n最適モデル: {best_name} (CV Score: {best_score:.4f})")
    
    # 最適モデルを訓練
    best_model.fit(X_combined, y)
    
    return best_model, best_name

def predict_insight_test(test_dir, vectorizer, scaler, model):
    """テスト予測"""
    print("=== テスト予測実行 ===")
    
    if not os.path.exists(test_dir):
        print(f"⚠️ テストディレクトリ '{test_dir}' が見つかりません")
        return []
    
    test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
    print(f"📊 テストフォルダ数: {len(test_folders)}")
    
    results = []
    confidence_scores = []
    
    for folder in test_folders:
        try:
            article_id = folder.split("_")[1]
            file1, file2 = read_text_pair(test_dir, article_id)
            
            # 両方向で予測
            prob1 = predict_insight_pair(file1, file2, vectorizer, scaler, model)
            prob2 = predict_insight_pair(file2, file1, vectorizer, scaler, model)
            
            # 信頼度スコア
            confidence1 = abs(prob1 - 0.5)
            confidence2 = abs(prob2 - 0.5)
            
            # より信頼度の高い予測を採用
            if confidence1 > confidence2:
                real = 1 if prob1 > 0.5 else 2
                confidence = confidence1
            else:
                real = 1 if prob1 > prob2 else 2
                confidence = max(confidence1, confidence2)
            
            results.append((int(article_id), real))
            confidence_scores.append(confidence)
            
        except Exception as e:
            print(f"予測エラー（{folder}）: {e}")
            # デフォルト予測
            article_id = folder.split("_")[1]
            results.append((int(article_id), 1))
            confidence_scores.append(0.0)
    
    # 信頼度統計
    if confidence_scores:
        print(f"予測信頼度 - 平均: {np.mean(confidence_scores):.3f}, 最小: {np.min(confidence_scores):.3f}, 最大: {np.max(confidence_scores):.3f}")
    
    return results

def predict_insight_pair(text1, text2, vectorizer, scaler, model):
    """単一ペア予測"""
    try:
        combined_text, features = create_comparative_features(text1, text2)
        
        # TF-IDF
        tfidf = vectorizer.transform([combined_text])
        
        # 統計特徴量
        features_scaled = scaler.transform(np.array([features]))
        
        # 結合
        X_test = hstack([tfidf, features_scaled])
        
        return model.predict_proba(X_test)[0][1]
    except Exception as e:
        print(f"予測エラー: {e}")
        return 0.5  # デフォルト値

def main():
    """メイン実行関数"""
    print("=== データ分析結果に基づく最適化分類器（修正版） ===\n")
    
    # データ準備
    train_df = pd.read_csv("train.csv")
    print(f"📊 訓練データ: {train_df.shape}")
    
    # 特徴量作成
    X_text, X_features, y = prepare_insight_training_data(train_df)
    
    if not X_features:
        print("❌ 特徴量の抽出に失敗しました")
        return
    
    # TF-IDF特徴量
    vectorizer, X_tfidf = create_optimized_tfidf(X_text)
    
    # 統計特徴量の正規化
    print("統計特徴量の正規化中...")
    scaler = StandardScaler()
    
    # numpy配列に変換（事前チェック）
    X_features_array = np.array(X_features, dtype=float)
    print(f"📊 統計特徴量の形状: {X_features_array.shape}")
    
    X_features_scaled = scaler.fit_transform(X_features_array)
    
    # 特徴量結合
    X_combined = hstack([X_tfidf, X_features_scaled])
    print(f"📊 最終特徴量行列: {X_combined.shape}")
    
    # モデル訓練
    model, model_name = train_insight_model(X_combined, y)
    
    # テスト予測
    results = predict_insight_test("test", vectorizer, scaler, model)
    
    if results:
        # 結果保存
        submission_df = pd.DataFrame(results, columns=["id", "real_text_id"])
        submission_df.to_csv("submission_insight_based_fixed.csv", index=False)
        
        print(f"\n=== 分析結果に基づく最適化完了 ===")
        print(f"使用モデル: {model_name}")
        print(f"予測分布: {submission_df['real_text_id'].value_counts().to_dict()}")
        print("✅ submission_insight_based_fixed.csv を保存しました")
    else:
        print("❌ 予測結果が得られませんでした")

if __name__ == "__main__":
    main()