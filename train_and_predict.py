import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

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

def extract_advanced_features(text):
    """テキストから拡張された統計的特徴量を抽出する関数"""
    text = text.strip()
    if not text:
        return [0.0] * 16  # 固定された特徴量数
    
    # 基本統計
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)
    
    # 基本特徴量
    avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
    
    # 語彙の多様性
    if word_count > 0:
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / word_count
    else:
        unique_words = 0
        lexical_diversity = 0.0
    
    # 句読点の特徴量
    punctuation_density = len(re.findall(r'[.!?,:;]', text)) / char_count if char_count > 0 else 0.0
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # 新規追加特徴量
    # 1. エラー検出特徴量
    spelling_errors = detect_spelling_errors(text)
    grammar_errors = detect_grammar_errors(text)
    
    # 2. 専門用語密度
    technical_density = calculate_technical_density(text)
    
    # 3. 感情分析特徴量
    sentiment_score, subjectivity_score = analyze_sentiment(text)
    
    # 4. 文構造の複雑性
    complexity_score = calculate_complexity(text)
    
    # 5. 大文字使用パターン
    uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0.0
    
    # 全ての特徴量を浮動小数点数に変換し、NaN/Infを0に置換
    features = [
        float(word_count), float(char_count), float(sentence_count),
        float(avg_word_length), float(avg_sentence_length),
        float(lexical_diversity), float(punctuation_density),
        float(exclamation_count), float(question_count),
        float(spelling_errors), float(grammar_errors),
        float(technical_density), float(sentiment_score), float(subjectivity_score),
        float(complexity_score), float(uppercase_ratio)
    ]
    
    # NaN/Infを0に置換
    features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
    
    return features

def detect_spelling_errors(text):
    """簡易的なスペルミス検出（英語の一般的な間違いパターン）"""
    # 一般的なスペルミスパターン
    error_patterns = [
        r'\b\w*off\s+information\b',  # "off information" instead of "of information"
        r'\btypescombinations\b',      # "typescombinations" (単語の結合ミス)
        r'\ballthe\b',                 # "allthe" instead of "all the"
        r'\bwhich\s+which\b',          # 重複
        r'\bthe\s+the\b',              # 重複
        r'\band\s+and\b',              # 重複
    ]
    
    error_count = 0
    for pattern in error_patterns:
        error_count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return error_count

def detect_grammar_errors(text):
    """簡易的な文法エラー検出"""
    # 文法エラーのパターン
    grammar_patterns = [
        r'\ba\s+[aeiou]',              # "a" before vowel sound
        r'\ban\s+[bcdfghjklmnpqrstvwxyz]',  # "an" before consonant sound
        r'\s{2,}',                     # 複数の連続スペース
        r'[a-z][A-Z]',                 # 大文字小文字の不適切な配置
        r'[.!?]\s*[a-z]',              # 文の始まりが小文字
    ]
    
    error_count = 0
    for pattern in grammar_patterns:
        error_count += len(re.findall(pattern, text))
    
    return error_count

def calculate_technical_density(text):
    """技術・専門用語の密度を計算"""
    # 天文学・科学技術用語
    technical_terms = [
        'telescope', 'astronomical', 'photometric', 'infrared', 'survey',
        'observatory', 'celestial', 'spacecraft', 'dataset', 'calibrated',
        'wavelength', 'spectrum', 'magnitude', 'luminosity', 'parallax',
        'redshift', 'nebula', 'galaxy', 'stellar', 'cosmic', 'exoplanet',
        'petabyte', 'terabyte', 'database', 'algorithm', 'processing'
    ]
    
    text_lower = text.lower()
    count = sum(1 for term in technical_terms if term in text_lower)
    word_count = len(text.split())
    
    return count / word_count if word_count > 0 else 0.0

def analyze_sentiment(text):
    """簡易的な感情分析（辞書ベース）"""
    # ポジティブ単語
    positive_words = ['excellent', 'great', 'amazing', 'wonderful', 'fantastic', 
                     'impressive', 'significant', 'valuable', 'important', 'useful']
    
    # ネガティブ単語
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 
                     'difficult', 'problem', 'error', 'failure', 'wrong']
    
    # 主観的単語
    subjective_words = ['believe', 'think', 'feel', 'hope', 'wish', 
                       'probably', 'maybe', 'perhaps', 'seems', 'appears']
    
    text_lower = text.lower()
    words = text_lower.split()
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    subj_count = sum(1 for word in words if word in subjective_words)
    
    sentiment_score = (pos_count - neg_count) / len(words) if words else 0.0
    subjectivity_score = subj_count / len(words) if words else 0.0
    
    return sentiment_score, subjectivity_score

def calculate_complexity(text):
    """文の複雑性を計算"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    
    # 接続詞の使用頻度
    connectives = ['however', 'therefore', 'moreover', 'furthermore', 
                  'additionally', 'consequently', 'nevertheless', 'although']
    
    complexity_score = 0.0
    for sentence in sentences:
        # 文の長さによる複雑性
        complexity_score += len(sentence.split()) / 10.0
        
        # 接続詞の使用
        for conn in connectives:
            if conn in sentence.lower():
                complexity_score += 1.0
    
    return complexity_score / len(sentences)

def create_enhanced_features(file1, file2):
    """2つのファイルから拡張された特徴量を作成する関数"""
    # TF-IDF用のテキスト
    combined_text = file1 + " [SEP] " + file2
    
    # 拡張された統計的特徴量
    features1 = extract_advanced_features(file1)
    features2 = extract_advanced_features(file2)
    
    # 特徴量の次元数を確認
    assert len(features1) == 16, f"features1の次元数が間違っています: {len(features1)}"
    assert len(features2) == 16, f"features2の次元数が間違っています: {len(features2)}"
    
    # 差分特徴量
    diff_features = [f1 - f2 for f1, f2 in zip(features1, features2)]
    
    # 比率特徴量
    ratio_features = []
    for f1, f2 in zip(features1, features2):
        if f2 != 0:
            ratio = f1 / f2
        else:
            ratio = 0.0 if f1 == 0 else 1.0
        ratio_features.append(ratio)
    
    # 追加の比較特徴量
    # 語彙の重複度
    words1 = set(file1.lower().split())
    words2 = set(file2.lower().split())
    vocabulary_overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0.0
    
    # 文長の分散
    sentences1 = [s.strip() for s in re.split(r'[.!?]+', file1) if s.strip()]
    sentences2 = [s.strip() for s in re.split(r'[.!?]+', file2) if s.strip()]
    
    var1 = np.var([len(s.split()) for s in sentences1]) if sentences1 else 0.0
    var2 = np.var([len(s.split()) for s in sentences2]) if sentences2 else 0.0
    sentence_length_var_diff = var1 - var2
    
    # 全特徴量を結合 (16 + 16 + 16 + 16 + 2 = 66次元)
    all_features = features1 + features2 + diff_features + ratio_features + [vocabulary_overlap, sentence_length_var_diff]
    
    # NaN/Infを0に置換
    all_features = [0.0 if np.isnan(f) or np.isinf(f) else float(f) for f in all_features]
    
    # 最終的な次元数を確認
    assert len(all_features) == 66, f"最終特徴量の次元数が間違っています: {len(all_features)}"
    
    return combined_text, all_features

# === 特徴量作成関連の関数 ===
def prepare_training_data(train_df):
    """訓練データから特徴量とラベルを作成する関数"""
    X_text = []
    X_features = []
    y = []
    
    print("\n各訓練データから拡張特徴量を抽出中...")
    for i, row in train_df.iterrows():
        try:
            file1, file2 = read_text_pair("train", row["id"])
            
            # パターン1: file1 + [SEP] + file2
            combined_text1, features1 = create_enhanced_features(file1, file2)
            X_text.append(combined_text1)
            X_features.append(features1)
            y.append(1 if row["real_text_id"] == 1 else 0)
            
            # パターン2: file2 + [SEP] + file1
            combined_text2, features2 = create_enhanced_features(file2, file1)
            X_text.append(combined_text2)
            X_features.append(features2)
            y.append(1 if row["real_text_id"] == 2 else 0)
            
        except Exception as e:
            print(f"エラー: ID {row['id']} の処理中にエラーが発生しました: {e}")
            continue
    
    print(f"📊 学習サンプル数: {len(X_text)} (元データの2倍)")
    print(f"📊 ラベル分布: {pd.Series(y).value_counts()}")
    
    # 特徴量の次元数を確認
    if X_features:
        print(f"📊 特徴量の次元数: {len(X_features[0])}")
        # 全ての特徴量が同じ次元数かチェック
        feature_lengths = [len(f) for f in X_features]
        if not all(length == feature_lengths[0] for length in feature_lengths):
            print("⚠️  特徴量の次元数が一致していません!")
            for i, length in enumerate(feature_lengths):
                if length != feature_lengths[0]:
                    print(f"   サンプル {i}: {length} 次元")
    
    return X_text, X_features, y

def create_optimized_tfidf_features(X_text):
    """最適化されたTF-IDF特徴量を作成する関数"""
    print("\n=== 最適化されたTF-IDF特徴量の作成 ===")
    
    # より広範囲のパラメータで最適化
    vectorizer = TfidfVectorizer(
        max_features=3000,        # 特徴量数を増加
        min_df=2,
        max_df=0.85,             # 閾値を微調整
        ngram_range=(1, 3),      # 3-gramまで拡張
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'\b\w+\b'  # より厳密なトークン化
    )
    
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"📊 TF-IDF特徴量の形状: {X_tfidf.shape}")
    
    return vectorizer, X_tfidf

def prepare_statistical_features(X_features):
    """統計的特徴量を準備する関数"""
    print("\n=== 拡張統計的特徴量の準備 ===")
    
    # リストをNumPy配列に変換前にデバッグ情報を表示
    print(f"📊 特徴量リストの長さ: {len(X_features)}")
    if X_features:
        print(f"📊 各特徴量の次元数: {[len(f) for f in X_features[:5]]}")  # 最初の5つを表示
    
    try:
        X_features_array = np.array(X_features, dtype=np.float64)
        print(f"📊 変換後の配列形状: {X_features_array.shape}")
    except ValueError as e:
        print(f"⚠️  配列変換でエラーが発生: {e}")
        # 各特徴量の長さをチェック
        feature_lengths = [len(f) for f in X_features]
        unique_lengths = set(feature_lengths)
        print(f"⚠️  特徴量の長さの種類: {unique_lengths}")
        
        # 最も多い長さに合わせて調整
        most_common_length = max(set(feature_lengths), key=feature_lengths.count)
        print(f"⚠️  最も多い長さ: {most_common_length}")
        
        # 長さを統一
        X_features_fixed = []
        for f in X_features:
            if len(f) == most_common_length:
                X_features_fixed.append(f)
            elif len(f) < most_common_length:
                # 不足分を0で埋める
                X_features_fixed.append(f + [0.0] * (most_common_length - len(f)))
            else:
                # 長すぎる場合は切り詰める
                X_features_fixed.append(f[:most_common_length])
        
        X_features_array = np.array(X_features_fixed, dtype=np.float64)
        print(f"📊 修正後の配列形状: {X_features_array.shape}")
    
    if np.any(np.isnan(X_features_array)):
        print("⚠️  NaN値を検出しました。0に置換します。")
        X_features_array = np.nan_to_num(X_features_array, nan=0.0)
    
    if np.any(np.isinf(X_features_array)):
        print("⚠️  Inf値を検出しました。0に置換します。")
        X_features_array = np.nan_to_num(X_features_array, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features_array)
    print(f"✅ 拡張統計的特徴量を正規化しました (次元数: {X_features_scaled.shape[1]})")
    
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

# === 改良されたモデル関連の関数 ===
def create_ensemble_model():
    """アンサンブルモデルを作成する関数"""
    print("\n=== アンサンブルモデルの作成 ===")
    
    # ベースモデル1: ロジスティック回帰
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=3000,
        solver='liblinear',
        C=0.5,                    # 正則化を強化
        penalty='l2'
    )
    
    # ベースモデル2: ランダムフォレスト
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # アンサンブルモデル
    ensemble_model = VotingClassifier([
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('xgb', xgb_model)
    ], voting='soft')
    
    return ensemble_model

def train_and_optimize_model(X_combined, y):
    """モデルを学習し最適化する関数"""
    print("\n=== モデルの学習と最適化 ===")
    
    # アンサンブルモデルの作成
    model = create_ensemble_model()
    
    # ハイパーパラメータの最適化（簡易版）
    param_grid = {
        'lr__C': [0.1, 0.5, 1.0],
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [15, 20, 25]
    }
    
    print("📊 グリッドサーチによる最適化を実行中...")
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3,  # 計算時間を考慮して3-fold
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_combined, y)
    best_model = grid_search.best_estimator_
    
    print(f"✅ 最適なパラメータ: {grid_search.best_params_}")
    print(f"✅ 最適化後のスコア: {grid_search.best_score_:.4f}")
    
    return best_model

def evaluate_model(model, X_combined, y):
    """モデルの性能を評価する関数"""
    print("\n=== 交差検証による性能評価 ===")
    cv_scores = cross_val_score(model, X_combined, y, cv=5, scoring='accuracy')
    print(f"📊 交差検証スコア: {cv_scores}")
    print(f"📊 平均スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def predict_test_data_enhanced(test_dir, vectorizer, scaler, model):
    """改良されたテストデータ予測関数"""
    print("\n=== テストデータでの予測（改良版） ===")
    test_folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])
    print(f"📊 テストフォルダ数: {len(test_folders)}")
    
    results = []
    prediction_confidence = []
    
    for i, folder in enumerate(test_folders):
        try:
            article_id = folder.split("_")[1]
            file1, file2 = read_text_pair(test_dir, article_id)
            
            # 複数回の予測で安定性を確保
            prob1_list = []
            prob2_list = []
            
            for _ in range(3):  # 3回の予測
                prob1 = predict_single_pair_enhanced(file1, file2, vectorizer, scaler, model)
                prob2 = predict_single_pair_enhanced(file2, file1, vectorizer, scaler, model)
                prob1_list.append(prob1)
                prob2_list.append(prob2)
            
            # 平均を取る
            prob1_avg = np.mean(prob1_list)
            prob2_avg = np.mean(prob2_list)
            
            # 予測結果と信頼度
            real = 1 if prob1_avg > prob2_avg else 2
            confidence = abs(prob1_avg - prob2_avg)
            
            results.append((int(article_id), real))
            prediction_confidence.append(confidence)
            
            if i < 5:
                print(f"テスト {i}: file1確率={prob1_avg:.4f}, file2確率={prob2_avg:.4f}, 予測={real}, 信頼度={confidence:.4f}")
        
        except Exception as e:
            print(f"⚠️  テスト {folder} でエラーが発生: {e}")
            # デフォルト予測を追加
            article_id = folder.split("_")[1]
            results.append((int(article_id), 1))
            prediction_confidence.append(0.5)
    
    # 信頼度の分析
    print(f"\n📊 予測信頼度統計:")
    print(f"   平均信頼度: {np.mean(prediction_confidence):.4f}")
    print(f"   信頼度中央値: {np.median(prediction_confidence):.4f}")
    print(f"   低信頼度予測数 (<0.1): {sum(1 for c in prediction_confidence if c < 0.1)}")
    
    return results

def predict_single_pair_enhanced(text1, text2, vectorizer, scaler, model):
    """単一ペアの予測確率を計算する改良版関数"""
    try:
        combined_text, features = create_enhanced_features(text1, text2)
        tfidf = vectorizer.transform([combined_text])
        features_scaled = scaler.transform(np.array([features]))
        X_test = hstack([tfidf, features_scaled])
        
        # アンサンブルモデルの予測確率
        proba = model.predict_proba(X_test)[0][1]
        return proba
    except Exception as e:
        print(f"⚠️  予測中にエラーが発生: {e}")
        return 0.5  # デフォルト確率

def save_results(results):
    """結果をCSVファイルに保存する関数"""
    print("\n=== 結果の保存 ===")
    submission_df = pd.DataFrame(results, columns=["id", "real_text_id"])
    print(f"📊 予測結果の分布:")
    print(submission_df["real_text_id"].value_counts())
    
    # 結果の保存
    submission_df.to_csv("submission_enhanced.csv", index=False)
    print("✅ submission_enhanced.csvを保存しました")

def main():
    """メインの実行関数"""
    print("=== 改良版テキスト分類システムの開始 ===")
    
    # データの準備
    train_df = pd.read_csv("train.csv")
    print(f"📊 訓練データの形状: {train_df.shape}")
    print(f"📊 正解ラベルの分布:")
    print(train_df['real_text_id'].value_counts())
    
    # 拡張特徴量の作成
    X_text, X_features, y = prepare_training_data(train_df)
    vectorizer, X_tfidf = create_optimized_tfidf_features(X_text)
    scaler, X_features_scaled = prepare_statistical_features(X_features)
    X_combined = combine_features(X_tfidf, X_features_scaled)
    
    # アンサンブルモデルの学習と最適化
    model = train_and_optimize_model(X_combined, y)
    cv_scores = evaluate_model(model, X_combined, y)
    
    # テストデータの予測と結果の保存
    results = predict_test_data_enhanced("test", vectorizer, scaler, model)
    save_results(results)
    
    print("\n🎉 改良版システムの実行が完了しました！")

if __name__ == "__main__":
    main()