import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

def read_text_pair(folder, article_index):
    """指定されたフォルダからfile_1.txtとfile_2.txtを読み込む関数"""
    folder_name = f"article_{int(article_index):04d}"
    base_path = os.path.join(folder, folder_name)
    with open(os.path.join(base_path, "file_1.txt"), encoding="utf-8") as f:
        file1 = f.read()
    with open(os.path.join(base_path, "file_2.txt"), encoding="utf-8") as f:
        file2 = f.read()
    return file1, file2

def analyze_text_characteristics(text):
    """テキストの詳細な特徴を分析"""
    if not text.strip():
        return {}
    
    # 基本統計
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    # 句読点の分析
    punctuation_patterns = {
        'periods': len(re.findall(r'\.', text)),
        'commas': len(re.findall(r',', text)),
        'exclamations': len(re.findall(r'!', text)),
        'questions': len(re.findall(r'\?', text)),
        'semicolons': len(re.findall(r';', text)),
        'colons': len(re.findall(r':', text)),
        'parentheses': len(re.findall(r'[()]', text)),
        'quotes': len(re.findall(r'["\']', text))
    }
    
    # 語彙の複雑さ
    word_lengths = [len(word) for word in words if word.isalpha()]
    
    # 繰り返しの分析
    word_freq = Counter(word.lower() for word in words if word.isalpha())
    most_common_word_freq = word_freq.most_common(1)[0][1] if word_freq else 0
    
    # 数字・記号の使用
    numbers = len(re.findall(r'\d+', text))
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:()\'"/-]', text))
    
    # 読みやすさ指標
    try:
        flesch_score = flesch_reading_ease(text)
        fk_grade = flesch_kincaid_grade(text)
    except:
        flesch_score = 0
        fk_grade = 0
    
    # 文の構造
    sentence_lengths = [len(s.split()) for s in sentences if s]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'char_count': len(text),
        'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
        'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'vocab_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0,
        'flesch_score': flesch_score,
        'fk_grade': fk_grade,
        'most_common_word_freq': most_common_word_freq,
        'numbers_count': numbers,
        'special_chars_count': special_chars,
        **punctuation_patterns,
        'sentence_length_std': np.std(sentence_lengths) if sentence_lengths else 0,
        'word_length_std': np.std(word_lengths) if word_lengths else 0
    }

def main():
    print("=== AI vs Human テキスト分析 ===\n")
    
    # データ読み込み
    train_df = pd.read_csv("train.csv")
    print(f"学習データ数: {len(train_df)}")
    print(f"ラベル分布:\n{train_df['real_text_id'].value_counts()}\n")
    
    # 全テキストの特徴を収集
    all_features_file1 = []
    all_features_file2 = []
    labels = []
    
    print("テキスト特徴を抽出中...")
    for idx, row in train_df.iterrows():
        file1, file2 = read_text_pair("train", row["id"])
        
        features1 = analyze_text_characteristics(file1)
        features2 = analyze_text_characteristics(file2)
        
        all_features_file1.append(features1)
        all_features_file2.append(features2)
        labels.append(row["real_text_id"])
        
        if idx < 3:  # 最初の3つのサンプルを表示
            print(f"\n--- Sample {idx} (Human text is file_{row['real_text_id']}) ---")
            print(f"File 1 preview: {file1[:200]}...")
            print(f"File 2 preview: {file2[:200]}...")
    
    # DataFrameに変換
    df1 = pd.DataFrame(all_features_file1)
    df2 = pd.DataFrame(all_features_file2)
    
    # Human vs AI の特徴比較
    print("\n=== Human vs AI 特徴比較 ===")
    
    # 人間のテキスト（real_text_id==1の場合はfile1、==2の場合はfile2）
    human_features = []
    ai_features = []
    
    for i, label in enumerate(labels):
        if label == 1:  # file1が人間
            human_features.append(all_features_file1[i])
            ai_features.append(all_features_file2[i])
        else:  # file2が人間
            human_features.append(all_features_file2[i])
            ai_features.append(all_features_file1[i])
    
    human_df = pd.DataFrame(human_features)
    ai_df = pd.DataFrame(ai_features)
    
    # 統計比較
    comparison_stats = []
    for feature in human_df.columns:
        human_mean = human_df[feature].mean()
        ai_mean = ai_df[feature].mean()
        human_std = human_df[feature].std()
        ai_std = ai_df[feature].std()
        
        comparison_stats.append({
            'feature': feature,
            'human_mean': human_mean,
            'ai_mean': ai_mean,
            'human_std': human_std,
            'ai_std': ai_std,
            'difference': human_mean - ai_mean,
            'ratio': human_mean / ai_mean if ai_mean != 0 else np.inf
        })
    
    comp_df = pd.DataFrame(comparison_stats)
    comp_df['abs_difference'] = comp_df['difference'].abs()
    comp_df = comp_df.sort_values('abs_difference', ascending=False)
    
    print("\n最も差のある特徴量 Top 10:")
    print(comp_df[['feature', 'human_mean', 'ai_mean', 'difference', 'ratio']].head(10))
    
    # 特徴量の分布を可視化
    plt.figure(figsize=(15, 10))
    
    # 重要な特徴量を選択して可視化
    important_features = comp_df.head(8)['feature'].tolist()
    
    for i, feature in enumerate(important_features):
        plt.subplot(2, 4, i+1)
        plt.hist(human_df[feature], alpha=0.7, label='Human', bins=20, density=True)
        plt.hist(ai_df[feature], alpha=0.7, label='AI', bins=20, density=True)
        plt.title(f'{feature}')
        plt.legend()
        plt.xlabel(feature)
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 相関分析
    print("\n=== 相関分析 ===")
    # 人間とAIの特徴量差分を計算
    diff_features = {}
    for feature in human_df.columns:
        diff_features[f'diff_{feature}'] = human_df[feature].values - ai_df[feature].values
    
    diff_df = pd.DataFrame(diff_features)
    
    # 相関の高い特徴量を表示
    correlations = []
    for col in diff_df.columns:
        # ラベルとの相関（1なら人間がfile1、0なら人間がfile2）
        binary_labels = [1 if label == 1 else 0 for label in labels]
        corr = np.corrcoef(diff_df[col], binary_labels)[0, 1]
        correlations.append({'feature': col, 'correlation': abs(corr)})
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    print("ラベルとの相関が高い差分特徴量 Top 10:")
    print(corr_df.head(10))
    
    # テキストの例を詳しく見る
    print("\n=== テキスト例の詳細分析 ===")
    for i in range(min(3, len(train_df))):
        row = train_df.iloc[i]
        file1, file2 = read_text_pair("train", row["id"])
        
        print(f"\n--- Article {row['id']} (Human is file_{row['real_text_id']}) ---")
        
        # 人間とAIのテキストを特定
        if row['real_text_id'] == 1:
            human_text, ai_text = file1, file2
        else:
            human_text, ai_text = file2, file1
        
        print(f"Human text (first 300 chars): {human_text[:300]}...")
        print(f"AI text (first 300 chars): {ai_text[:300]}...")
        
        # 特徴比較
        human_features = analyze_text_characteristics(human_text)
        ai_features = analyze_text_characteristics(ai_text)
        
        print(f"Human - Words: {human_features['word_count']}, Sentences: {human_features['sentence_count']}, Flesch: {human_features['flesch_score']:.1f}")
        print(f"AI - Words: {ai_features['word_count']}, Sentences: {ai_features['sentence_count']}, Flesch: {ai_features['flesch_score']:.1f}")
    
    print("\n=== 分析完了 ===")
    print("重要な発見:")
    print("1. 最も差のある特徴量を feature_distributions.png で確認してください")
    print("2. 相関の高い差分特徴量を使って新しいモデルを構築できます")
    print("3. 個別のテキスト例から、AIと人間の書き方の違いがわかります")

if __name__ == "__main__":
    main()