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

    #=== 特徴量の作成とモデルの学習 ===



    # テストデータの予測と結果の保存
    save_results(results)

if __name__ == "__main__":
    main()