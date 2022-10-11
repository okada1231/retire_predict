from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df_df)

    # Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Machine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':

        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

        #df = pd.read_csv(r"C:~~\ks-projects-201801.csv")
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))

        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '要約統計量':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 要約統計量の表示
            st_display_table(df.describe())
            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == 'グラフ表示':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # グラフの表示
            act = ["退職", "月給(ドル)"]
            ch = st.sidebar.selectbox("グラフのx軸", act)
            if ch == "退職":
                st_display_graph(df, str("退職"))
            if ch == "月給(ドル)":
                st_display_graph(df, str("月給(ドル)"), )
            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '学習と検証':

        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 説明変数と目的変数の設定
            X = df.drop("退職", axis=1)   # 退職列以外を説明変数にセット
            y = df["退職"]                # 退職列を目的変数にセット

            # 決定木による予測
            num = st.sidebar.number_input('深さ', min_value=1, max_value=3)
            method = ["決定木", "ランダムフォレスト"]
            learning = st.sidebar.selectbox("学習手法", method)
            
            if learning == "決定木":
                clf, tra_pred_Y, val_pred_Y, acc_scores_tra, acc_scores_val, rec_scores_tra, rec_scores_val, pre_scores_tra, pre_scores_val = ml_dtree(X, y, num)

                # 決定木のツリーを出力
                st_display_dtree(clf, list(X))

                # 正解率、再現率、適合率を表示
                st.header('訓練用データでの予測精度')
                acc_tra, rec_tra, pre_tra = st.columns(3)

                with acc_tra:
                    st.write('正解率')
                    st.subheader(round(acc_scores_tra, 5))

                with rec_tra:
                    st.write('再現率')
                    st.subheader(round(rec_scores_tra, 5))

                with pre_tra:
                    st.write('適合率')
                    st.subheader(round(pre_scores_tra, 5))
            
                st.header('検証用データでの予測精度')
                acc_val, rec_val, pre_val = st.columns(3)

                with acc_val:
                    st.write('正解率')
                    st.subheader(round(acc_scores_val, 5))

                with rec_val:
                    st.write('再現率')
                    st.subheader(round(rec_scores_val, 5))

                with pre_val:
                    st.write('適合率')
                    st.subheader(round(pre_scores_val, 5))

            if learning == "ランダムフォレスト":
                s

        else:
            st.subheader('訓練用データをアップロードしてください')

if __name__ == "__main__":
    main()

