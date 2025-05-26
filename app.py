# app.py (POSベースキーワード抽出 軽量化案)

import streamlit as st
import pandas as pd
from io import StringIO

# nlp_processing.py から定義された関数をインポート
from nlp_processing import (
    load_pos_pipeline_definition,
    load_stopwords_from_file_definition, # ストップワードはキーワード抽出で使うかもしれないので残す
    extract_keywords_from_pos_tags, # KeyBERT版から変更
    tag_pos_execution
)

# --- Streamlit UI ---
st.title("日本語テキスト分析ツール (キーワード抽出 & 品詞タグ付け)")
st.markdown("""
ユーザーインタビューなどの日本語テキストを入力またはアップロードして、キーワード抽出（品詞ベース）、品詞タグ付けを実行します。
Enter or upload Japanese text (e.g., user interviews) to perform POS-based keyword extraction and POS tagging.
""")

# --- モデルとデータのロード ---
# 品詞タグ付けモデルのみロード
with st.spinner("必要なモデルとデータをロード中です..."):
    pos_model_instance = None
    japanese_stopwords_loaded = ["これ", "それ", "あれ"]

    try:
        pos_model_instance = load_pos_pipeline_definition()
        if pos_model_instance:
            st.sidebar.success("品詞タグ付けパイプラインのロード完了")
        else:
            st.sidebar.error("品詞タグ付けパイプラインのロード失敗。機能が制限されます。")
    except Exception as e:
        st.sidebar.error(f"品詞タグ付けパイプラインのロード中に致命的なエラー: {e}")

    try:
        # ストップワードは現状の extract_keywords_from_pos_tags では未使用だが、将来的に使う可能性を考慮し残す
        japanese_stopwords_loaded = load_stopwords_from_file_definition()
        if japanese_stopwords_loaded:
             st.sidebar.success(f"ストップワードリストのロード完了 (件数: {len(japanese_stopwords_loaded)})")
    except Exception as e:
        st.sidebar.error(f"ストップワードリストのロード中に致命的なエラー: {e}")


# --- UI要素 ---
# (変更なし)
input_text_area = st.text_area("分析するテキストを入力してください (Enter text to analyze):", height=200, placeholder="ここに日本語のテキストを入力...")
uploaded_file = st.file_uploader("または、テキストファイルをアップロードしてください (Or, upload a text file):", type=["txt"])

final_input_text = ""
if uploaded_file is not None:
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        file_content = stringio.read()
        final_input_text = file_content
        st.text_area("ファイルの内容 (File content):", value=file_content, height=150, disabled=True, key="file_display")
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        final_input_text = ""
elif input_text_area:
    final_input_text = input_text_area

if 'keyword_results' not in st.session_state:
    st.session_state.keyword_results = []
if 'pos_results' not in st.session_state:
    st.session_state.pos_results = []


if st.button("分析実行 (Run Analysis)"):
    if final_input_text.strip():
        # 結果をリセット
        st.session_state.keyword_results = []
        st.session_state.pos_results = []

        with st.spinner("分析中です...しばらくお待ちください (Analyzing... please wait...)"):
            if pos_model_instance:
                st.session_state.pos_results = tag_pos_execution(final_input_text, pos_model_instance)
                # POSタグ付けの結果を使ってキーワード抽出
                st.session_state.keyword_results = extract_keywords_from_pos_tags(
                    st.session_state.pos_results, # POSタグの結果を渡す
                    top_n=10
                    # stopwords_list は extract_keywords_from_pos_tags の実装による
                )
            else:
                st.warning("品詞タグ付けモデルがロードされていないため、分析をスキップします。")
            
            st.success("分析が完了しました。")
    else:
        st.warning("テキストが入力されていません (Please enter some text to analyze).")
        st.session_state.keyword_results = []
        st.session_state.pos_results = []


# --- 結果表示セクション ---
if final_input_text.strip() and (st.session_state.keyword_results or st.session_state.pos_results):
    st.subheader("分析結果 (Analysis Results)")

    # キーワード抽出結果
    st.markdown("---")
    st.subheader("キーワード抽出 (品詞ベース)")
    if pos_model_instance and st.session_state.keyword_results: # POSモデルがないとキーワードも抽出できない
        try:
            st.write(f"**トップ {len(st.session_state.keyword_results)} キーワード (Top {len(st.session_state.keyword_results)} Keywords):**")
            # extract_keywords_from_pos_tags が [(kw, count), ...] の形式で返すと仮定
            keywords_df = pd.DataFrame(st.session_state.keyword_results, columns=["キーワード (Keyword)", "出現回数 (Frequency)"])
            st.table(keywords_df)
        except Exception as e:
            st.error(f"キーワード抽出結果の表示中にエラー: {e}")
            st.write(f"生データ: {st.session_state.keyword_results}")
    elif not pos_model_instance:
        st.info("品詞タグ付けモデルが利用できないため、キーワード抽出も実行できません。")
    elif st.session_state.keyword_results == []: # 結果が空の場合
        st.info("キーワードは見つかりませんでした。")


    # 品詞タグ付け結果
    st.markdown("---")
    st.subheader("品詞タグ付け (Part-of-Speech Tagging)")
    if pos_model_instance and st.session_state.pos_results:
        try:
            pos_df = pd.DataFrame(st.session_state.pos_results)
            st.write(f"**トークン数 (Token Count):** {len(pos_df)}")
            st.dataframe(pos_df[['text', 'pos']], height=300) # lemmaとtagは内容により省略も可
        except Exception as e:
            st.error(f"品詞タグ付け結果の表示中にエラー: {e}")
            st.write(f"生データ: {st.session_state.pos_results}")
    elif not pos_model_instance:
        st.info("品詞タグ付けモデルが利用できません。")
    elif st.session_state.pos_results == []:
        st.info("品詞タグは見つかりませんでした。")

elif not final_input_text.strip() and not (st.session_state.keyword_results or st.session_state.pos_results):
    st.info("テキストを入力して「分析実行」ボタンを押してください。")
