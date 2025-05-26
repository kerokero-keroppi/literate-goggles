# app.py (軽量化案)

import streamlit as st
import pandas as pd
from io import StringIO

# nlp_processing.py から定義された関数をインポート
from nlp_processing import (
    load_embedding_model_for_keybert_definition,
    load_keybert_model_definition,
    load_pos_model_definition,
    load_stopwords_from_file_definition,
    extract_keywords_text_execution,
    tag_pos_execution
)

# --- Streamlit UI ---
st.title("日本語テキスト分析ツール (キーワード抽出 & 品詞タグ付け)")
st.markdown("""
ユーザーインタビューなどの日本語テキストを入力またはアップロードして、キーワード抽出、品詞タグ付けを実行します。
Enter or upload Japanese text (e.g., user interviews) to perform keyword extraction and POS tagging.
""")

# --- モデルとデータのロード ---
with st.spinner("必要なモデルとデータをロード中です..."):
    embedding_model_instance = None
    keybert_model_instance = None
    pos_model_instance = None
    japanese_stopwords_loaded = ["これ", "それ", "あれ"] # フォールバック

    try:
        embedding_model_instance = load_embedding_model_for_keybert_definition()
        if embedding_model_instance:
            st.sidebar.success("KeyBERT埋め込みモデルのロード完了")
        else:
            st.sidebar.error("KeyBERT埋め込みモデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"KeyBERT埋め込みモデルのロード中に致命的なエラー: {e}")

    if embedding_model_instance: # 埋め込みモデルが成功した場合のみ KeyBERT をロード
        try:
            keybert_model_instance = load_keybert_model_definition(embedding_model_instance)
            if keybert_model_instance:
                st.sidebar.success("KeyBERTモデルのロード完了")
            else:
                st.sidebar.error("KeyBERTモデルのロード失敗")
        except Exception as e:
            st.sidebar.error(f"KeyBERTモデルのロード中に致命的なエラー: {e}")
    elif embedding_model_instance is None:
         st.sidebar.warning("KeyBERTモデルは埋め込みモデルがないためロードされませんでした。")


    try:
        pos_model_instance = load_pos_model_definition()
        if pos_model_instance:
            st.sidebar.success("品詞タグ付けモデルのロード完了")
        else:
            st.sidebar.error("品詞タグ付けモデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"品詞タグ付けモデルのロード中に致命的なエラー: {e}")

    try:
        japanese_stopwords_loaded = load_stopwords_from_file_definition()
        if japanese_stopwords_loaded:
             st.sidebar.success(f"ストップワードリストのロード完了 (件数: {len(japanese_stopwords_loaded)})")
    except Exception as e:
        st.sidebar.error(f"ストップワードリストのロード中に致命的なエラー: {e}")
        # フォールバックは既に上で定義済み


# --- UI要素 ---
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
        with st.spinner("分析中です...しばらくお待ちください (Analyzing... please wait...)"):
            st.session_state.keyword_results = extract_keywords_text_execution(
                final_input_text,
                keybert_model_instance,
                top_n=10,
                ngram_range=(1,2),
                stopwords_list=japanese_stopwords_loaded
            )
            st.session_state.pos_results = tag_pos_execution(final_input_text, pos_model_instance)
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
    st.subheader("キーワード抽出 (Keyword Extraction)")
    if keybert_model_instance and st.session_state.keyword_results:
        try:
            st.write(f"**トップ {len(st.session_state.keyword_results)} キーワード (Top {len(st.session_state.keyword_results)} Keywords):**")
            keywords_df = pd.DataFrame(st.session_state.keyword_results, columns=["キーワード (Keyword)", "関連度 (Relevance)"])
            st.table(keywords_df)
        except Exception as e:
            st.error(f"キーワード抽出結果の表示中にエラー: {e}")
            st.write(f"生データ: {st.session_state.keyword_results}")
    elif not keybert_model_instance:
        st.info("キーワード抽出モデルが利用できません。")
    else: # 結果が空の場合
        st.info("キーワード抽出は実行されましたが、結果がありません。")


    # 品詞タグ付け結果
    st.markdown("---")
    st.subheader("品詞タグ付け (Part-of-Speech Tagging)")
    if pos_model_instance and st.session_state.pos_results:
        try:
            pos_df = pd.DataFrame(st.session_state.pos_results)
            st.write(f"**トークン数 (Token Count):** {len(pos_df)}")
            st.dataframe(pos_df[['text', 'lemma', 'pos', 'tag']], height=300) # 'tag' を残すか確認
        except Exception as e:
            st.error(f"品詞タグ付け結果の表示中にエラー: {e}")
            st.write(f"生データ: {st.session_state.pos_results}")
    elif not pos_model_instance:
        st.info("品詞タグ付けモデルが利用できません。")
    else: # 結果が空の場合
        st.info("品詞タグ付けは実行されましたが、結果がありません。")


elif not final_input_text.strip() and not (st.session_state.keyword_results or st.session_state.pos_results):
    st.info("テキストを入力して「分析実行」ボタンを押してください。")
