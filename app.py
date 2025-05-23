# app.py (修正案)

import streamlit as st
import pandas as pd
from io import StringIO

# nlp_processing.py から定義された関数をインポート
from nlp_processing import (
    load_sentiment_model_definition,
    load_embedding_model_for_keybert_definition,
    load_keybert_model_definition,
    load_pos_model_definition,
    load_stopwords_from_file_definition,
    analyze_sentiment_execution,
    extract_keywords_text_execution,
    tag_pos_execution
)

# --- Streamlit UI ---
st.title("日本語テキスト分析ツール (Japanese Text Analysis Tool)")
st.markdown("""
ユーザーインタビューなどの日本語テキストを入力またはアップロードして、感情分析、キーワード抽出、品詞タグ付けを実行します。
Enter or upload Japanese text (e.g., user interviews) to perform sentiment analysis, keyword extraction, and POS tagging.
""")

# --- モデルとデータのロード ---
# 実際のモデルロード処理はここで行う (キャッシュされる)
# 関数名に `_definition` をつけて、`nlp_processing.py` で定義されたものであることを明確にしています。
# 呼び出す際には `_definition` を取った名前の変数に格納するのが一般的です。

# sentiment_model_instance = load_sentiment_model_definition()
# embedding_model_instance = load_embedding_model_for_keybert_definition()
# keybert_model_instance = load_keybert_model_definition(embedding_model_instance) # 埋め込みモデルを渡す
# pos_model_instance = load_pos_model_definition()
# japanese_stopwords_loaded = load_stopwords_from_file_definition() # デフォルトの "stopwords-ja.txt" を試みる

# より安全なロード方法：各モデルのロードが成功したか確認できるようにする
# また、Streamlitのキャッシュデコレータは関数定義時に適用されているため、
# ここでは単に関数を呼び出すだけです。

with st.spinner("必要なモデルとデータをロード中です..."):
    try:
        sentiment_model_instance = load_sentiment_model_definition()
        if sentiment_model_instance:
            st.sidebar.success("感情分析モデルのロード完了")
        else:
            st.sidebar.error("感情分析モデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"感情分析モデルのロード中に致命的なエラー: {e}")
        sentiment_model_instance = None

    try:
        embedding_model_instance = load_embedding_model_for_keybert_definition()
        if embedding_model_instance:
            st.sidebar.success("KeyBERT埋め込みモデルのロード完了")
        else:
            st.sidebar.error("KeyBERT埋め込みモデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"KeyBERT埋め込みモデルのロード中に致命的なエラー: {e}")
        embedding_model_instance = None

    try:
        keybert_model_instance = load_keybert_model_definition(embedding_model_instance)
        if keybert_model_instance:
            st.sidebar.success("KeyBERTモデルのロード完了")
        elif embedding_model_instance is None: # 埋め込みモデルがない場合は KeyBERT もロードできない
             st.sidebar.warning("KeyBERTモデルは埋め込みモデルがないためロードされませんでした。")
        else:
            st.sidebar.error("KeyBERTモデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"KeyBERTモデルのロード中に致命的なエラー: {e}")
        keybert_model_instance = None

    try:
        pos_model_instance = load_pos_model_definition()
        if pos_model_instance:
            st.sidebar.success("品詞タグ付けモデルのロード完了")
        else:
            st.sidebar.error("品詞タグ付けモデルのロード失敗")
    except Exception as e:
        st.sidebar.error(f"品詞タグ付けモデルのロード中に致命的なエラー: {e}")
        pos_model_instance = None

    try:
        japanese_stopwords_loaded = load_stopwords_from_file_definition() # デフォルトファイルパス "stopwords-ja.txt"
        if japanese_stopwords_loaded:
             st.sidebar.success(f"ストップワードリストのロード完了 (件数: {len(japanese_stopwords_loaded)})")
    except Exception as e:
        st.sidebar.error(f"ストップワードリストのロード中に致命的なエラー: {e}")
        japanese_stopwords_loaded = ["これ", "それ", "あれ"] # 最小限のフォールバック


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
        final_input_text = "" # エラー時は空にする
elif input_text_area:
    final_input_text = input_text_area

# 分析結果を格納する変数をセッション状態で管理すると、再実行間で保持しやすい
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = []
if 'keyword_results' not in st.session_state:
    st.session_state.keyword_results = []
if 'pos_results' not in st.session_state:
    st.session_state.pos_results = []


if st.button("分析実行 (Run Analysis)"):
    if final_input_text.strip():
        with st.spinner("分析中です...しばらくお待ちください (Analyzing... please wait...)"):
            # NLP分析関数の呼び出し (ロード済みのモデルインスタンスを渡す)
            st.session_state.sentiment_results = analyze_sentiment_execution(final_input_text, sentiment_model_instance)
            st.session_state.keyword_results = extract_keywords_text_execution(
                final_input_text,
                keybert_model_instance,
                top_n=10, # パラメータ例
                ngram_range=(1,2), # パラメータ例
                stopwords_list=japanese_stopwords_loaded
            )
            st.session_state.pos_results = tag_pos_execution(final_input_text, pos_model_instance)
            st.success("分析が完了しました。")
    else:
        st.warning("テキストが入力されていません (Please enter some text to analyze).")
        # 結果をクリア
        st.session_state.sentiment_results = []
        st.session_state.keyword_results = []
        st.session_state.pos_results = []


# --- 結果表示セクション ---
if final_input_text.strip() and (st.session_state.sentiment_results or st.session_state.keyword_results or st.session_state.pos_results):
    st.subheader("分析結果 (Analysis Results)")

    # 感情分析結果
    st.markdown("---")
    st.subheader("感情分析 (Sentiment Analysis)")
    if sentiment_model_instance and st.session_state.sentiment_results:
        try:
            # sentiment_results の形式: [{'label': 'positive', 'score': 0.9}, ...]
            # または、Hugging Face pipeline の仕様によっては、ネストしたリスト [[{'label': 'positive', 'score': 0.9}, ...]] になることがある
            current_sent_results = st.session_state.sentiment_results
            if isinstance(current_sent_results, list) and len(current_sent_results) > 0 and isinstance(current_sent_results[0], list):
                # ネストしたリストの場合、最初の要素を取り出す
                display_sent_results = current_sent_results[0]
            else:
                display_sent_results = current_sent_results

            if not display_sent_results: # 空の場合
                 st.write("感情分析結果がありません。")
            else:
                top_sentiment = max(display_sent_results, key=lambda x: x['score'])
                st.write(f"**主要な感情 (Main Sentiment):** {top_sentiment['label']} (スコア (Score): {top_sentiment['score']:.4f})")
                sentiment_df = pd.DataFrame(display_sent_results)
                st.bar_chart(sentiment_df.set_index('label')['score'])
        except Exception as e:
            st.error(f"感情分析結果の表示中にエラー: {e}")
            st.write(f"生データ: {st.session_state.sentiment_results}") # デバッグ用
    else:
        st.info("感情分析は実行されていないか、モデルが利用できません。")

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
    else:
        st.info("キーワード抽出は実行されていないか、モデルが利用できません。")

    # 品詞タグ付け結果
    st.markdown("---")
    st.subheader("品詞タグ付け (Part-of-Speech Tagging)")
    if pos_model_instance and st.session_state.pos_results:
        try:
            pos_df = pd.DataFrame(st.session_state.pos_results)
            st.write(f"**トークン数 (Token Count):** {len(pos_df)}")
            st.dataframe(pos_df[['text', 'lemma', 'pos', 'tag']], height=300)
        except Exception as e:
            st.error(f"品詞タグ付け結果の表示中にエラー: {e}")
    else:
        st.info("品詞タグ付けは実行されていないか、モデルが利用できません。")

elif not final_input_text.strip() and not (st.session_state.sentiment_results or st.session_state.keyword_results or st.session_state.pos_results):
    st.info("テキストを入力して「分析実行」ボタンを押してください。")