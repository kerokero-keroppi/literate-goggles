# nlp_processing.py (POSベースキーワード抽出 軽量化案)

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import Counter # For keyword counting

# KeyBERT と SentenceTransformer は不要になる

# --- モデル読み込み関数の定義 ---

@st.cache_resource
def load_pos_pipeline_definition():
    """品詞タグ付けのための Transformers Pipeline をロードして返します。"""
    try:
        model_name = "cl-tohoku/bert-base-japanese-wikipedia-cabocha-pos-sup"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        pos_tagger = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple" # サブワードをまとめる
        )
        return pos_tagger
    except Exception as e:
        st.error(f"品詞タグ付けパイプラインのロード中にエラーが発生しました: {e}")
        # エラー発生時はNoneを返すことで、app.py側で処理をスキップできるようにする
        return None

# --- 分析実行関数の定義 ---

def extract_keywords_from_pos_tags(pos_tagged_output, top_n=10, target_pos_prefixes=("名詞", "固有名詞", "形容詞")):
    """
    品詞タグ付けの結果からキーワードを抽出します。
    指定された品詞（名詞、固有名詞、形容詞など）の単語を抽出し、頻度順に上位N件を返します。
    """
    if not pos_tagged_output:
        return []

    keywords = []
    for token_info in pos_tagged_output:
        # token_info は {'text': '単語', 'pos': '品詞(-詳細品詞)'} のような辞書を期待
        # 'pos' の主要な品詞カテゴリをチェック (例: "名詞-普通名詞-一般" -> "名詞")
        main_pos = token_info.get('pos', '').split('-')[0]
        if main_pos in target_pos_prefixes:
            keywords.append(token_info['text'])

    if not keywords:
        return []

    # 単語の出現頻度をカウント
    keyword_counts = Counter(keywords)
    
    # 頻度上位のキーワードを取得
    # KeyBERTの出力形式に合わせて [(keyword, relevance_score), ...] とする
    # ここでは relevance_score を正規化された頻度とする（例）
    # もしくは、単純にカウント数でも良い
    most_common_keywords = keyword_counts.most_common(top_n)
    
    # アプリケーションの表示に合わせて整形 (キーワード, 関連度)
    # ここでは関連度をカウント数そのまま使用
    formatted_keywords = [[kw, count] for kw, count in most_common_keywords]
    
    return formatted_keywords


def tag_pos_execution(text, pos_pipeline_instance):
    """品詞タグ付けを実行します (Transformers Pipeline版)。"""
    if pos_pipeline_instance is None:
        st.warning("品詞タグ付けパイプラインが利用できません。")
        return []
    try:
        # pipelineの出力例: [{'entity_group': '名詞-普通名詞-一般', 'score': 0.999, 'word': '開発', ...}, ...]
        raw_results = pos_pipeline_instance(text)
        pos_tags = []
        for entity in raw_results:
            pos_tags.append({
                "text": entity['word'],
                "lemma": entity['word'],  # lemma は text で代用
                "pos": entity['entity_group'], # 品詞タグ (例: 名詞-普通名詞-一般)
                "tag": entity['entity_group'] # 詳細タグの代わりにposを使用
            })
        return pos_tags
    except Exception as e:
        st.error(f"品詞タグ付け (pipeline) の実行中にエラーが発生しました: {e}")
        return []

@st.cache_data
def load_stopwords_from_file_definition(filepath="stopwords-ja.txt"):
    """ストップワードファイルをロードします。"""
    # (変更なし)
    default_stopwords_for_fallback = ["これ", "それ", "あれ", "この", "その", "あの", "私", "あなた", "彼", "彼女", "です", "ます", "ました", "する", "いる", "ある", "の", "は", "が", "を", "に", "へ", "と", "も", "や", "で"]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            stopwords = [line.strip() for line in f if line.strip()]
        if not stopwords:
             st.info(f"ストップワードファイル {filepath} は空です。デフォルトの短いリストを使用します。")
             return default_stopwords_for_fallback
        return list(set(stopwords))
    except FileNotFoundError:
        st.warning(f"ストップワードファイル {filepath} が見つかりません。デフォルトの短いリストを使用します。")
        return default_stopwords_for_fallback
    except Exception as e:
        st.error(f"ストップワードファイルの読み込み中にエラーが発生しました ({filepath}): {e}")
        return default_stopwords_for_fallback
