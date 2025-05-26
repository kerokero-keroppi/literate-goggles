import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy
import ginza # spaCyがモデルをロードできれば不要な場合あり
import fugashi
import unidic_lite
import torch # 明示的なインポートは通常不要

# --- モデル読み込み関数の定義 ---
# これらの関数はモデルをロードして返すだけで、グローバルスコープでは呼び出さない

@st.cache_resource
def load_sentiment_model_definition():
    """感情分析モデルをロードして返します。"""
    try:
        # 以前のエラーログに基づき、SSL検証を無効にする環境変数が設定されている前提
        sentiment_analyzer = pipeline("sentiment-analysis", model="christian-phu/bert-finetuned-japanese-sentiment")
        return sentiment_analyzer
    except Exception as e:
        st.error(f"感情分析モデルのロード中にエラーが発生しました: {e}")
        return None

@st.cache_resource
def load_embedding_model_for_keybert_definition():
    """KeyBERT用の埋め込みモデルをロードして返します。"""
    try:
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        return model
    except Exception as e:
        st.error(f"KeyBERT用埋め込みモデルのロード中にエラーが発生しました: {e}")
        return None

@st.cache_resource
def load_keybert_model_definition(_embedding_model):
    """KeyBERTモデルをロードして返します。埋め込みモデルを引数に取ります。"""
    if _embedding_model is None:
        st.warning("KeyBERTの埋め込みモデルがロードされていません。KeyBERTモデルはロードされません。")
        return None
    try:
        kw_model = KeyBERT(model=_embedding_model)
        return kw_model
    except Exception as e:
        st.error(f"KeyBERTモデルのロード中にエラーが発生しました: {e}")
        return None

@st.cache_resource
def load_pos_model_definition():
    """品詞タグ付けモデル(GiNZA)をロードして返します。"""
    try:
        # ja_ginza_electra が正しくインストールされていることを前提
        # requirements.txt でモデルバンドル版を指定することが理想
        nlp_pos = spacy.load("ja_ginza") 
        return nlp_pos
    except OSError:
        st.error("GiNZA ja_ginza_electra モデルが見つかりません。requirements.txtでモデルバンドル版が指定されているか確認してください。")
        return None
    except Exception as e:
        st.error(f"品詞タグ付けモデルのロード中にエラーが発生しました: {e}")
        return None

# --- 分析実行関数の定義 ---
# これらの関数は、ロード済みのモデルインスタンスを引数として受け取ります

def analyze_sentiment_execution(text, model_instance):
    """感情分析を実行します。"""
    if model_instance is None:
        st.warning("感情分析モデルが利用できません。")
        return []
    try:
        results = model_instance(text, return_all_scores=True)
        return results
    except Exception as e:
        st.error(f"感情分析の実行中にエラーが発生しました: {e}")
        return []

def extract_keywords_text_execution(text, keybert_model_instance, top_n=5, ngram_range=(1, 2), stopwords_list=None):
    """キーワード抽出を実行します。"""
    if keybert_model_instance is None:
        st.warning("キーワード抽出モデルが利用できません。")
        return []

    # デフォルトストップワード (ファイルロードが失敗した場合などのフォールバック)
    default_stopwords = ["これ", "それ", "あれ", "この", "その", "あの", "私", "あなた", "彼", "彼女", "です", "ます", "ました", "する", "いる", "ある", "の", "は", "が", "を", "に", "へ", "と", "も", "や", "で"]
    if stopwords_list is None:
        stopwords_list = default_stopwords

    try:
        # e5モデルでは "query: " プレフィックスを追加
        processed_text = f"query: {text}"
        keywords = keybert_model_instance.extract_keywords(
            processed_text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stopwords_list,
            top_n=top_n,
            use_mmr=True,
            diversity=0.7
        )
        return keywords
    except Exception as e:
        st.error(f"キーワード抽出の実行中にエラーが発生しました: {e}")
        return []

def tag_pos_execution(text, pos_model_instance):
    """品詞タグ付けを実行します。"""
    if pos_model_instance is None:
        st.warning("品詞タグ付けモデルが利用できません。")
        return []
    try:
        doc = pos_model_instance(text)
        pos_tags = []
        for token in doc:
            pos_tags.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_
            })
        return pos_tags
    except Exception as e:
        st.error(f"品詞タグ付けの実行中にエラーが発生しました: {e}")
        return []

@st.cache_data # ストップワードリストはデータとしてキャッシュ
def load_stopwords_from_file_definition(filepath="stopwords-ja.txt"):
    """ストップワードファイルをロードします。"""
    default_stopwords_for_fallback = ["これ", "それ", "あれ", "この", "その", "あの", "私", "あなた", "彼", "彼女", "です", "ます", "ました", "する", "いる", "ある", "の", "は", "が", "を", "に", "へ", "と", "も", "や", "で"]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            stopwords = [line.strip() for line in f if line.strip()]
        if not stopwords: # ファイルは存在するが空の場合
             st.info(f"ストップワードファイル {filepath} は空です。デフォルトの短いリストを使用します。")
             return default_stopwords_for_fallback
        return list(set(stopwords))
    except FileNotFoundError:
        st.warning(f"ストップワードファイル {filepath} が見つかりません。デフォルトの短いリストを使用します。")
        return default_stopwords_for_fallback
    except Exception as e:
        st.error(f"ストップワードファイルの読み込み中にエラーが発生しました ({filepath}): {e}")
        return default_stopwords_for_fallback


