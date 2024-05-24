
from random import randint
from pydantic import BaseModel
import torch
import re,string
from model.transformer import Seq2SeqTransformer
from utils.utils import Translation
from langdetect import detect
from langdetect import detect_langs
import streamlit as st
from streamlit_option_menu import option_menu
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

en_sents = open('data/en_sents', "r", encoding='utf-8').read().splitlines()
vi_sents = open('data/vi_sents', "r", encoding='utf-8').read().splitlines()

def preprocessing(data):
    # Loại bỏ dấu câu và ký tự đặc biệt
    data = [line.translate(str.maketrans('', '', string.punctuation)) for line in data]
    # Chuyển đổi văn bản thành chữ thường
    data = [line.lower() for line in data]
    # Loại bỏ các khoảng trắng ở đầu và cuối câu
    data = [line.strip() for line in data]
    # Thay thế các khoảng trắng kép bằng một khoảng trắng đơn
    data = [re.sub("\s+", " ", line) for line in data]
    return data

en_sents = preprocessing(en_sents)
vi_sents = preprocessing(vi_sents)

DEVICE = torch.device('cuda')
TRANSFORMER_EN2VI_WEIGHTS = 'weights/en2vi.pth'
TRANSFORMER_VI2EN_WEIGHTS = 'weights/vi2en.pth'
vocab_src_en = 'weights/src_vocab_1.pkl'
vocab_tgt_vi = 'weights/tgt_vocab_1.pkl'
vocab_src_vi = 'weights/src_vocab_2.pkl'
vocab_tgt_en = 'weights/tgt_vocab_2.pkl'


EMB_SIZE = 512
NHEAD = 8 # embed_dim must be divisible by num_heads
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROP_OUT = 0.1

def load_en2vi_model():
    translation = Translation(vocab_src_en, vocab_tgt_vi, DEVICE)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = translation.len_vocab()

    # Load transformer model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)

    transformer.load_state_dict(torch.load(TRANSFORMER_EN2VI_WEIGHTS))
    transformer.eval()
    transformer = transformer.to(DEVICE)

    return transformer, translation  

def load_vi2en_model():
    translation = Translation(vocab_src_vi, vocab_tgt_en, DEVICE)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = translation.len_vocab()

    # Load transformer model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)

    transformer.load_state_dict(torch.load(TRANSFORMER_VI2EN_WEIGHTS))
    transformer.eval()
    transformer = transformer.to(DEVICE)

    return transformer, translation 

def preprocess_input(text):
    text = (lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))
    text = (lambda ele: ele.lower())
    text = (lambda ele: ele.strip())
    text = (lambda ele: re.sub("\s+", " ", ele))
    return text


def get_bleu_score():
    rand = [randint(170000, 200000) for i in range(10)]
    test_set = [
        [en_sents[i] for i in rand],
        [vi_sents[i] for i in rand]
    ]
    results = []
    for i in range(len(test_set[0])):
        transformer, translation = load_en2vi_model()
        source = test_set[0][i]
        actual = test_set[1][i]
        predict = translation.translate(transformer, source)
        bleu = bleu_score(actual, predict)
        results.append((source, actual, predict, bleu))
    return results

def get_bleu_score_vi2en():
    rand = [randint(170000, 200000) for i in range(10)]
    test_set = [
        [vi_sents[i] for i in rand],
        [en_sents[i] for i in rand]
    ]
    results = []
    for i in range(len(test_set[0])):
        transformer, translation = load_vi2en_model()
        source = test_set[0][i]
        actual = test_set[1][i]
        predict = translation.translate(transformer, source)
        bleu = bleu_score(actual, predict)
        results.append((source, actual, predict, bleu))
    return results

def bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()  
    return sentence_bleu(reference, candidate)


st.set_page_config(page_title='BLEU Score', page_icon="💩")
st.page_link("pages/2_🤖_Translation.py", label="Translation Page", icon="1️⃣")
st.title('Độ đo BLEU')

st.write('BLEU là viết tắt của Bilingual Evaluation Understudy, là phương pháp đánh giá một bản dịch dựa trên các bản dịch tham khảo, được giới thiệu trong paper BLEU: a Method for Automatic Evaluation of Machine Translation). BLEU được thiết kế để sử dụng trong dịch máy (Machine Translation), nhưng thực tế, phép đo này cũng được sử dụng trong các nhiệm vụ như tóm tắt văn bản, nhận dạng giọng nói, sinh nhãn ảnh v..v.. Bên cạnh đó phép đo này hoàn toàn có thể sử dụng để đánh giá chất lượng bản dịch của nhân viên.')

st.subheader('BLEU cho mô hình English to Vietnamese')
results = get_bleu_score()
df = pd.DataFrame(results, columns=['Source', 'Actual', 'Predicted', 'BLEU Score'])
st.dataframe(df)

st.subheader('BLEU cho mô hình Vietnamese to English')
results = get_bleu_score_vi2en()
df2 = pd.DataFrame(results, columns=['Source', 'Actual', 'Predicted', 'BLEU Score'])
st.dataframe(df2)