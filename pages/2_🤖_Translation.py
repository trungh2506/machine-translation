from random import randint
from pydantic import BaseModel
import torch
import re,string
from model.transformer import Seq2SeqTransformer
from utils.utils import Translation
from langdetect import detect
from langdetect import detect_langs
import streamlit as st

DEVICE = torch.device('cuda')
TRANSFORMER_EN2VI_WEIGHTS = 'weights/en2vi.pth'
TRANSFORMER_VI2EN_WEIGHTS = 'weights/vi2en.pth'
vi_sents = 'data/vi_sents'
en_sents = 'data/en_sents'
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

# Title for the page and nice icon
st.set_page_config(page_title="Translation", page_icon="🤖")
st.page_link("pages/3_💩_BLEU.py", label="BLEU Score Page", icon="2️⃣")
# Header
st.title("Dịch máy")

options = ['Phát hiện ngôn ngữ','English-to-Vietnamese', 'Vietnamese-to-English']


# Form to add your items
with st.form("my_form"):
    # Dropdown menu to select a language pair
    lang_pair = st.selectbox("Chọn cặp ngôn ngữ",
                             options, 1)
    # st.write('You selected:', options.index(lang_pair))

    # Textarea to type the source text.
    user_input = st.text_area("Văn bản nguồn")
    user_input+="."
    # Preprocess user input
    preprocess_input(user_input)

    #save the punctuation
    if user_input != '':
        punctuations = []  # Khởi tạo biến 'punctuations'
        english_sentences = []  # Khởi tạo biến 'english_sentences'
        
    # Lặp qua mỗi ký tự trong chuỗi đầu vào
        for char in user_input:
        # Kiểm tra nếu ký tự là dấu câu
            if char in [".", ",", ";", "?", "!", ":", '"' "-", "/", "[", "]", "(", ")", "{", "}"]:
            # Thêm ký tự vào mảng dấu câu
                punctuations.append(char)

        # Sử dụng regex để cắt câu khi gặp các dấu câu "!.?,:"
        sentences = re.split(r'[!?,.:;\"\']|["“”]', user_input)

        # Loại bỏ các chuỗi trống và khoảng trắng ở đầu và cuối của mỗi câu
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        # Tách từng câu thành các từ khi gặp dấu phẩy ","
        sub_sentences = []
        for sentence in sentences:
            sub_sentences.extend(sentence.split(","))
        for sub_sentence in sub_sentences:
            english_sentences.append(sub_sentence.strip()) # split on new line.
    # print(english_sentences)
    # print(punctuations)
    # transformer, translation = load_en2vi_model()
    if options.index(lang_pair) == 1:
        transformer, translation = load_en2vi_model()
    elif options.index(lang_pair) == 2:
        transformer, translation = load_vi2en_model()

    translations = []
    
    # Create a button
    submitted = st.form_submit_button("Dịch")
    # If the button pressed, print the translation
    if submitted:
        for english_sentence in english_sentences:
            translations.append(translation.translate(transformer, english_sentence))
        result = ''
        for translation, punctuation in zip(translations, punctuations):
            result += translation+punctuation
        if(user_input == ''):
            st.error('Vui lòng nhập văn bản cần dịch !')
        else:
            st.write("Kết quả dịch")
            # st.write(detect(user_input))
            st.success(result)
            # st.warning(user_input)
            # st.code(punctuations)
            # st.code(english_sentences)