from random import random
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import transformers
import shap
from utils import vector_to_sentence

import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length = 1024)
classifier  = transformers.pipeline("sentiment-analysis")
#tokenizer.add_special_tokens(["</s>", '<unk>', '<s>', '<blank>'])


import spacy
textcat_spacy = spacy.load("en_core_web_sm")

###输入一文字
# def tok_adapter(text, return_offsets_mapping=False):
#     doc = textcat_spacy.tokenizer(text)
#     out = {"input_ids": [tok.norm for tok in doc]}
#     if return_offsets_mapping:
#         out["offset_mapping"] = [(tok.idx, tok.idx + len(tok)) for tok in doc]
#     return out

def tok_adapter(text, return_offsets_mapping=False):

    text = text[0].split(' ')
    input_ids = []
    for i in text:
        doc = textcat_spacy.tokenizer(i)
        input_ids.extend([doc[0].norm])
        #out = {"input_ids": [tok.norm for tok in doc]}
    out = {"input_ids": input_ids}

def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
    #src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
    sentence = []
    print(out.size())
    for i in range(out.size(0)):
    
      for l in range(out[i,:].size(0)):
          
          word = SRC.vocab.itos[out[i][l]]
          sentence.append(word)
    print(len(sentence))

    # for i in range(out.size(0)):
    #   a = vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0)
    #   sentence.append(a)
    #   print([a])
    #   print("tokenizer(a)[",len(tok_adapter(a)["input_ids"]))
    #   print([textcat_spacy.tokenizer(a)])
    # sentence = [" ".join(sentence)]
    # print(sentence)
    # print("tokenizer(sentence)[",len(tok_adapter(sentence[0])["input_ids"]))
    # #print(tokenizer.convert_ids_to_tokens(tok_adapter(sentence)["input_ids"][0]))
    sentence = [" ".join(sentence)]
    print("tokenizer(sentence)[",len(tok_adapter(sentence[0])["input_ids"]))
    print(sentence)
    explainer = shap.Explainer(classifier,shap.maskers.Text(tok_adapter))
    
    shap_values = explainer(sentence)
    #score = classifier(sentence)
    s_values = torch.tensor(shap_values.values)
    print(s_values.size())
   
   # print(score)
    return s_values #,score




import matplotlib.pyplot as plt
import random

def plot_sentiment(socres):
        plt.plot(socres, color = "magenta", marker = "o", mfc = "pink" )
        plt.xticks(range(0,len(socres)+1,1))#set the tick frequency on x-axis

        plt.ylabel('sentiment scores')#set the label for y axis
        plt.xlabel('index')#set the label for x-axis
        plt.title("sentiment scores")#set the title of the graph
        plt.show()#display the graph

