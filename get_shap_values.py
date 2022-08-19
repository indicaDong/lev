from random import random
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import transformers
import shap
from utils import vector_to_sentence
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length = 1024)
classifier  = transformers.pipeline("sentiment-analysis",tokenizer = tokenizer)
#tokenizer.add_special_tokens(["</s>", '<unk>', '<s>', '<blank>'])


import spacy
textcat_spacy = spacy.load("en_core_web_sm")


def tok_adapter(text, return_offsets_mapping=False):
    doc = textcat_spacy.tokenizer(text)
    out = {"input_ids": [tok.norm for tok in doc]}
    if return_offsets_mapping:
        out["offset_mapping"] = [(tok.idx, tok.idx + len(tok)) for tok in doc]
    return out

# def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
#     src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
#     sentence = [" ".join(src_sentences)]
#     print(len(tokenizer(sentence)["input_ids"][]))
#     explainer = shap.Explainer(classifier)
#     print(sentence)
#     shap_values = explainer(sentence)
#     #score = classifier(sentence)
#     s_values = torch.tensor(shap_values.values)
#     print(s_values.size())
   
#    # print(score)
#     return s_values #,score



# def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
#     #src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
#     sentence = []
#     for i in range(out.size(0)):
#       a = vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0)
#       sentence.append(a)
#       print(a)
#       print("tokenizer(a)[",len(tokenizer(a)["input_ids"]))
#       print(tokenizer.convert_ids_to_tokens(tokenizer(a)["input_ids"]))
#     sentence = [" ".join(sentence)]
#     print(sentence)
#     print("tokenizer(sentence)[",len(tokenizer(sentence)["input_ids"][0]))
#     print(tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"][0]))
#     explainer = shap.Explainer(classifier)
    
#     shap_values = explainer(sentence)
#     #score = classifier(sentence)
#     s_values = torch.tensor(shap_values.values)
#     print(s_values.size())
   
#    # print(score)
#     return s_values #,score



def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
    #src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
    tokens = []
    for i in range(out.size(0)):
      for j in i:
        tokens.append(vector_to_sentence(j, SRC, EOS_WORD, start_from=0 ))
    print("tokens",len(tokens))


    sentence = [" ".join(tokens)]
    #print(len(tokenizer(sentence)["input_ids"][0]))
    explainer = shap.Explainer(classifier,shap.maskers.Text(tok_adapter))
    
    print(sentence)
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

