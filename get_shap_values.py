from random import random
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import transformers
import shap
from utils import vector_to_sentence
import torch
from spacy.tokens import Doc
import spacy

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length = 2048)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
classifier  = transformers.pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)



textcat_spacy = spacy.load("en_core_web_sm")

#tokenizer.add_special_tokens(["</s>", '<unk>', '<s>', '<blank>'])





def tok_adapter(texts, return_offsets_mapping=False):
    text = texts.split(" ")
    spaces = [False for i in range(len(text))]
    #print(text)

    if texts == "":
        doc = textcat_spacy.tokenizer(texts)
        out = {"input_ids": [tok.norm for tok in doc]}
        
    else:
        text = texts.split(" ")
        spaces = [False for i in range(len(text))]
        
        for i in range(len(text)):
            if text[i] == "":
                text[i] = " "
        #print(text)
        doc = Doc(textcat_spacy.vocab, words=text, spaces=spaces)
        #print(doc[-1])
        out = {"input_ids": [tok.norm for tok in doc]}
        #print(out)
    if return_offsets_mapping:
        mapping = []
        a = 0
        for tok in doc:
            
            b = a + len(tok)
            mapping.append((a,b))
            a = b + 1
#             if (tok.idx != 0):
#                 mapping.append((tok.idx, tok.idx + len(tok)))
#             else:
#                 mapping.append((tok.idx, tok.idx + len(tok)))
                
        out["offset_mapping"] = mapping
    return out




def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
    #src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
    tokens = []
    for i in range(out.size(0)):
      for j in out[i,:]:
        word = SRC.vocab.itos[j]
        tokens.append(word)
    print("tokens",len(tokens))


    sentence = [" ".join(tokens)]
    print(len(tok_adapter(sentence[0])["input_ids"]))
    explainer = shap.Explainer(classifier,shap.maskers.Text(tok_adapter))
    
    print(sentence)
    shap_values = explainer(sentence)
    #score = classifier(sentence)
    s_values = torch.tensor(shap_values.values.T[0])
    s_values = s_values.reshape_as(out)
    print(s_values.size())
  #  s_values = s_values
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

