from random import random
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import transformers
import shap
from utils import vector_to_sentence

import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, max_lenth = 1024)
classifier  = transformers.pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,  return_all_scores=True)



def get_shap_values(out ,SRC ,EOS_WORD ='</s>'):
    src_sentences = [vector_to_sentence(out[i, :], SRC, EOS_WORD, start_from=0) for i in range(out.size(0))]
    sentence = [" ".join(src_sentences)]
    explainer = shap.Explainer(classifier)
    shap_values = explainer(sentence)
    score = classifier(sentence)[1]["score"]
    s_values = torch.tensor(shap_values.values)
    return s_values ,score




import matplotlib.pyplot as plt
import random

def plot_sentiment(socres):
        plt.plot(socres, corlor = "magenta", marker = "o", mfc = "pink" )
        plt.xticks(range(0,len(data)+1,1))#set the tick frequency on x-axis

        plt.ylabel('sentiment scores')#set the label for y axis
        plt.xlabel('index')#set the label for x-axis
        plt.title("sentiment scores")#set the title of the graph
        plt.show()#display the graph

