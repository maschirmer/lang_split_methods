## libraries for splitting tet into sentences: spacy, nltk, re, stanza
## libraries for lang detect: textblob, langdetect, fasttext

# spacy: nlp = spacy.load('en'), tokens = nlp(text)
# for sent in tokens.sents:
    #print(sent.string.strip()) 

# nltk: sent_tokenize

# re: split by pattern and then delete short sentences of few characters, cleanup...

# stanza:
# import stanza
#stanza.download('en')
#nlp = stanza.Pipeline(lang='en', processors='tokenize')
# doc = nlp(t_en)
# for sentence in doc.sentences:
#     print(sentence.text)


import pandas as pd
from nltk import sent_tokenize
from datetime import datetime
import numpy as np
from langdetect import detect
import re
import spacy
import stanza
from textblob import TextBlob
import langid


DATA = "~/Datasets/temp_ds/cleaning_eval_dataset.csv"


## wrapping the different functions, taking the bare text and returning
# for lang the label of {0,1} for english or not
# for split a list of sentences

print("loading models and invariants")


nlp = spacy.load('en_core_web_sm')

#stanza.download('en')
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize')

pattern1 = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
pattern2 = "[.!?]"

def split_nltk(text):
    return sent_tokenize(text)

def split_spacy(text):
    tokens = nlp(text)
    return [x for x in tokens.sents]

def split_re_long(text):
    split = re.split(pattern1, text)
    for sen in split:
        if len(sen) < 4:
            split.remove(sen)
    return split

def split_re_short(text):
    split = re.split(pattern2, text)
    for sen in split:
        if len(sen) < 4:
            split.remove(sen)
    return split

def split_stanza(text):
    doc = nlp_stanza(text)
    return [x.text for x in doc.sentences]

def lang_ld(text):
    try:
        return detect(text)
    except:
        return "failed"

def lang_tb(text):
    try:
        b = TextBlob(text)
        return b.detect_language()
    except:
        return "failed"

def lang_li(text):
    try:
        return langid.classify(text)[0]
    except:
        return "failed"



METHODS = {
    "lang":{
        "langdetect":lang_ld,
        "textblob":lang_tb,
        "langid":lang_li
    },
    "split":{
        "nltk":split_nltk,
        "spacy":split_spacy,
        "re_long":split_re_long,
        "re_short":split_re_short,
        "stanza":split_stanza
    }
}

print("done, loading prerequisites...")

if __name__ == "__main__":

    print("entering main...")
    results_lang = pd.DataFrame(columns = ["source", "method", "runtime", "failed"]) # failed: langdetect failed or not, 0,1
    results_split = pd.DataFrame(columns = ["source", "method", "runtime", "num_sentences", "length_sentences"])
    
    print("loading dataframe...")
    
    in_df = pd.read_csv(DATA, index_col=0)
    texts = in_df["text"]
    sources = in_df["source"]
    
    for i in range(len(texts)):
        
        t = texts[i]
        source = sources[i]
        
        # compute languages on complete texts 
        for name, method in METHODS["lang"].items():
            failed = 0
            
            start = datetime.now()
            lang = method(t)
            runtime = (datetime.now() - start).total_seconds()

            if lang == "failed":
                failed = 1
            
            rl = pd.DataFrame({"source": source, "method":name, "runtime":runtime, "failed":failed}, index=[0])
            results_lang = results_lang.append(rl)
            

        for name, method in METHODS["split"].items():
           
            start = datetime.now()
            # code here
            res = method(t)
            runtime = (datetime.now() - start).total_seconds()

            # compute number of resulting sentences
            ns = len(res)

            # compute mean length of sentences
            sum_length = 0
            for sentence in res:
                sum_length += len(sentence)
            if ns != 0:
                ls = sum_length / ns
            else: ls = 0

            rs = pd.DataFrame({"source": source, "method":name, "runtime":runtime, "num_sentences":ns,"length_sentences":ls }, index=[0])
            results_split = results_split.append(rs)


    results_lang.to_csv("./lang_detecction_eval.csv")
    results_split.to_csv("./sentence_splitting_eval.csv")
