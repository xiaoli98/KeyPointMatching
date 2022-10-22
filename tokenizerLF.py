import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as stemmer
import numpy as np
from tqdm import tqdm
import string
from track_1_kp_matching import *
import time

def tokenizeLF(toTokenize, dictionary):
    newDic = dictionary
    
    words_toID = []
    words_mask = []   
    mask = 0
    
    stop_words = stopwords.words('english')
    #print("STOP WORDS")
    #print(stop_words)
    porter = stemmer()

    #print(toTokenize)
    filtered_words = toTokenize.lower() #to lower case
    #print("*" *40)
    #print('after lower')
    #print(filtered_words)

    filtered_words = "".join([char for char in filtered_words if (char not in string.punctuation)]) # remove punctuation 
    #print("*" *40)
    #print("after removing punctuation")
    #print(filtered_words)
    filtered_words = tokenizer(filtered_words, "english")   # tokenize
    #print("*" *40)
    #print("after tokenizer")
    #print(filtered_words)
    filtered_words = [word for word in filtered_words if not word in stop_words] # remove stopwords (try without removing during training)
    #print("*" *40)
    #print("after removing stopwords")
    #print(filtered_words)
    #print(len(filtered_words))
    stemmed_words = []

    for word in filtered_words:
        if(word != "cls" and word != "sep"):
            stemmed_words.append(porter.stem(word))
        elif word == "cls":
            stemmed_words.append("[CLS]")
        else:
            stemmed_words.append("[SEP]")
    
    #filtered_words = [porter.stem(word) for word in filtered_words if(word != "cls" and word != "sep")] #stemming
    #print("*" *40)
    #print("after stemming")
    #print(stemmed_words)
    
    
    index = len(newDic)
    for word in stemmed_words:
        if word != " ":
            if word=='[CLS]':
                words_toID.append(1)
            elif word == '[SEP]':
                words_toID.append(2)
                mask+=1
            elif word in newDic.keys():
                words_toID.append(newDic[word])
                words_mask.append(mask)
                
            # il dizionario e' efficiente se si usa la key per la ricerca
            
            # found = False
            # for w_in, index in newDic.items():
            #     if w_in == word and (word != "cl" and word != "sep"):
            #         words_toID.append(index)
            #         words_mask.append(mask)
            #         found = True
            #         break
            #     elif word == "cl":
            #         words_toID.append(1)
            #         found = True
            #         break
            #     elif word == "sep":
            #         words_toID.append(2)
            #         mask+=1
            #         found = True
            #         break
            else: 
                # convertire il dizionario in lista ogni volta che trovo una nuova parola
                # e trovare l'ultima posizione costa troppo
                # ho fatto un indice sopra che da' la lunghezza del dizionario, e se trovo un nuovo elemento me lo incrementa
                 
                # _, index = list(newDic.items())[-1]
                index += 1
                newDic[word] = index
                words_toID.append(index)
                words_mask.append(mask)
    #print(words_toID)
    #input()
    
    return newDic, words_toID, words_mask


def load_vocab_file():
    dictionary = {" ": 0, "[CLS]":1,"[SEP]":2}
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    _, index = list(dictionary.items())[-1]
    
    for word in valid_words:
        dictionary[word] = index + 1
        index+=1       

    return dictionary

def padSeq(toPad, newLeng, pad = 0):
    padded = []
    padded = toPad

    while newLeng > len(padded):
        padded.append(pad)

    return padded

def padArray(toPad, pad = 0):
    padded = []

    for f in fr:
        padded.append(padSeq(f, maxLen, pad))
        
    return padded

dic = load_vocab_file()

arguments_df, key_points_df, labels_file_df = load_kpm_data("kpm_data", "train")
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

ar = arguments_df['argument']
tp = arguments_df['topic']
toTk = []
i = 0
for a in ar:
    toTk.append("[CLS] "+a+" [SEP] "+tp[i])
    i+=1
#print(toTk)

#toTk = ["[CLS] house house On a windy winter morning, a woman looked out of the window.The only thing she saw, a garden. A smile spread across her face as she spotted Maria, her daughter, in the middle of the garden enjoying the weather. It started drizzling. Maria started dancing joyfully.She tried to wave to her daughter, but her elbow was stuck, her arm hurt, her smile turned upside down. Reality came crashing down as the drizzle turned into a storm. Maria's murdered corpse consumed her mind.On a windy winter morning, a woman looked out of the window of her jail cell. [SEP] The schoolboy squirmed. Another two minutes? He knew he should stand at attention. The drillmaster's cane loomed large.Vindhya Himachal … He grunted in discomfort. This was unbearable. He considered making a dash; after all he was in the last row. What if the master noticed? The cane loomed again. He gritted his teeth. Tava shubha … This is it. He cast his eyes around.Jaya he …He started running.Jaya he …He was almost there.Jaya he … The chorus floated from afar. He was already in the toilet, heaving a relieved sigh.", "[CLS] Unlike novels, short stories have a finite amount of time to tell a tale, introduce characters and themes, and tie it all together in a neat proverbial bow. [SEP] While novels are 200-400 pages on average, short stories tell a complete story in 10,000 words or less. They also cut to the chase and establish and resolve conflict.", ""]
fr = []
mask = []
maxLen = 128

tic = time.perf_counter()
i = 0
with tqdm(total=len(toTk)) as pbar:
    for tt in toTk:
        dic, afr, amask = tokenizeLF(tt, dic)
        fr.append(afr)
        amask.append(amask)
        
        if len(afr) > maxLen:
            maxLen = len(afr)        
            
        print("processed " + str(i) +" out of "+ str(len(toTk)) + "\r")
        i+=1 
        # pbar.update(1)
        #print(afr)
        #print("\n")
        #print(amask)
        #print("---")

tokenized = padArray(fr)
toc = time.perf_counter()
print(fr[0])
print(amask[0])
print(f"tokenized in {toc - tic:0.4f} seconds")
