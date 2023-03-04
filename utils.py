from re import X
from sklearn.feature_extraction.text import TfidfVectorizer
from src.track_1_kp_matching import *

def def_corpus(key_points, arguments, topic):
    
    corpus = [arg for arg in arguments]
    corpus.append(topic)
    for key in key_points:
        corpus.append(key)
        
    return corpus
 
arguments_df, key_points_df, labels_file_df = load_kpm_data("kpm_data", "train")
arg = arguments_df[arguments_df['topic']=='Assisted suicide should be a criminal offence']['argument'].to_numpy().tolist()
kp = key_points_df[key_points_df['topic']=='Assisted suicide should be a criminal offence']['key_point'].to_numpy().tolist()


cp = def_corpus(kp, arg, 'Assisted suicide should be a criminal offence')
#for c in cp:
#    print(c)       
    
vectorize = TfidfVectorizer()
X = vectorize.fit_transform(cp)
print(X)
print('..........................')
print(vectorize.get_feature_names_out())
