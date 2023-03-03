import os
import sys
sys.path.insert(1, "../KeyPointMatching/src")
from dataPreprocess import Data
from bert_tokenizer import KPMTokernizer

def main():
    
    corpus = "src/corpuses/my_corpus"
    data = Data()
    
    if not os.path.exists(corpus):
        data.build_corpus(outFile=corpus)
    else:
        print(f"using existing corpus {corpus}")
    file_path = corpus
    tokenizer = KPMTokernizer(pretrained="bert-base-cased")
    tokenizer.train([file_path], "./my_pretrained_bert_tok.tkn")
    
    out = tokenizer.encode("`people reach their limit when it comes to their quality of life and should be able to end their suffering. this can be done with little or no suffering by assistance and the person is able to say good bye., Assisted suicide should be a criminal offence")
    print(out.ids)
    out2 = tokenizer.decode(out.ids)
    print(out2)
    
if __name__ == "__main__":
    main()