from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders

#from customPreTokenizer import CustomPreTokenizer

class KPMTokernizer():
    def __init__(self, pretrained = None, *args, **kwargs):
        """define the tokenizer pipeline, which consist of:
        1- token normalizers, we apply these normalizers as a pipe to all tokens
        2- pre tokenization process, or better how do we split the text
        3- a tokenizer algorithm, eventually specifying the unknown token 
        4- post tokenization process, how do we want the final output of our tokenizer
        """
        
        unk_token = None
        tokenizer = None
        normalizers_list = None
        pre_tokenizer_list = None
        post_processor = None
        decoder = None
        
        for kw, value in kwargs.items():
            if kw == "tokenizer":
                tokenizer = value
            elif kw == "normalizers":
                normalizers_list = value
            elif kw == "pre_tokenizer":
                pre_tokenizer_list = value
            elif kw == "post_processor":
                post_processor = value
            elif kw == "unk_token":
                unk_token = value
            elif kw == "decoder":
                decoder = value
        
        if unk_token is None:
            unk_token = "[UNK]"
        if tokenizer is None and pretrained is None:
            tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        elif pretrained is not None and tokenizer is not None:
            self.tokenizer = tokenizer.from_pretrained(pretrained)
            self.tokenizer.enable_padding(length=256)
            return
        else:
            print("please provide tokenizer and the pretrained, or leave empty to use the default (Tokenizer and WordPiece)")
            exit()
        if normalizers_list is None:
            normalizers_list = [NFD(), Lowercase(), StripAccents()]
        if pre_tokenizer_list is None:
            pre_tokenizer_list = [Whitespace(), Punctuation(behavior="removed")]
        if post_processor is None:
            post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", 1),
                    ("[SEP]", 2),
                ],
            )
        if decoder is None:
            decoder = decoders.WordPiece()
        
        self.tokenizer = tokenizer
        self.tokenizer.normalizer = normalizers.Sequence(normalizers_list)
        
        #cstPretoken = CustomPreTokenizer()
        #pre_tokenizer_list.append(pre_tokenizers.PreTokenizer.custom(cstPretoken))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_list)#pre_tokenizers.PreTokenizer.custom(cstPretoken) #pre_tokenizers.Sequence(pre_tokenizer_list)
        
        self.tokenizer.post_processor = post_processor
        self.tokenizer.decoder = decoder
        self.tokenizer.enable_padding(length=256)
        
    def train(self, files, save_path=None, trainer=WordPieceTrainer, **kwargs):
        """train the tokenizer with a trainer

        Args:
            files (string): the path of the file of text to be trained on
            save_path (string): the path to save the .json file of the train
            trainer (Trainer, optional): trainer to be used to do the tokenization. Defaults to WordPieceTrainer.
        """
        print("training the tokenizer...", end="")
        self.tokenizer.train(files, trainer(**kwargs))
        
        if save_path is not None:
            self.tokenizer.save(save_path)
        print("done!")
        
    def encode(self, text, text2=None):
        return self.tokenizer.encode(text, text2)

    def decode(self, ids):
        return self.tokenizer.decode(ids)



    