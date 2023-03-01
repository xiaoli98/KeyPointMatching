from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


class KPMTokernizer():
    def __init__(self, *args, **kwargs):
        """define the tokenizer pipeline, which consist of:
        1- token normalizers, we apply these normalizers as a pipe to all tokens
        2- pre tokenization process, or better how do we split the text
        3- a tokenizer algorithm, eventually specifying the unknown token 
        4- post tokenization process, how do we want the final output of our tokenizer
        """
        for kw, value in kwargs.items():
            if kw == "tokenizer":
                tokenizer = value
            elif kw == "normalizers":
                normalizers = value
            elif kw == "pre_tokenizer":
                pre_tokenizer = value
            elif kw == "post_processor":
                post_processor = value
            elif kw == "unk_token":
                unk_token = value
            elif kw == "decoder":
                decoder = value
            
        if unk_token is None:
            unk_token = "[UNK]"
        if tokenizer is None:
            tokenizer = Tokenizer(WordPiece(unk_token))
        if normalizers is None:
            normalizers = [NFD(), Lowercase(), StripAccents()]
        if pre_tokenizer is None:
            pre_tokenizer = Whitespace()
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
        self.tokenizer.normalizer = normalizers.Sequence(normalizers)
        self.tokenizer.pre_tokenizer = pre_tokenizer
        self.tokenizer.post_processor = post_processor
        self.tokenizer.decoder = decoder
        
    def train(self, files, save_path, trainer=WordPieceTrainer, **kwargs):
        """train the tokenizer with a trainer

        Args:
            files (string): the path of the file of text to be trained on
            save_path (string): the path to save the .json file of the train
            trainer (Trainer, optional): trainer to be used to do the tokenization. Defaults to WordPieceTrainer.
        """
        self.tokenizer.train(files, trainer(**kwargs))
        self.tokenizer.save(save_path)
        
    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    