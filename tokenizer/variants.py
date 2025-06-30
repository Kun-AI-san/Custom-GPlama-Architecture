import tiktoken
import spacy

class SpacyTokenizer:
    def __init__(self, language_model="en_core_web_sm"):
        self.vocab = {}
        self.reverse_vocab = {}
        self.is_built = False
        self.nlp = spacy.load(language_model, disable=["parser", "ner", "tagger"])  # 只保留 tokenizer

    def tokenize(self, text):
        return [token.text.lower() for token in self.nlp(text) if not token.is_space]

    def build_vocab(self, texts):
        unique_tokens = set()
        for text in texts:
            tokens = self.tokenize(text)
            unique_tokens.update(tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(unique_tokens))}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.is_built = True

    def encode(self, text):
        if not self.is_built:
            raise Exception("Call build_vocab first.")
        return [self.vocab.get(token, -1) for token in self.tokenize(text)]

    def decode(self, token_ids):
        return ' '.join([self.reverse_vocab.get(idx, '[UNK]') for idx in token_ids])
    

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.is_built = False

    def build_vocab(self, texts):
        unique_tokens = set()
        for text in texts:
            tokens = text.lower().split()
            unique_tokens.update(tokens)
        self.vocab = {word: idx for idx, word in enumerate(sorted(unique_tokens))}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.is_built = True

    def encode(self, text):
        if not self.is_built:
            raise Exception("Call build_vocab first.")
        return [self.vocab.get(word, -1) for word in text.lower().split()]

    def decode(self, token_ids):
        return ' '.join([self.reverse_vocab.get(idx, '[UNK]') for idx in token_ids])


class BPE_tokenizer:
    def __init__(self, bpe_type = 'cl100k_base'):
        self.tokenizer = tiktoken.get_encoding(bpe_type)

    def encode(self, text: str):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list):
        return self.tokenizer.decode(token_ids)
    
    def vocab_size(self):
        return self.tokenizer.n_vocab
