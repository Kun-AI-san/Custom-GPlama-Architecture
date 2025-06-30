#from tokenizers.simple_tokenizer import SimpleTokenizer
from tokenizer.spaCy_tokenizer import SpacyTokenizer

# read sample.txt
with open("data/sample.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split into sentences text
texts = content.strip().split(". ")
texts = [text.strip() for text in texts if text]

#tokenizer = SimpleTokenizer()
tokenizer = SpacyTokenizer()
tokenizer.build_vocab(texts)

sample = texts[0]
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Original:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)
