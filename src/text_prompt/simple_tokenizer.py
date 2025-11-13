import gzip
import html
import os
import regex as re
import urllib.request

def _download_vocab(target):
    url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.isfile(target):
        urllib.request.urlretrieve(url, target)

class SimpleTokenizer:
    def __init__(self):
        self.byte_encoder = {
            i: chr(i) for i in range(256)
        }
        vocab_path = os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz")
        _download_vocab(vocab_path)
        with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
            merges = f.read().splitlines()
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.encoder = {"<|startoftext|>": 49406, "<|endoftext|>": 49407}

    def encode(self, text):
        text = html.unescape(text)
        tokens = re.findall(r"\S+|\n", text)
        return [self.encoder.get(t, 0) for t in tokens]
