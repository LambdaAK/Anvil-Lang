"""BERT's WordPiece tokenizer (uncased), in plain Python: the tokenizer of BAAI/bge-small-en-v1.5.

It follows the model's tokenizer.json: BertNormalizer (clean the text, space out CJK characters,
lowercase, strip accents), BertPreTokenizer (split on whitespace and punctuation), then WordPiece
(the longest piece of the vocabulary first, later pieces marked `##`), with [CLS] ... [SEP] around it.
"""
import unicodedata

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"


class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
        self.pad, self.unk, self.cls, self.sep = (self.vocab[t] for t in (PAD, UNK, CLS, SEP))
        self.words = {i: w for w, i in self.vocab.items()}

    def normalize(self, text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or is_control(ch):
                continue
            if is_cjk(cp):
                out.append(f" {ch} ")
            elif is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out).lower())
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def split(self, text: str) -> list[str]:
        words, cur = [], []
        for ch in text:
            if is_whitespace(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
            elif is_punctuation(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(ch)
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))
        return words

    def wordpiece(self, word: str) -> list[int]:
        if len(word) > 100:
            return [self.unk]
        out, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    out.append(self.vocab[piece])
                    break
                end -= 1
            if end == start:                    # no piece fits: the whole word is unknown
                return [self.unk]
            start = end
        return out

    def encode(self, text: str) -> list[int]:
        ids = [self.cls]
        for w in self.split(self.normalize(text)):
            ids.extend(self.wordpiece(w))
        return ids + [self.sep]

    def decode(self, ids) -> str:
        return " ".join(self.words[int(i)] for i in ids if int(i) != self.pad).replace(" ##", "")


def is_whitespace(ch: str) -> bool:
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F)
