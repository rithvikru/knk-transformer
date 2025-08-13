import re

token_to_id = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "tt": 11,
    "ff": 12,
    "isKnight": 13,
    "isKnave": 14,
    "says": 15,
    "not": 16,
    "and": 17,
    "or": 18,
    "imp": 19,
    "iff": 20,
    "(": 21,
    ")": 22,
    ",": 23,
    "K": 24,
    "N": 25,
    "<PAD>": 26,
    "<BOS>": 27,
    "<EOS>": 28,
    "<SEP>": 29,
    "<UNK>": 30,
}

id_to_token = {v: k for k, v in token_to_id.items()}

def tokenize(text):
    text = re.sub(r'([(),])', r' \1 ', text)
    return [t for t in text.split() if t]

def encode(text):
    ids = []
    for token in tokenize(text):
        if token in token_to_id:
            ids.append(token_to_id[token])
        else:
            ids.append(token_to_id["<UNK>"])
    return ids

def decode(ids):
    text = " ".join([id_to_token[id] for id in ids])
    text = re.sub(r'\s+([),])', r'\1', text)
    text = re.sub(r'(\()\s+', r'\1', text)
    text = re.sub(r'\)\s*,', r'),', text)
    return text.strip()