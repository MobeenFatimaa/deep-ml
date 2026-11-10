import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        # Split using the specified regex
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Strip whitespace and discard empty strings
        tokens = [p.strip() for p in pieces if p and p.strip()]
        
        unk_id = vocab['<|unk|>']
        # Map tokens to IDs, falling back to <|unk|> if not in vocab
        return [vocab.get(token, unk_id) for token in tokens]
        
    elif mode == 'decode':
        # Create an inverse vocabulary mapping IDs to token strings
        id_to_vocab = {v: k for k, v in vocab.items()}
        
        # Map IDs back to their token strings (defaulting to '<|unk|>' if ID is missing)
        tokens = [id_to_vocab.get(idx, '<|unk|>') for idx in text]
        
        # Join tokens with a single space
        joined = ' '.join(tokens)
        
        # Remove any space appearing before the specified punctuation characters: , . ? ! " ( ) ' : ;
        decoded = re.sub(r'\s+([,.?!()\':;"])', r'\1', joined)
        
        return decoded