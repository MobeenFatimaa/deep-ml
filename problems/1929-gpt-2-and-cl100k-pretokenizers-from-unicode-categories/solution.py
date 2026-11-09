import unicodedata

def is_letter(ch):
    return bool(ch) and unicodedata.category(ch).startswith("L")

def is_number(ch):
    return bool(ch) and unicodedata.category(ch).startswith("N")

def is_whitespace_char(ch):
    if not ch:
        return False
    if ch in ("\x1c", "\x1d", "\x1e", "\x1f"):
        return False
    return ch.isspace()

def is_other(ch):
    return not is_letter(ch) and not is_number(ch) and not is_whitespace_char(ch)


def gpt2_split(text):
    """Return the list of GPT-2 pretokens of text."""
    if not text:
        return []
    
    tokens = []
    i = 0
    n = len(text)
    
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
    
    while i < n:
        matched = False
        
        # Rule 1: Contractions (lowercase only)
        for c in contractions:
            if text.startswith(c, i):
                tokens.append(c)
                i += len(c)
                matched = True
                break
        if matched:
            continue
            
        # Rule 2: ?\p{L}+
        if text[i] == ' ' and i + 1 < n and is_letter(text[i + 1]):
            j = i + 1
            while j < n and is_letter(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        elif is_letter(text[i]):
            j = i
            while j < n and is_letter(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Rule 3: ?\p{N}+
        if text[i] == ' ' and i + 1 < n and is_number(text[i + 1]):
            j = i + 1
            while j < n and is_number(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        elif is_number(text[i]):
            j = i
            while j < n and is_number(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Rule 4: ?[^\s\p{L}\p{N}]+
        if text[i] == ' ' and i + 1 < n and is_other(text[i + 1]):
            j = i + 1
            while j < n and is_other(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        elif is_other(text[i]):
            j = i
            while j < n and is_other(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Rule 5 & 6: Whitespace \s+(?!\S) | \s+
        if is_whitespace_char(text[i]) or text[i] in ('\r', '\n', '\t', ' '):
            j = i
            while j < n and is_whitespace_char(text[j]):
                j += 1
            length = j - i
            if j < n and not is_whitespace_char(text[j]):
                if length >= 2:
                    tokens.append(text[i:j-1])
                    i = j - 1
                else:
                    tokens.append(text[i:j])
                    i = j
            else:
                tokens.append(text[i:j])
                i = j
            continue
            
        tokens.append(text[i])
        i += 1
        
    return tokens


def cl100k_split(text):
    """Return the list of cl100k pretokens of text."""
    if not text:
        return []
    
    tokens = []
    i = 0
    n = len(text)
    
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
    
    while i < n:
        matched = False
        
        # Rule 1: Contractions (case-insensitive)
        for c in contractions:
            if text.lower().startswith(c, i):
                tokens.append(text[i:i+len(c)])
                i += len(c)
                matched = True
                break
        if matched:
            continue
            
        # Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if i < n and text[i] not in ('\r', '\n') and not is_letter(text[i]) and not is_number(text[i]) and i + 1 < n and is_letter(text[i + 1]):
            j = i + 1
            while j < n and is_letter(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        elif i < n and is_letter(text[i]):
            j = i
            while j < n and is_letter(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Rule 3: \p{N}{1,3}
        if i < n and is_number(text[i]):
            j = i
            while j < n and is_number(text[j]):
                j += 1
            length = min(3, j - i)
            tokens.append(text[i:i+length])
            i += length
            continue
            
        # Rule 4: ?[^\s\p{L}\p{N}]+[\r\n]*
        other_start = -1
        if text[i] == ' ' and i + 1 < n and is_other(text[i + 1]):
            other_start = i
            j = i + 1
            while j < n and is_other(text[j]):
                j += 1
        elif is_other(text[i]):
            other_start = i
            j = i
            while j < n and is_other(text[j]):
                j += 1
                
        if other_start != -1:
            while j < n and text[j] in ('\r', '\n'):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Rule 5: \s*[\r\n]+ (if contains \r or \n up to last \r or \n)
        if is_whitespace_char(text[i]) or text[i] in ('\r', '\n', '\t', ' '):
            j = i
            while j < n and (is_whitespace_char(text[j]) or text[j] in ('\r', '\n', '\t', ' ')):
                j += 1
            ws_chunk = text[i:j]
            if '\r' in ws_chunk or '\n' in ws_chunk:
                last_nl_idx = max(ws_chunk.rfind('\r'), ws_chunk.rfind('\n'))
                cut_pos = i + last_nl_idx + 1
                tokens.append(text[i:cut_pos])
                i = cut_pos
                continue
            else:
                # Fall through to Rule 6
                length = j - i
                if j < n and not is_whitespace_char(text[j]):
                    if length >= 2:
                        tokens.append(text[i:j-1])
                        i = j - 1
                    else:
                        tokens.append(text[i:j])
                        i = j
                else:
                    tokens.append(text[i:j])
                    i = j
                continue
                
        tokens.append(text[i])
        i += 1
        
    return tokens


def allowed_span(text, start, end, split):
    """True if text[start:end] lies inside one piece of split(text)."""
    if not (0 <= start < end <= len(text)):
        raise ValueError("Invalid span range")
        
    pieces = split(text)
    curr = 0
    for p in pieces:
        p_start = curr
        p_end = curr + len(p)
        curr = p_end
        
        if p_start <= start and end <= p_end:
            return True
            
    return False