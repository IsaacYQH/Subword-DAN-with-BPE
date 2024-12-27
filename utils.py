# utils.py
from typing import Iterable

# Byte Pair Encoding (BPE) tokenizer with End of Word (EOW) token
class BPE:
    EOW = 255  # End-of-word token (0xFF in hex)

    def __init__(self, vocab_size, encoding='utf-8'):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size (int): The size of the vocabulary.
            encoding (str): The encoding standard ('utf-8', 'utf-16', 'utf-32').
        """
        assert isinstance(vocab_size, int), "vocab_size must be an integer."
        assert encoding in ['utf-8', 'utf-16', 'utf-32'], "Unsupported encoding."
        
        self.vocab_size = vocab_size
        self.encoding = encoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[self.EOW] = b'<EOW>'  # Add EOW token to the vocab
        
        if encoding == 'utf-8':
            assert vocab_size >= 258, "Vocabulary size must be at least 258."
            self.num_merges = vocab_size - 258
        elif encoding == 'utf-16':
            self.num_merges = vocab_size - 2**16
        elif encoding == 'utf-32':
            self.num_merges = vocab_size - 2**32

    def train(self, training_words):
        """
        Train the BPE tokenizer on a list of training words.

        Args:
            training_words (Iterable[str]): The training corpus as an iterable of words.
        """
        assert isinstance(training_words, Iterable), "training_words must be iterable."
        assert all(isinstance(word, str) for word in training_words), "Each word must be a string."

        # Encode each word and add EOW at the end of each word
        tokens = []
        for word in training_words:
            tokens.extend(word.encode(self.encoding))
            tokens.append(self.EOW)  # Add EOW to separate words

        self.merges = {}
        for _ in range(self.num_merges):
            pair_counts = self.count_token_pairs(tokens)
            if not pair_counts:
                break

            # Select the most frequent pair, with tie-breaking by smallest token values
            best_pair = max(pair_counts, key=lambda x: (pair_counts[x], -x[0], -x[1]))
            new_token_idx = max(self.vocab) + 1
            self.merges[best_pair] = new_token_idx

            tokens = self.merge_tokens(tokens, best_pair, new_token_idx)
            self.vocab[new_token_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
    def encode(self, text):
        """
        Encode the input text into BPE tokens with EOW appended after each word.

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: Encoded token indices.
        """
        # Split the text into words and encode each separately, adding EOW
        tokens = []
        for word in text.split():
            word_tokens = list(word.encode(self.encoding))
            tokens.extend(word_tokens)
            tokens.append(self.EOW)  # Add EOW after each word
        
        # Perform BPE merges
        while True:
            pair_counts = self.count_token_pairs(tokens)
            if not pair_counts:
                break
            best_pair = min(pair_counts, key=lambda x: self.merges.get(x, float('inf')))
            if best_pair not in self.merges:
                break
            new_token_idx = self.merges[best_pair]
            tokens = self.merge_tokens(tokens, best_pair, new_token_idx)
        
        return tokens

    def decode(self, ids):
        """
        Decode a list of BPE token indices back to the original text with word boundaries.

        Args:
            ids (List[int]): List of token indices.

        Returns:
            str: Decoded text.
        """
        assert all(isinstance(i, int) and i >= 0 for i in ids), "Token indices must be non-negative integers."
        
        decoded_bytes = []
        for token in ids:
            if token == self.EOW:
                decoded_bytes.append(b" ")  # Insert space at word boundaries
            else:
                decoded_bytes.append(self.vocab[token])

        byte_seq = b''.join(decoded_bytes).rstrip()
        return byte_seq.decode(self.encoding, errors='replace')

    def count_token_pairs(self, tokens):
        """Count occurrences of adjacent token pairs."""
        counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens, pair, new_token_idx):
        """Merge all occurrences of the given pair in the token list."""
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged_tokens.append(new_token_idx)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
# class BPE(object):
    # def __init__(self, vocab_size, encoding:str = 'utf-8'):
    #     '''
    #     Initialize the BPE providing traning words and setting vocabulary size.

    #     Arguments:
    #         vocab_size (int): vocabulary size

    #     Keyword Arguments:
    #         encoding (str, optional): encoding standard (default: {'utf-8'})
    #     '''
    #     assert isinstance(vocab_size, int), "vocab_size should be an integer!"
    #     assert isinstance(encoding, str) and encoding in ['utf-8','utf-16','utf-32'], "Encoding standard not supported!"

    #     self.vocab_size = vocab_size
    #     self.vocab = {idx: bytes([idx]) for idx in range(256)}
    #     self.encoding = encoding
    #     if self.encoding == 'utf-8':
    #         assert vocab_size >= 2**8, "Vocab size of utf-8 should not be smaller than 256!"
    #         self.num_merges = vocab_size - 2**8
    #     elif self.encoding == 'utf-16':
    #         self.num_merges = vocab_size - 2**16
    #     elif self.encoding == 'utf-32':
    #         self.num_merges = vocab_size - 2**32
    
    # def train(self, training_words):
    #     '''
    #     Train the BPE tokenizer
        
    #     Arguments:
    #         training_words (Iterable[str] or str): words used in the training of the tokenizer.

    #     Params:
    #         ids (Iterable[int]): current merged bytes, converted to integers with base 10, of the orginal training texts.
    #         idx (int): new token waiting to be allocated a pair of old tokens.
    #         self.merges (dict): dictionary containing all merged tokens.
    #     '''
    #     assert isinstance(training_words, (str, Iterable)), "Training words are not strings!"
    #     if isinstance(training_words, Iterable):
    #         assert all(isinstance(item, str) for item in training_words), "Not all items in the iterable are strings."
    #         training_words = " ".join(training_words)

    #     tokens = training_words.encode("utf-8") # raw bytes
    #     # self.tokens = list(map(int, self.tokens)) # convert to a list of integers in range 0..255 for convenience
    #     # self.training_words = training_words
    #     # convert to a list of integers in range 0..255 for convenience and copy so we don't destroy the original list
    #     ids = list(map(int, tokens))
    #     self.merges = {} # (int, int) -> int
    #     for i in range(self.num_merges):
    #         counts = self.count_token(ids)
    #         pair = max(counts, key=counts.get)
    #         idx = 256 + i
    #         print(f"merging {pair} into a new token {idx}")
    #         ids = self.merge(ids, pair, idx)
    #         self.merges[pair] = idx
    #     # update vocab
    #     for (p0, p1), idx in self.merges.items():
    #         self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    # def encode(self, text, encoding:str = 'utf-8'):
    #     '''
    #     Given an external new text string, return list of integers (the tokens)

    #     Args:
    #         text (str)
    #         encoding (str, optional): encoding standard. Defaults to 'utf-8'.

    #     Returns:
    #         list: tokens
    #     '''
    #     assert isinstance(text, (str,Iterable)), "Input text is not string!"
    #     if isinstance(text, Iterable):
    #         assert all(isinstance(item, str) for item in text), "Not all items in the iterable are strings."
    #         text = " ".join(text)

    #     assert isinstance(encoding, str) and encoding in ['utf-8','utf-16','utf-32'], "Encoding standard not supported!"
    #     # given a string, return list of integers (the tokens)
    #     tokens = list(map(int, text.encode(encoding)))
    #     while len(tokens) >= 2:
    #         counts = self.count_token(tokens)
    #         pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
    #         if pair not in self.merges:
    #             break # nothing else can be merged
    #         idx = self.merges[pair]
    #         tokens = self.merge(tokens, pair, idx)
    #     return tokens

    # def decode(self, ids, encoding:str = 'utf-8'):
    #     '''
    #     Decode given ids/tokens (list of integers) to text strings.

    #     Args:
    #         ids (Iterable[int]): tokens
    #         encoding (str, optional): encoding standard. Defaults to 'utf-8'.

    #     Returns:
    #         str: retrieved text.
    #     '''
    #     assert isinstance(ids, Iterable), "Input is not iterable ints representing tokens!"
    #     assert all(isinstance(item, int) for item in ids) and all(i>=0 for i in ids), "Not all items in the iterable are non-negative integers."
    #     assert isinstance(encoding, str) and encoding in ['utf-8','utf-16','utf-32'], "Encoding standard not supported!"

    #     tokens = b"".join(self.vocab[idx] for idx in ids)
    #     text = tokens.decode(encoding, errors="replace")
    #     return text
    
    
    
# class BPE(object):
#     # EOW = 255  # End-of-word token (0xFF in hex)

#     def __init__(self, vocab_size, encoding: str = 'utf-8'):
#         '''
#         Initialize the BPE providing training words and setting vocabulary size.
        
#         Arguments:
#             vocab_size (int): vocabulary size

#         Keyword Arguments:
#             encoding (str, optional): encoding standard (default: {'utf-8'})
#         '''
#         assert isinstance(vocab_size, int), "vocab_size should be an integer!"
#         assert isinstance(encoding, str) and encoding in ['utf-8', 'utf-16', 'utf-32'], "Encoding standard not supported!"
        
#         self.vocab_size = vocab_size
#         self.vocab = {idx: bytes([idx]) for idx in range(256)}
#         # self.vocab[self.EOW] = b'<EOW>'  # Add EOW token to the vocab
#         self.encoding = encoding
#         if self.encoding == 'utf-8':
#             assert vocab_size >= 257, "Vocab size should be at least 257 to accommodate EOW!"
#             self.num_merges = vocab_size - 257  # Account for EOW token
#         elif self.encoding == 'utf-16':
#             self.num_merges = vocab_size - 2**16
#         elif self.encoding == 'utf-32':
#             self.num_merges = vocab_size - 2**32

#     def train(self, training_words):
#         '''
#         Train the BPE tokenizer
        
#         Arguments:
#             training_words (Iterable[str]): words used in the training of the tokenizer.
#         '''
#         assert isinstance(training_words, (str, Iterable)), "Training words are not strings!"
#         if isinstance(training_words, Iterable):
#             assert all(isinstance(item, str) for item in training_words), "Not all items in the iterable are strings."
#             training_words = " ".join(training_words)

#         tokens = training_words.encode("utf-8") # raw bytes
#         # convert to a list of integers in range 0..255 for convenience and copy so we don't destroy the original list
#         ids = list(map(int, tokens))
#         # ids = [[int(byte) for byte in word.encode(self.encoding)]+[self.EOW] for word in training_words]
#         self.merges = {}  # (int, int) -> int
#         for i in range(self.num_merges):
#             counts = self.count_token(ids)
#             pair = max(counts, key=counts.get)
#             idx = 256 + i
#             print(f"merging {pair} into a new token {idx}")
#             ids = self.merge(ids, pair, idx)
#             self.merges[pair] = idx
#         # Update vocab
#         for (p0, p1), idx in self.merges.items():
#             self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

#     def encode(self, text, encoding: str = 'utf-8'):
#         '''
#         Given an external new text string, return list of integers (the tokens)
        
#         Arguments:
#             text (str): input text
#             encoding (str, optional): encoding standard. Defaults to 'utf-8'.
        
#         Returns:
#             list: list of tokens
#         '''
#         assert isinstance(text, (str, Iterable)), "Input text is not string!"
#         if isinstance(text, Iterable):
#             assert all(isinstance(item, str) for item in text), "Not all items in the iterable are strings."
#             text = " ".join(text)

#         assert isinstance(encoding, str) and encoding in ['utf-8', 'utf-16', 'utf-32'], "Encoding standard not supported!"
#         # Given a string, return list of integers (the tokens)
#         tokens = list(map(int, text.encode(encoding)))\
#             # + [self.EOW]  # Add EOW token
#         while len(tokens) >= 2:
#             counts = self.count_token(tokens)
#             pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
#             if pair not in self.merges:
#                 break  # nothing else can be merged
#             idx = self.merges[pair]
#             tokens = self.merge(tokens, pair, idx)
#         return tokens

#     def decode(self, ids, encoding: str = 'utf-8'):
#         '''
#         Decode given ids/tokens (list of integers) to text strings.
        
#         Arguments:
#             ids (Iterable[int]): tokens
#             encoding (str, optional): encoding standard. Defaults to 'utf-8'.
        
#         Returns:
#             str: retrieved text
#         '''
#         assert isinstance(ids, Iterable), "Input is not iterable ints representing tokens!"
#         assert all(isinstance(item, int) for item in ids) and all(i >= 0 for i in ids), "Not all items in the iterable are non-negative integers."
#         assert isinstance(encoding, str) and encoding in ['utf-8', 'utf-16', 'utf-32'], "Encoding standard not supported!"

#         tokens = b"".join(self.vocab[idx] for idx in ids if idx != self.EOW)  # Ignore EOW during decoding
#         text = tokens.decode(encoding, errors="replace")
#         return text

#     def count_token(self, ids):
#         counts = {}
#         for pair in zip(ids, ids[1:]):
#             counts[pair] = counts.get(pair, 0) + 1
#         return counts

#     def merge(self, ids, pair, idx):
#         newids = []
#         i = 0
#         while i < len(ids):
#             if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
#                 newids.append(idx)
#                 i += 2
#             else:
#                 newids.append(ids[i])
#                 i += 1
#         return newids


class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

