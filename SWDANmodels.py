# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings, WordEmbeddings
from torch.utils.data import Dataset
import numpy as np
from utils import BPE

# Dataset class for handling subword sentiment analysis data - SW short for SubWord
class SentimentDatasetSWDAN(Dataset):
    def __init__(self, data_path: str, tokenizer: BPE):
        ''' Sentiment Dataset Class for DAN

        Args:
            data_path (str): path to the dataset
            tokenizer (BPE): the BPE tokenizer
        '''
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.examples = read_sentiment_examples(data_path)
        
        # Count tokens and record in a tensor
        self.token_count = []
        for ex in self.examples:
            tmp_count = torch.zeros(1, tokenizer.vocab_size)
            
            # Add EOW token after each word
            tokens = []
            for word in ex.words:
                word_tokens = tokenizer.encode(word)
                tokens.extend(word_tokens)
                tokens.append(tokenizer.EOW)  # Append EOW token
            
            for token in tokens:
                tmp_count[0, token] += 1
            
            self.token_count.append(tmp_count.to(device))

        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long).to(device)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.token_count[idx], self.labels[idx]
    
class NN2_subwordDAN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, emb_size):
        super().__init__()
        # self.WordEmbeddings = read_word_embeddings(emb_pretrained)
        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
        Arguments:
            x 

        Returns:
            x {tensr} -- calculated embeddings
        '''
        # avg_emb = np.mean([self.emb_layer(word) for word in x], axis=0)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        input_indices = torch.tensor(range(x.shape[2]), dtype=torch.long).to(device)
        embeddings = self.emb_layer(input_indices)
        
        avg_emb = (torch.matmul(x, embeddings)/torch.sum(input=x, dim=2).unsqueeze(2)).squeeze(1)
        x = F.relu(self.fc1(avg_emb))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


# Three-layer fully connected neural network
class NN3_subwordDAN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, emb_size):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        input_indices = torch.tensor(range(x.shape[2]), dtype=torch.long).to(device)
        embeddings = self.emb_layer(input_indices)
        
        avg_emb = (torch.matmul(x, embeddings)/torch.sum(input=x, dim=2).unsqueeze(2)).squeeze(1)

        x = F.relu(self.fc1(avg_emb))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

