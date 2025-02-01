#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess


# In[ ]:


import pandas as pd

import torch
from torch import nn
from d2l import torch as d2l

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
devices = d2l.try_all_gpus()


# ### BiRNN

# In[3]:


import os
import json

import nltk
from nltk.tokenize import word_tokenize

import torch
from torch import nn
from d2l import torch as d2l

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# #### Seed Setting

# ```markdown
# In here, the code sets the random seed for reproducibility PyTorch operations. This ensures consistent results by fixing the seed for both CPU and GPU computations.
# ```

# In[4]:


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Tokenization

# ```markdown
# In here, the code defines a tokenizer function that processes text by converting it to lowercase, tokenizing it into words, and filtering out non-alphanumeric tokens. This function is essential for preparing the textual data for further processing and model training.
# ```

# In[5]:


def tokenizer(text):
    return [
        tok.lower() 
        for tok in word_tokenize(text) 
        if tok.isalnum()
    ]


# #### Data Loading

# ```markdown
# In here, the code defines the read_imdb function to read the IMDB dataset from the specified directory. It processes the reviews by reading each file, decoding the text, and assigning labels based on the folder (pos for positive and neg for negative sentiments). This function is used to load both training and testing data.
# ```

# In[6]:


def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


# #### Dataset and DataLoader

# ```markdown
# In here, the code defines the load_data_imdb function, which orchestrates the loading and preprocessing of the IMDB dataset. It reads the training and testing data, tokenizes the reviews, builds the vocabulary with a minimum frequency threshold, and converts the tokens into padded sequences. The function returns data loaders for training and testing, the vocabulary, and the original test data for later use.
# ```

# In[9]:


def load_data_imdb(batch_size, num_steps=500):
    data_dir = os.path.join('../../data', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    
    train_tokens = [tokenizer(review) for review in train_data[0]]
    test_tokens = [tokenizer(review) for review in test_data[0]]
    
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    
    train_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens
    ])
    test_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
    ])
    
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size, is_train=False)
    return train_iter, test_iter, vocab, test_data[0]


# In[10]:


batch_size = 64
train_iter, test_iter, vocab, test_data = load_data_imdb(batch_size)


# In[11]:


print(len(vocab))


# #### Embedding Loading

# ```markdown
# In here, the code loads pre-trained GloVe embeddings using the d2l.TokenEmbedding class. It retrieves the embedding vectors corresponding to the tokens in the vocabulary, which are later used to initialize the embedding layer of the model.
# ```

# In[8]:


glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]


# #### Model Definition

# ```markdown
# In here, the code defines the BiRNN class, a bidirectional recurrent neural network model for sentiment analysis. The model includes an embedding layer, a bidirectional LSTM encoder, and a fully connected decoder layer that outputs the sentiment prediction. The forward method processes the input sequences through these layers to produce the final output.
# ```

# In[9]:


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        # Embedding Layer for Token Representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # BiLSTM Encoder Layer for Contextual Sequence Representation
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                                bidirectional=True)
        # Fully Connected Decoder Layer for Classification Output
        self.decoder = nn.Linear(4 * hidden_dim, 2)

    def forward(self, inputs):
        # Token Embeddings of Input Sequence
        embeddings = self.embedding(inputs.T)
        # Parameter Flattening for Optimized LSTM Execution
        self.encoder.flatten_parameters()
        # Contextual Sequence Representation w/ BiLSTM Encoder
        outputs, _ = self.encoder(embeddings)
        # Concatenation of Forward and Backward Hidden States for First and Last Time Steps
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        # Transformation of Encoded Features → Sentiment Scores
        outs = self.decoder(encoding)
        return outs


# In[11]:


embedding_dim, hidden_dim, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embedding_dim, hidden_dim, num_layers)


# #### Weight Initialization

# ```markdown
# In here, the code defines the init_weights function to initialize the weights of the model. For linear layers, it uses Xavier uniform initialization for the weights and zeros for the biases. For LSTM layers, it initializes the input-hidden and hidden-hidden weights with Xavier uniform and sets the biases to zero. This initialization helps in stabilizing the training process.
# ```

# In[10]:


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


# In[ ]:


net.apply(init_weights)


# In[12]:


net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False


# #### Training

# ```markdown
# In here, the code defines the learning rate and number of epochs for training. It sets up the optimizer as Adam with the specified learning rate and defines the loss function as cross-entropy loss without reduction, allowing for more granular loss computation. It then calls the d2l.train_ch13 function to train the model using the training and testing data loaders, the loss function, the optimizer, and the number of epochs on the available devices.
# ```

# In[13]:


lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, criterion, optimizer, num_epochs, devices)


# #### Evaluation Metrics

# ```markdown
# In here, the code defines the cal_metrics function, which evaluates the trained model on the test dataset. It sets the model to evaluation mode, iterates through the test data loader, makes predictions, and accumulates the true and predicted labels. It then computes and prints the confusion matrix components (True Positives, False Positives, True Negatives, False Negatives) and calculates metrics such as accuracy, precision, recall, and F1 score to assess the model's performance.
# ```

# In[ ]:


def cal_metrics(net, test_iter, test_texts):
    net.eval()
    device = next(net.parameters()).device

    all_preds = []
    all_labels = []
    all_texts = []
    sample_idx = 0

    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            outputs = net(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            batch_size = X.size(0)
            for i in range(batch_size):
                all_texts.append(test_texts[sample_idx])
                sample_idx += 1

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Negatives (FN): {fn}')

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    F1_Score = f1_score(all_labels, all_preds)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {F1_Score:.4f}')

    return None


# In[15]:


cal_metrics(net, test_iter, test_data)


# #### Prediction

# ```markdown
# In here, the code defines the predict_sentiment function, which takes a trained model, vocabulary, and a input sentence to predict its sentiment. The function tokenizes and encodes the input sentence, feeds it through the model, and returns 'positive' or 'negative' based on the model's prediction.
# ```

# In[ ]:


def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# In[14]:


predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is so bad')


# #### ONNX Exporting

# In[ ]:


# vocab_dict = vocab.token_to_idx
# with open(os.path.join('.', 'models', 'bi-rnn', 'vocab-dict.json'), 'w') as json_file:
#     json.dump(vocab_dict, json_file)


# In[ ]:


# dummy_sequence_length = 32
# dummy_input = torch.randint(0, len(vocab), (1, dummy_sequence_length), dtype=torch.long).to(device)

# torch.onnx.export(
#     net,
#     dummy_input,
#     os.path.join('.', 'models', 'bi-rnn', 'model.onnx'),
#     export_params=True,
#     opset_version=12,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output']
# )


# In[ ]:


# subprocess.run([
#     "python", "-m", "onnxruntime.quantization.preprocess",
#     "--input", os.path.join('.', 'models', 'bi-rnn', 'model.onnx'),
#     "--output", os.path.join('.', 'models', 'bi-rnn', 'model-p.onnx')
# ])


# In[ ]:


# quantize_dynamic(
#     model_input=os.path.join('.', 'models', 'bi-rnn', 'model-p.onnx'),
#     model_output=os.path.join('.', 'models', 'bi-rnn', 'model-q.onnx'),
#     weight_type=QuantType.QUInt8
# )


# ### TextCNN

# In[16]:


import os
import json

import nltk
from nltk.tokenize import word_tokenize

import torch
from torch import nn
from d2l import torch as d2l

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# #### Seed Setting

# ```markdown
# In here, the code sets the random seed for reproducibility PyTorch operations. This ensures consistent results by fixing the seed for both CPU and GPU computations.
# ```

# In[ ]:


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Tokenization

# ```markdown
# In here, the code defines a tokenizer function that processes text by converting it to lowercase, tokenizing it into words, and filtering out non-alphanumeric tokens. This function is essential for preparing the textual data for further processing and model training.
# ```

# In[17]:


def tokenizer(text):
    return [
        tok.lower() 
        for tok in word_tokenize(text) 
        if tok.isalnum()
    ]


# #### Data Loading

# ```markdown
# In here, the code defines the read_imdb function to read the IMDB dataset from the specified directory. It processes the reviews by reading each file, decoding the text, and assigning labels based on the folder (pos for positive and neg for negative sentiments). This function is used to load both training and testing data.
# ```

# In[ ]:


def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


# #### Dataset and DataLoader

# ```markdown
# In here, the code defines the load_data_imdb function, which orchestrates the loading and preprocessing of the IMDB dataset. It reads the training and testing data, tokenizes the reviews, builds the vocabulary with a minimum frequency threshold, and converts the tokens into padded sequences. The function returns data loaders for training and testing, the vocabulary, and the original test data for later use.
# ```

# In[ ]:


def load_data_imdb(batch_size, num_steps=500):
    data_dir = os.path.join('.', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    
    train_tokens = [tokenizer(review) for review in train_data[0]]
    test_tokens = [tokenizer(review) for review in test_data[0]]
    
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    
    train_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens
    ])
    test_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
    ])
    
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size, is_train=False)
    return train_iter, test_iter, vocab, test_data[0]


# In[18]:


batch_size = 64
train_iter, test_iter, vocab, test_data = load_data_imdb(batch_size)


# #### Embedding Loading

# ```markdown
# In here, the code loads pre-trained GloVe embeddings using the d2l.TokenEmbedding class. It retrieves the embedding vectors corresponding to the tokens in the vocabulary, which are later used to initialize the embedding layer of the model.
# ```

# In[19]:


glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]


# #### Model Definition

# ```markdown
# In here, the code defines the TextCNN class, a convolutional neural network model tailored for sentiment analysis. The model includes embedding layers for words, convolutional layers with varying kernel sizes to capture different n-gram features, an adaptive average pooling layer, a dropout layer for regularization, and a fully connected decoder layer that outputs sentiment predictions.
# ```

# In[20]:


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        # Embedding Layer for Learnable Token Representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Constant Embedding Layer for Fixed Token Representations
        self.constant_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Dropout Layer for Regularization
        self.dropout = nn.Dropout(0.5)
        # Fully Connected Decoder Layer for Classification Output
        self.decoder = nn.Linear(sum(num_channels), 2)
        # Adaptive Average Pooling Layer for Fixed-Length Feature Reduction
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Activation Layer for Non-Linear Transformations
        self.relu = nn.ReLU()
        # Convolutional Layers for Extracting Local Features
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            # Convolutional Layer for Feature Extraction w/ Kernel Size k and c Channels
            self.convs.append(nn.Conv1d(2 * embedding_dim, c, k))

    def forward(self, inputs):
        # Concatenation of Learnable and Constant Token Embeddings
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Permutation of Embedding Dimensions for Convolutional Input
        embeddings = embeddings.permute(0, 2, 1)
        # Feature Extraction w/ Convolutional Layers and Pooling
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        # Dropout Application to Encoded Features
        outputs = self.decoder(self.dropout(encoding))
        # Transformation of Encoded Features → Sentiment Scores
        return outputs


# In[22]:


embedding_dim, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
net = TextCNN(len(vocab), embedding_dim, kernel_sizes, nums_channels)


# #### Weight Initialization

# ```markdown
# In here, the code defines the init_weights function to initialize the weights of the model. For linear and convolutional layers, it uses Xavier uniform initialization for the weights and zeros for the biases. This initialization helps in stabilizing the training process by ensuring that the weights are set to appropriate starting values.
# ```

# In[21]:


def init_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# In[ ]:


net.apply(init_weights)


# In[23]:


net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False


# #### Training

# ```markdown
# In here, the code defines the learning rate and number of epochs for training. It sets up the optimizer as Adam with the specified learning rate and defines the loss function as cross-entropy loss without reduction, allowing for more granular loss computation. It then calls the d2l.train_ch13 function to train the model using the training and testing data loaders, the loss function, the optimizer, and the number of epochs on the available devices.
# ```

# In[24]:


lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, criterion, optimizer, num_epochs, devices)


# #### Evaluation Metrics

# ```markdown
# In here, the code defines the cal_metrics function, which evaluates the trained model on the test dataset. It sets the model to evaluation mode, iterates through the test data loader, makes predictions, and accumulates the true and predicted labels. It then computes and prints the confusion matrix components (True Positives, False Positives, True Negatives, False Negatives) and calculates metrics such as accuracy, precision, recall, and F1 score to assess the model's performance.
# ```

# In[ ]:


def cal_metrics(net, test_iter, test_texts):
    net.eval()
    device = next(net.parameters()).device

    all_preds = []
    all_labels = []
    all_texts = []
    sample_idx = 0

    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            outputs = net(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            batch_size = X.size(0)
            for i in range(batch_size):
                all_texts.append(test_texts[sample_idx])
                sample_idx += 1

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Negatives (FN): {fn}')

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    F1_Score = f1_score(all_labels, all_preds)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {F1_Score:.4f}')

    return None


# In[ ]:


cal_metrics(net, test_iter, test_data)


# #### Prediction

# ```markdown
# In here, the code defines the predict_sentiment function, which takes a trained model, vocabulary, and a input sentence to predict its sentiment. The function tokenizes and encodes the input sentence, feeds it through the model, and returns 'positive' or 'negative' based on the model's prediction.
# ```

# In[ ]:


def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# In[25]:


predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is so bad')


# #### ONNX Exporting

# In[ ]:


# vocab_dict = vocab.token_to_idx
# with open(os.path.join('.', 'models', 'text-cnn', 'vocab-dict.json'), 'w') as json_file:
#     json.dump(vocab_dict, json_file)


# In[ ]:


# dummy_sequence_length = 32
# dummy_input = torch.randint(0, len(vocab), (1, dummy_sequence_length), dtype=torch.long).to(device)

# torch.onnx.export(
#     net,
#     dummy_input,
#     os.path.join('.', 'models', 'text-cnn', 'model.onnx'),
#     export_params=True,
#     opset_version=12,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output']
# )


# In[ ]:


# subprocess.run([
#     "python", "-m", "onnxruntime.quantization.preprocess",
#     "--input", os.path.join('.', 'models', 'text-cnn', 'model.onnx'),
#     "--output", os.path.join('.', 'models', 'text-cnn', 'model-p.onnx')
# ])


# In[ ]:


# quantize_dynamic(
#     model_input=os.path.join('.', 'models', 'text-cnn', 'model-p.onnx'),
#     model_output=os.path.join('.', 'models', 'text-cnn', 'model-q.onnx'),
#     weight_type=QuantType.QUInt8
# )


# ### HybridCNNRNN

# In[9]:


import os
import re
import json

import numpy as np

import nltk
from nltk.corpus import stopwords
import spacy
from spacy.util import is_package
from gensim.utils import simple_preprocess
from gensim.models.fasttext import load_facebook_model

import torch
from torch import nn, optim
from d2l import torch as d2l

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


# #### Seed Setting

# ```markdown
# In here, the code sets the random seed for reproducibility PyTorch operations. This ensures consistent results by fixing the seed for both CPU and GPU computations.
# ```

# In[ ]:


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()


# #### Tokenization

# ```markdown
# In here, the code defines a tokenizer function that cleans the input text by removing HTML tags, tokenizes the text into words using simple_preprocess, removes non-alphanumeric tokens, and filters out stopwords. This function prepares the text data for model training by converting raw text into a list of meaningful tokens
# ```

# In[ ]:


try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# In[10]:


def tokenizer(text): 
    text = re.sub(r'<.*?>', '', text)
    tokens = simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


# #### Data Loading

# ```markdown
# In here, the code defines the read_imdb function to read the IMDB dataset from the specified directory. It processes the reviews by reading each file, decoding the text, and assigning labels based on the folder (pos for positive and neg for negative sentiments). This function is used to load both training and testing data.
# ```

# In[ ]:


def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


# #### Dataset and DataLoader

# ```markdown
# In here, the code defines the load_data_imdb function, which orchestrates the loading and preprocessing of the IMDB dataset. It reads the training and testing data, tokenizes the reviews, builds the vocabulary with a minimum frequency threshold, and converts the tokens into padded sequences. The function returns data loaders for training and testing, the vocabulary, and the original test data for later use.
# ```

# In[ ]:


def load_data_imdb(batch_size, num_steps=500):
    data_dir = os.path.join('.', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    
    train_tokens = [tokenizer(review) for review in train_data[0]]
    test_tokens = [tokenizer(review) for review in test_data[0]]
    
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    
    train_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens
    ])
    test_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
    ])
    
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size, is_train=False)
    return train_iter, test_iter, vocab, test_data[0]


# In[11]:


batch_size = 64
train_iter, test_iter, vocab, test_data = load_data_imdb(batch_size)


# #### Embedding Loading

# ```markdown
# In here, the code initializes an embedding matrix of zeros with dimensions corresponding to the size of the vocabulary and the embedding dimension (200). It then loads a pre-trained FastText model ('cc.en.200.bin') and retrieves the word vectors. For each token in the vocabulary, it attempts to assign the corresponding FastText vector to the embedding matrix.
# ```

# In[12]:


embedding_dim = 200
embedding_matrix = np.zeros((len(vocab), embedding_dim))


# In[13]:


# fasttext.util.download_model('en', if_exists='ignore')
# fasttext_model = fasttext.load_model('cc.en.300.bin')
# fasttext.util.reduce_model(fasttext_model, 200)
fasttext_model = load_facebook_model('cc.en.200.bin')
word_vectors = fasttext_model.wv


# In[15]:


unk_count = 0
for token in vocab.idx_to_token:
    if token not in word_vectors:
        unk_count += 1
print(f"Number of OOV words: {unk_count} out of {len(vocab)}")


# In[14]:


for i, token in enumerate(vocab.idx_to_token):
    try:
        embedding_matrix[i] = word_vectors[token]
    except KeyError:
        embedding_matrix[i] = word_vectors.get_vector(token)
        print(f"Token {token} not in vocabulary")


# In[ ]:


for i, token in enumerate(vocab.idx_to_token):
    try:
        embedding_matrix[i] = word_vectors[token]
    except KeyError:
        embedding_matrix[i] = word_vectors.get_vector(token)
        print(f"Token {token} not in vocabulary")


# In[16]:


embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)


# #### Model Definition

# ```markdown
# In here, the code defines the HybridCNNRNN class, a neural network model that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for sentiment analysis. The model includes embedding layers for words and a constant embedding, multiple convolutional layers with different kernel sizes to capture various n-gram features, a bidirectional LSTM layer to capture sequential information, an attention mechanism to focus on relevant parts of the sequence, and a fully connected decoder layer that outputs the sentiment classification.
# ```

# In[17]:


class HybridCNNRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels,
                 lstm_hidden_size, num_lstm_layers, dropout=0.5, **kwargs):
        super(HybridCNNRNN, self).__init__(**kwargs)
        # Embedding Layer for Learnable Token Representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Constant Embedding Layer for Fixed Token Representations
        self.constant_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Dropout Layer for Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Convolutional Layers for Extracting Local Features
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            # Convolutional Layer w/ Kernel Size k, Channels c, and Padding
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=2 * embedding_dim,
                    out_channels=c,
                    kernel_size=k,
                    padding=padding
                )
            )
        # Activation Layer for Non-Linear Transformations
        self.relu = nn.ReLU()
        
        # BiLSTM Layer for Contextual Sequence Representation
        self.lstm = nn.LSTM(
            input_size=sum(num_channels),
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention Layer for Weighted Context Vector Calculation
        self.attention = nn.Linear(2 * lstm_hidden_size, 1)
        # Fully Connected Decoder Layer for Classification Output
        self.decoder = nn.Linear(2 * lstm_hidden_size, 2)


    def forward(self, inputs):
        # Concatenation of Learnable and Constant Token Embeddings
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Dropout Application to Token Embeddings
        embeddings = self.dropout(embeddings)
        # Permutation of Embedding Dimensions for Convolutional Input
        embeddings = embeddings.permute(0, 2, 1)

        # Feature Extraction w/ Convolutional Layers
        conv_outputs = []
        for conv in self.convs:
            # Activation of Convolutional Outputs
            conv_out = self.relu(conv(embeddings))
            conv_outputs.append(conv_out)
        # Concatenation of Convolutional Outputs
        conv_outputs = torch.cat(conv_outputs, dim=1)
        # Permutation of Convolutional Outputs for LSTM Input
        conv_outputs = conv_outputs.permute(0, 2, 1)

        # Contextual Sequence Representation w/ BiLSTM
        lstm_out, _ = self.lstm(conv_outputs)

        # Calculation of Attention Weights for LSTM Outputs
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1)
        # Weighted Summation of LSTM Outputs Using Attention Weights
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        # Transformation of Context Vector → Sentiment Scores
        outputs = self.decoder(self.dropout(context_vector))
        return outputs


# In[19]:


embedding_dim = 200
kernel_sizes = [3, 5, 7]
num_channels = [100, 100, 100]
lstm_hidden_size = 150
num_lstm_layers = 2
dropout = 0.5

net = HybridCNNRNN(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    kernel_sizes=kernel_sizes,
    num_channels=num_channels,
    lstm_hidden_size=lstm_hidden_size,
    num_lstm_layers=num_lstm_layers,
    dropout=dropout
).to(device)


# #### Weight Initialization

# ```markdown
# In here, the code defines the init_weights function to initialize the weights of the model. For convolutional and linear layers, it applies Xavier uniform initialization to the weights and zeros to the biases. For LSTM layers, it initializes both input-hidden and hidden-hidden weights with Xavier uniform initialization and sets biases to zero. This initialization strategy helps in stabilizing the training process by ensuring that the weights start with appropriate values.
# ```

# In[18]:


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


# In[ ]:


net.apply(init_weights)


# In[20]:


net.embedding.weight.data.copy_(embedding_matrix)
net.constant_embedding.weight.data.copy_(embedding_matrix)
net.constant_embedding.weight.requires_grad = False


# #### Training Function

# ```markdown
# In here, the code defines the train function, which manages the overall training process across multiple epochs. It initializes a timer and an accumulator to track metrics. For each epoch, it iterates through the training data loader, calls the train_batch function for each batch, and accumulates the loss and accuracy. It also evaluates the model on the test data loader at the end of each epoch, updates the learning rate scheduler based on the test accuracy, and updates the training progress visualizations using the d2l.Animator class.
# ```

# In[21]:


def train_batch(net, X, y, criterion, optimizer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    l = criterion(pred, y)
    l.sum().backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


# In[ ]:


def train(net, train_iter, test_iter, criterion, optimizer, num_epochs,
              devices=d2l.try_all_gpus(), scheduler=None):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, criterion, optimizer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
                scheduler.step(test_acc)
            else:
                scheduler.step()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


# #### Training

# ```markdown
# In here, the code defines the learning rate, number of training epochs, and the device to be used for training. It initializes the optimizer as AdamW with the specified learning rate and weight decay for regularization. It also sets up a learning rate scheduler (ReduceLROnPlateau) that reduces the learning rate when the validation accuracy plateaus. The loss function is defined as cross-entropy loss without reduction, allowing for more granular loss computation. Finally, it calls the train function to commence the training process with the specified parameters and scheduler.
# ```

# In[22]:


lr, num_epochs = 0.0001, 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                 factor=0.1, patience=3,
                                                 verbose=True)
criterion = nn.CrossEntropyLoss(reduction='none')
train(net, train_iter, test_iter, criterion, optimizer, num_epochs, devices, scheduler=scheduler)


# #### Evaluation Metrics

# ```markdown
# In here, the code defines the cal_metrics function, which evaluates the trained model on the test dataset. It sets the model to evaluation mode, iterates through the test data loader, makes predictions, and accumulates the true and predicted labels. It then computes and prints the confusion matrix components (True Positives, False Positives, True Negatives, False Negatives) and calculates metrics such as accuracy, precision, recall, and F1 score to assess the model's performance.
# ```

# In[ ]:


def cal_metrics(net, test_iter, test_texts):
    net.eval()
    device = next(net.parameters()).device

    all_preds = []
    all_labels = []
    all_texts = []
    sample_idx = 0

    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            outputs = net(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            batch_size = X.size(0)
            for i in range(batch_size):
                all_texts.append(test_texts[sample_idx])
                sample_idx += 1

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Negatives (FN): {fn}')

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    F1_Score = f1_score(all_labels, all_preds)
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {F1_Score:.4f}')

    return None


# In[ ]:


cal_metrics(net, test_iter, test_data)


# #### Prediction

# ```markdown
# In here, the code defines the predict_sentiment function, which takes a trained model, vocabulary, and a input sentence to predict its sentiment. The function tokenizes and encodes the input sentence, feeds it through the model, and returns 'positive' or 'negative' based on the model's prediction.
# ```

# In[ ]:


def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


# In[ ]:


predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is so bad')


# #### ONNX Exporting

# In[ ]:


# vocab_dict = vocab.token_to_idx
# with open(os.path.join('.', 'models', 'hybrid_cnn-rnn', 'vocab-dict.json'), 'w') as json_file:
#     json.dump(vocab_dict, json_file)


# In[ ]:


# dummy_sequence_length = 32
# dummy_input = torch.randint(0, len(vocab), (1, dummy_sequence_length), dtype=torch.long).to(device)

# torch.onnx.export(
#     net,
#     dummy_input,
#     os.path.join('.', 'models', 'hybrid_cnn-rnn', 'model.onnx'),
#     export_params=True,
#     opset_version=12,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output']
# )


# In[ ]:


# subprocess.run([
#     "python", "-m", "onnxruntime.quantization.preprocess",
#     "--input", os.path.join('.', 'models', 'hybrid_cnn-rnn', 'model.onnx'),
#     "--output", os.path.join('.', 'models', 'hybrid_cnn-rnn', 'model-p.onnx')
# ])


# In[ ]:


# quantize_dynamic(
#     model_input=os.path.join('.', 'models', 'hybrid_cnn-rnn', 'model-p.onnx'),
#     model_output=os.path.join('.', 'models', 'hybrid_cnn-rnn', 'model-q.onnx'),
#     weight_type=QuantType.QUInt8
# )

