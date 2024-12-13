import os
import json
import streamlit as st
import numpy as np
import spacy
import torch
from torch import nn
import onnx
import onnxruntime

# ----------------------
# Configuration
# ----------------------

# @st.cache_resource
def load_spacy():
    model_path = os.path.join("data", "en_core_web_sm", "en_core_web_sm-3.8.0")
    if os.path.isdir(model_path):
        return spacy.load(model_path)
    else:
        raise FileNotFoundError(f"SpaCy model not found at {model_path}. Please ensure it is correctly placed.")
spacy_en = load_spacy()

# ----------------------
# Model Definition
# ----------------------
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))
    def forward(self, inputs):
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

class HybridCNNRNN200(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 lstm_hidden_size, num_lstm_layers, dropout=0.5, **kwargs):
        super(HybridCNNRNN200, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=2 * embed_size,
                    out_channels=c,
                    kernel_size=k,
                    padding=padding
                )
            )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=sum(num_channels),
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Linear(2 * lstm_hidden_size, 1)
        self.decoder = nn.Linear(2 * lstm_hidden_size, 2)
    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(embeddings))
            conv_outputs.append(conv_out)
        conv_outputs = torch.cat(conv_outputs, dim=1)
        conv_outputs = conv_outputs.permute(0, 2, 1)

        lstm_out, _ = self.lstm(conv_outputs)
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        outputs = self.decoder(self.dropout(context_vector))
        return outputs


# ----------------------
# Model Information
# ----------------------
model_info = {
    "bi-rnn": {
        "subheader": "Model: Bi-RNN",
        "pre_processing": """
Dataset = IMDb Movie Reviews Dataset
Tokenizer = NLTK("Word Tokenizer")
Embedding Model = GloVe("6B.100d")
        """,
        "parameters": """
Batch Size = 64
Embedding Size = 100
LSTM Hidden Size = 100
Number of LSTM Layers = 2
Learning Rate = 0.01
Epochs = 5
Optimizer = Adam
Loss Function = CrossEntropyLoss
        """,
        "model_code": """
class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(Model, self).__init__(**kwargs)
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # LSTM
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        # Fully Connected Layer
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
        """
    },
    "text-cnn": {
        "subheader": "Model: Text CNN",
        "pre_processing": """
Dataset = IMDb Movie Reviews Dataset
Tokenizer = NLTK("Word Tokenizer")
Embedding Model = GloVe("6B.200d")
        """,
        "parameters": """
Batch Size = 64
Embedding Size = 200
Kernel Sizes = [3, 5, 7]
Number of Channels = [100, 100, 100]
LSTM Hidden Size = 150
Learning Rate = 0.01
Epochs = 5
Optimizer = Adam
Loss Function = CrossEntropyLoss
        """,
        "model_code": """
class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(Model, self).__init__(**kwargs)
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Constant Embedding Layer
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        # Dropout Layer
        self.dropout = nn.Dropout(0.5)
        # Fully Connected Layer
        self.decoder = nn.Linear(sum(num_channels), 2)
        # Pooling Layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        # ReLU Activation
        self.relu = nn.ReLU()
        # Convolutional Layer
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))
            
    def forward(self, inputs):
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
        """
    },
    "hybrid_cnn-rnn": {
        "subheader": "Model: Hybrid CNN-RNN w/ Attention Mechanism",
        "pre_processing": """
Dataset = IMDb Movie Reviews Dataset
Tokenizer = spacy("en_core_web_sm")
Embedding Model = FastText("wiki-news-subwords-300")
        """,
        "parameters": """
Batch Size = 64
Embedding Size = 300
Kernel Sizes = [3, 5, 7]
Number of Channels = [100, 100, 100]
LSTM Hidden Size = 150
Number of LSTM Layers = 2
Dropout Rate = 0.5
Learning Rate = 0.0005
Epochs = 10
Optimizer = AdamW
Weight Decay = 0.01
Loss Function = CrossEntropyLoss
Learning Rate Scheduler = ReduceLROnPlateau
Hyperparameter Tuning: Grid Sampler
        """,
        "model_code": """
class Model(nn.Module): 
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                lstm_hidden_size, num_lstm_layers, dropout=0.5, **kwargs):
        super(Model, self).__init__(**kwargs)
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Constant Embedding Layer
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        # Convolutional Layer
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=2 * embed_size,
                    out_channels=c,
                    kernel_size=k,
                    padding=padding
                )
            )
        # ReLU Activation
        self.relu = nn.ReLU()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=sum(num_channels),
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        # Attention Layer
        self.attention = nn.Linear(2 * lstm_hidden_size, 1)
        # Fully Connected Layer
        self.decoder = nn.Linear(2 * lstm_hidden_size, 2)

    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(embeddings))
            conv_outputs.append(conv_out)
        conv_outputs = torch.cat(conv_outputs, dim=1)
        conv_outputs = conv_outputs.permute(0, 2, 1)

        lstm_out, _ = self.lstm(conv_outputs)
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        outputs = self.decoder(self.dropout(context_vector))
        return outputs
        """,
        "forward_pass": {
            "Embedding": r'''
        \mathbf{h}_i = \begin{bmatrix} \mathbf{e}(x_i) \\ \mathbf{e}_c(x_i) \end{bmatrix}, \quad \tilde{\mathbf{h}}_i = \text{Dropout}(\mathbf{h}_i) \\~~\\
        \tilde{\mathbf{H}} = [\tilde{\mathbf{h}}_1, \tilde{\mathbf{h}}_2, \dots, \tilde{\mathbf{h}}_T]^\top \in \mathbb{R}^{2d \times T}
        ''',
            "Convolution": r'''
        \mathbf{y}_m = \text{ReLU}(\mathbf{W}_m * \tilde{\mathbf{H}} + \mathbf{b}_m), \quad m = 1, 2, \dots, M \\~~\\
        \mathbf{Y} = [\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_M] \in \mathbb{R}^{\left( \sum_{m=1}^M c_m \right) \times T} \\~~\\
        \mathbf{Y}' = \mathbf{Y}^\top \in \mathbb{R}^{T \times \sum_{m=1}^M c_m}
        ''',
            "Recurrent (BiLSTM)": r'''
        \mathbf{H}_{\text{LSTM}} = \text{BiLSTM}(\mathbf{Y}')
        ''',
            "Attention": r'''
        \alpha_t = \frac{\exp(\mathbf{w}^\top \mathbf{h}_t + b_a)}{\sum_{k=1}^T \exp(\mathbf{w}^\top \mathbf{h}_k + b_a)} \\~~\\
        \mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t
        ''',
            "Decoder": r'''
        \mathbf{o} = \mathbf{W}_d \cdot \text{Dropout}(\mathbf{c}) + \mathbf{b}_d
        ''',
            "Output": r'''
        \hat{\mathbf{y}} = \text{Softmax}(\mathbf{o}) \\~~\\
        '''
        }
    }
}

# @st.cache_resource
def load_model(model_name, vocab):
    try:
        model_path = os.path.join("models", str(model_name), "model-q.onnx")
        net = onnx.load(model_path)
        onnx.checker.check_model(net)
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}. Please ensure 'model-state.pth' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    ort_session = onnxruntime.InferenceSession(model_path)
    return ort_session

# @st.cache_resource
def load_vocab(model_name):
    try:
        model_path = os.path.join("models", model_name, "vocab-dict.json")
        with open(model_path, 'r') as json_file:
            vocab = json.load(json_file)
            return vocab
    except FileNotFoundError:
        st.error(f"Vocabulary file not found for {model_name}. Please ensure 'vocab.pkl' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the vocabulary for {model_name}: {e}")
        st.stop()

# ----------------------
# Prediction Function
# ----------------------
def tokenizer(text):
    return [
        tok.text.lower() 
        for tok in spacy_en.tokenizer(text) 
        if not tok.is_punct and not tok.is_space
    ]

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def predict_sentiment(ort_session, vocab, sequence, max_length=100):
    tokens = sequence.strip().split()
    token_indices = [vocab.get(token, 0) for token in tokens]
    
    if len(token_indices) < max_length:
        token_indices += [0] * (max_length - len(token_indices))
    else:
        token_indices = token_indices[:max_length]
    
    input_array = np.array(token_indices, dtype=np.int64).reshape(1, -1)
    ort_inputs = {'input': input_array}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    probabilities = softmax(logits, axis=1)
    confidence = np.max(probabilities, axis=1)[0]
    label = np.argmax(probabilities, axis=1)[0]
    sentiment = 'positive' if label == 1 else 'negative'
    return sentiment, confidence

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Sentiment Analysis")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    
    vocab = load_vocab(model)
    net = load_model(model, vocab)
    
    st.subheader(model_info[model]["subheader"])
    user_input = st.text_area("Enter Text Here:")
    
    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner('Analyzing...'):
                sentiment, confidence = predict_sentiment(net, vocab, user_input, max_length=32)
            if sentiment == 'positive':
                st.success(f"**Sentiment:** {sentiment.capitalize()}")
            else:
                st.error(f"**Sentiment:** {sentiment.capitalize()}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
        else:
            st.warning("Please enter some text for analysis.")
            
    st.feedback("thumbs")
    st.warning("""Disclaimer: This model has been quantized for optimization.
            Check here for more details: [GitHub Repo🐙](https://github.com/verneylmavt/nlp/tree/main/Final%20Project)""")
    # st.link_button("GitHub Repository", "https://streamlit.io/gallery", icon="🐙")
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key.capitalize())
            st.latex(value)
    else: pass

if __name__ == "__main__":
    main()