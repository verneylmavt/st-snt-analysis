# ⛔️ Sentiment Analysis Model Collections

This project focuses on sentiment analysis using the IMDB dataset, where reviews are processed and labeled as positive or negative. The code handles the reading of raw review data, applies tokenization to transform text into a list of words, and builds a vocabulary with a minimum frequency threshold. It also pads and truncates the sequences to a fixed length, ensuring the data is in a suitable format for model consumption.

Three distinct neural network architectures are implemented for sentiment classification. The first model, BiRNN, employs a bidirectional LSTM that captures context from both the beginning and the end of a review to make a sentiment prediction. The second model, TextCNN, uses convolutional layers with multiple kernel sizes to extract local n-gram features, and then pools these features before making a classification. The third model, HybridCNNRNN, combines convolutional layers with a bidirectional LSTM and incorporates an attention mechanism, allowing the network to focus on the most informative parts of the review.

The project leverages pre-trained word embeddings to enhance performance. GloVe embeddings are integrated with the BiRNN and TextCNN models, while a FastText model provides embeddings for the HybridCNNRNN. Training is carried out using optimizers like Adam and AdamW, with cross-entropy loss guiding the learning process. Additionally, the model is exported to ONNX format for deployment in Streamlit.

For more information about the training process, please check the `snt-analysis.ipynb` file in the `training` folder.

[Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-snt-analysis.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-snt-analysis/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-snt-analysis.git
   cd st-snt-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Alternatively you can run `jupyter notebook demo.ipynb` for a minimal interface to quickly test the model (implemented w/ `ipywidgets`).

## ⚖️ Acknowledgement

I acknowledge the use of the **Large Movie Review Dataset (aclImdb)** provided by **Andrew L. Maas and colleagues**. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: Large Movie Review Dataset (aclImdb)
- **Source**: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Description**: This dataset contains 50,000 movie reviews from IMDb, divided into 25,000 reviews for training and 25,000 for testing, with an equal distribution of positive and negative sentiments. It is widely used for binary sentiment classification tasks.

I deeply appreciate the efforts of Andrew L. Maas and his team in making this dataset available.
