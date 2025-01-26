# ⛔️ Sentiment Analysis Model Collections

This repository contains machine learning models of Sentiment Analysis, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `snt-analysis.ipynb` files in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-snt-analysis.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-snt-analysis/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps

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
