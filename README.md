# ⛔️ Sentiment Analysis Model Collections

This repository contains machine learning models of Sentiment Analysis, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing various tasks using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `.ipynb` files in the `training` folder.

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

Alternatively you can run `demo.ipynb` for a minimal setup to quickly test the model (implemented w/ `ipywidgets`).

<!-- ### Notes

- Ensure all required model files (ONNX models, vocabulary files, etc.) are placed in the appropriate directories as referenced in the app.
- If you encounter issues, check the error logs and ensure all dependencies are correctly installed. -->
<!--

## Features

- Multiple neural network models for sentiment analysis, including Bi-RNN, Text CNN, and Hybrid CNN-RNN with Attention Mechanism.
- Interactive model selection and sentiment analysis interface.
- Transparency with preprocessing steps, parameters, and architecture details displayed.

## Contributions

Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request for improvements or additional features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details. -->
