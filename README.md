# sentix
## Sentiment Analysis and OCR Web Application

## Overview

This is a Flask-based web application that provides multiple functionalities:
- **Sentiment Analysis**: Analyze the sentiment (positive, neutral, negative) of a given text or uploaded file using Azure Text Analytics.
- **Text Summarization**: Summarize a document using Azure's abstractive summarization capabilities.
- **Entity Linking**: Identify and link entities within a document to appropriate Wikipedia/LinkedIn pages.
- **Optical Character Recognition (OCR)**: Extract text from images using Azure Vision's OCR capabilities.

The application is hosted on Azure and uses Azure Cognitive Services for various AI-driven functionalities.

## Features

1. **Sentiment Analysis**:
   - Analyze sentiment in input text or uploaded documents (Excel).
   - Visualize the sentiment distribution with pie charts and average confidence scores with bar charts.
  
2. **Text Summarization**:
   - Summarizes long texts using Azure's abstractive summarization API.
   
3. **Entity Linking**:
   - Links entities in the input text to external knowledge bases like Wikipedia.
   
4. **OCR (Optical Character Recognition)**:
   - Extracts text from uploaded images in common formats (PNG, JPG, JPEG, GIF) and displays recognized text along with its position in the image.

## Technologies Used

- **Flask**: Python web framework.
- **Azure Cognitive Services**: For Text Analytics, Vision API, and Sentiment Analysis.
- **Pandas**: For data manipulation.
- **Seaborn & Matplotlib**: For generating sentiment analysis graphs.
- **HTML/CSS**: Frontend design for templates.
- **GitHub**: Source code repository.
- **Azure App Service**: Hosting platform.

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name```
2. **Create a virtual environment (optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables: You will need to set up your Azure credentials for Text Analytics and Vision API**:
   ```bash
   export LANG_KEY="your_azure_text_analytics_key"
   export LANG_ENDPOINT="your_azure_text_analytics_endpoint"
   export VISION_KEY="your_azure_vision_key"
   export VISION_ENDPOINT="your_azure_vision_endpoint"
   ```
5. **Run the application**:
   ```bash
   Run the app:
   ```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
