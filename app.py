from flask import Flask, render_template,redirect, url_for,request, send_file,after_this_request
import glob
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import matplotlib
matplotlib.use('Agg')
KEY = '2732d37019cf49bbaee14ceae1040b4c'
ENDPOINT = 'https://lang1234.cognitiveservices.azure.com/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Define your routes
@app.route('/')
def index():
    return render_template('home.html')

# Perform sentiment analysis using Azure Text Analytics
def perform_sentiment_analysis(documents):
    # Authenticate with Azure Text Analytics
    credential = AzureKeyCredential(KEY)
    text_analytics_client = TextAnalyticsClient(endpoint=ENDPOINT, credential=credential)

    sentiment_results = []

    # Define the chunk size for splitting the documents
    chunk_size = 10  # Adjust as needed based on the maximum allowed payload size

    # Split the documents into chunks and process each chunk separately
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        
        # Convert the input documents to the required format for this chunk
        documents_list = [{"id": str(i), "text": doc} for i, doc in enumerate(chunk)]

        # Perform sentiment analysis for this chunk
        response = text_analytics_client.analyze_sentiment(documents=documents_list, show_opinion_mining=False)

        # Extract sentiment analysis results for this chunk
        for doc in response:
            sentiment_results.append({
                'text': chunk[int(doc['id'])],
                'confidence_score_positive': doc.confidence_scores["positive"],
                'confidence_score_neutral': doc.confidence_scores["neutral"],
                'confidence_score_negative': doc.confidence_scores["negative"],
                'sentiment': doc.sentiment
            })

    return sentiment_results
def graph_analysis(df_file):
    # Calculate average confidence scores
    average_confidence_scores = df_file[['confidence_score_positive', 'confidence_score_neutral', 'confidence_score_negative']].mean(axis=0)
    
    # Bar plot for average confidence scores
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    sns.barplot(x=['positive', 'neutral', 'negative'], y=average_confidence_scores.values, hue=['positive', 'neutral', 'negative'], palette=colors, legend=False)
    plt.xlabel('Sentiment Category')
    plt.ylabel('Average Confidence Score')
    plt.title('Average Confidence Scores for Sentiment Analysis')
    for index, value in enumerate(average_confidence_scores.values):
        plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')
    plt.savefig(os.path.join('static','average_confidence_scores.png'))

    plt.close()
    
    # Pie chart for sentiment distribution
    sentiment_counts = df_file['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Categories')
    plt.savefig(os.path.join('static', 'sentiment_distribution.png'))
    plt.close()
    
@app.route('/sentimentanalysis', methods=['GET'])
def sentiment_analysis_get():
    # Clear any previously stored data if the files exist
    if os.path.exists(os.path.join('static','sentiment_analysis_results.xlsx')):
        os.remove(os.path.join('static','sentiment_analysis_results.xlsx'))
    if os.path.exists(os.path.join('static','average_confidence_scores.png')):
        os.remove(os.path.join('static','average_confidence_scores.png'))
    if os.path.exists(os.path.join('static', 'sentiment_distribution.png')):
        os.remove(os.path.join('static', 'sentiment_distribution.png'))

    return render_template('sentiment_analysis.html', sentences=[], overall_sentiment={}, document="")


        

# Route for sentiment analysis
@app.route('/sentimentanalysis', methods=['POST'])
def sentiment_analysis():
    if request.method == 'POST':
        document = request.form['document']
        file = request.files['file']
        if not file and not document:
            return redirect(url_for('sentiment_analysis_get'))
        if file:
            if file.filename != '':
                # Save the file to a temporary location
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Read the contents of the file
                Data=pd.read_excel(file_path)

                # Delete the temporary file
                os.remove(file_path)
                sentiment_results_file=perform_sentiment_analysis(Data.iloc[:,0].tolist())
                df_file=pd.DataFrame(sentiment_results_file)
                
                df_file.to_excel(os.path.join('static','sentiment_analysis_results.xlsx'), index=False)
                graph_analysis(df_file)
                '''
                @after_this_request
                def remove_file(response):
                    try:
                        # Remove the file after the response is sent
                        #os.remove(os.path.join('static','sentiment_analysis_results.xlsx'))
                        #os.remove(os.path.join('static','average_confidence_scores.png'))
                        #os.remove(os.path.join('static', 'sentiment_distribution.png'))
                    except Exception as error:
                        app.logger.error("Error removing or closing downloaded file handle", error)
                    return response
                # Render the HTML template with the images
                '''
                return render_template('sentiment_analysis.html',graph_attached=True)
                #return send_file(os.path.join('static','sentiment_analysis_results.xlsx'), as_attachment=True)

        if document:
            credential = AzureKeyCredential(KEY)
            text_analytics_client = TextAnalyticsClient(endpoint=ENDPOINT, credential=credential)
            response = text_analytics_client.analyze_sentiment(documents=[document])[0]
            # Get detailed sentiment analysis
            sentences = []
            overall_sentiment_score = {'positive': 0, 'neutral': 0, 'negative': 0,'sentiment_final':''}
            for idx, sentence in enumerate(response.sentences):
                sentences.append({
                    'text': sentence.text,
                    'confidence_scores': sentence.confidence_scores,
                    'sentiment': sentence.sentiment
                })
            # Update overall sentiment score
                overall_sentiment_score['positive'] += sentence.confidence_scores.positive
                overall_sentiment_score['neutral'] += sentence.confidence_scores.neutral
                overall_sentiment_score['negative'] += sentence.confidence_scores.negative

            # Calculate average sentiment
            num_sentences = len(response.sentences)
            overall_sentiment = {
                'positive': round(overall_sentiment_score['positive'] / num_sentences,3),
                'neutral': round(overall_sentiment_score['neutral'] / num_sentences,3),
                'negative': round(overall_sentiment_score['negative'] / num_sentences,3),
                'sentiment_final': response.sentiment
            }

            return render_template('sentiment_analysis.html', document=document, sentences=sentences, overall_sentiment=overall_sentiment)
        return redirect(url_for('sentiment_analysis_get'))
    return redirect(url_for('sentiment_analysis_get'))



@app.route('/summarization', methods=['POST','GET'])
def summarization():
    if request.method == 'POST':
        # Get text from the form
        document = request.form['document']
        if document:
        # Initialize Text Analytics client
            text_analytics_client = TextAnalyticsClient(
                endpoint=ENDPOINT,
                credential=AzureKeyCredential(KEY)
            )

        # Call Azure Text Analytics service for abstractive summarization
            poller = text_analytics_client.begin_abstract_summary([document])
            abstract_summary_results = poller.result()

        # Extract and display the summary
            summaries = []
            for result in abstract_summary_results:
                if result.kind == "AbstractiveSummarization":
                    summaries.extend([summary.text for summary in result.summaries])
                elif result.is_error is True:
                    error_message = f"Error with code '{result.error.code}' and message '{result.error.message}'"
                    summaries.append(error_message)

            return render_template('summarization.html', document=document, summary=summaries[0])
        return render_template('summarization.html')
    # If it's a GET request or the POST request doesn't have the required data
    return render_template('summarization.html')

@app.route('/entitylinking', methods=['GET', 'POST'])
def entity_linking():
    if request.method == 'POST':
        document = request.form['document']
        if document:

            # Authenticate with Azure Text Analytics
            credential = AzureKeyCredential(KEY)
            text_analytics_client = TextAnalyticsClient(endpoint=ENDPOINT, credential=credential)

            # Perform entity linking
            response = text_analytics_client.recognize_linked_entities(documents=[document])[0]
            print(response)
            # Get the linked entities
            entities = []
            for entity in response.entities:
                entities.append({"name": entity.name, "url": entity.url})

            return render_template('entity_linking.html', document=document, entities=entities)
        return render_template('entity_linking.html')
    return render_template('entity_linking.html')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ocr', methods=['GET'])
def ocr_get():
    if any(glob.glob(os.path.join('static', 'ocr_analysis.*'))):
        file_path = glob.glob(os.path.join('static', 'ocr_analysis.*'))[0]
        os.remove(file_path)
    
    return render_template('ocr.html',file_exist=False)
@app.route('/ocr', methods=['POST'])
def ocr():
    file_exist = False
    if request.method == 'POST':
        try:
            # Retrieve Azure Computer Vision credentials from environment variables
            endpoint = 'https://vision1234.cognitiveservices.azure.com/'
            key = 'f11aecc9873449d893bedc3829a2145f'

            # Create an Image Analysis client
            client = ImageAnalysisClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )

            # Check if an image file was uploaded
            if 'image' not in request.files:
                return redirect(url_for('ocr_get'))

            image_file = request.files['image']

            # If the user does not select a file, the browser submits an empty file without a filename
            if image_file.filename == '':
                return redirect(url_for('ocr_get'))

            # Check if the file is an allowed image type
            if image_file and allowed_file(image_file.filename):
                # Load image to analyze into a 'bytes' object
                file_name="ocr_analysis."+ image_file.filename.split('.')[-1]
                image_path = os.path.join('static', file_name)
                print(image_path)
                image_file.save(image_path)
                with open(image_path, 'rb') as f:
                    image_data = f.read()

                # Extract text (OCR) from the image
                result = client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                # Prepare OCR results for rendering in the template
                ocr_results = []
                dimentions={}
                if result.read is not None:
                    dimentions=result.metadata
                    for line in result.read.blocks[0].lines:
                        ocr_results.append({
                            'line_text': line.text,
                            'line_bounding_box': line.bounding_polygon
                        })

                file_exist = True
                # Render the template with OCR results
                return render_template('ocr.html', ocr_results=ocr_results, file_exist=file_exist,image_url=image_path,dimentions=dimentions)

        except KeyError:
            # Handle missing environment variables
            return redirect(url_for('ocr_get'))



if __name__ == '__main__':
    app.run(debug=True)
