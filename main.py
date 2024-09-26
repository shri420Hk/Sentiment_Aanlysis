import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline
import textblob
import matplotlib.pyplot as plt
import re

#Load the DialoGPT medium model from Hugging Face
generator = pipeline('text-generation', model='microsoft/DialoGPT-medium')

def clean_response(text):
    text = re.sub(r'([!?.])\1+', r'\1', text)
    sentences = re.split(r'(?<=[.!?]) +', text) 
    return sentences[0].strip() 

def get_huggingface_response(user_input):
    prompt = f"User: {user_input}\nAI:"
    try:
        response = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)
        cleaned_response = clean_response(response[0]['generated_text'])
        return cleaned_response
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request."

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = textblob.TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


def generate_sentiment_report(sentiments):
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [sentiments.count('Positive'), sentiments.count('Negative'), sentiments.count('Neutral')]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Analysis Report')
    plt.show()
    
# Main Function
def main():
    print("Welcome to the AI-powered sentiment analysis tool.")
    conversations = []
    sentiments = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_huggingface_response(user_input)
        print("AI: " + response)
        sentiment = analyze_sentiment(response)
        print(f"Sentiment: {sentiment}")
        conversations.append(response)
        sentiments.append(sentiment)
    generate_sentiment_report(sentiments)

if __name__ == "__main__":
    main()
