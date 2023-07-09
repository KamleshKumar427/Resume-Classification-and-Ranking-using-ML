from joblib import load
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import pickle

# Load the model from the file
nb = load('naive_bayes.joblib')


word_vectorizer = load('word_vectorizer.joblib')
label=load('label.joblib')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespaces
    return resumeText


directory = 'INFORMATION-TECHNOLOGY'

categories_dict = {}

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):  # Make sure the file is a PDF
        # Open the PDF file
        with open(os.path.join(directory, filename), 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            text = ''
            # Loop over all pages in the PDF and extract text
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Clean the text
            cleaned_text = cleanResume(text)

            # Transform the cleaned text using the vectorizer
            WordFeatures = word_vectorizer.transform([cleaned_text])

            # Predict the category of the resume
            prediction = nb.predict(WordFeatures)

            # print(f'The predicted category for {filename} is: {prediction[0]}')

            cat = label.inverse_transform(prediction)

            if cat[0] in categories_dict:
                categories_dict[cat[0]].append(filename)
            else:
                categories_dict[cat[0]] = [filename]

with open('categories_dict.pkl', 'wb') as file:
    pickle.dump(categories_dict, file)

