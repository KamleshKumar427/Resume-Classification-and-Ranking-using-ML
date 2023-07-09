import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from joblib import dump, load
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('UpdatedResumeDataSet.csv')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

df['cleaned'] = df['Resume'].apply(lambda x:cleanResume(x))

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from wordcloud import WordCloud

# Now encode the data
label = LabelEncoder()
df['new_Category'] = label.fit_transform(df['Category'])

# Vectorizing the cleaned columns
text = df['cleaned'].values
target = df['new_Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(text)

dump(word_vectorizer, 'word_vectorizer.joblib')
dump (label,'label.joblib' )

WordFeatures = word_vectorizer.transform(text)

# print(WordFeatures[1])

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, target, random_state=24, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

y_pred_class = nb.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred_class)

print(f'Accuracy: {accuracy}')

# print(metrics.classification_report(y_test, y_test))
print(metrics.classification_report(y_test, y_pred_class))

print(f'---------------------------------\n| Training Accuracy   :- {(nb.score(X_train, y_train)*100).round(2)}% |')
print(f'--------------------print(metrics.classification_report(y_test, y_test))-------------\n| Validation Accuracy :- {(nb.score(X_test, y_test)*100).round(2)}% |\n---------------------------------')

# Save the model to a file
dump(nb, 'naive_bayes.joblib')