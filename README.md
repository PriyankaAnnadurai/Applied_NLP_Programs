# Program1

### Apply Naive Bayes Algorithm to perform sentiment analysis by using a customized data set.

```py
# Import necessary libraries
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download NLTK data (stopwords)
nltk.download('stopwords')

# Sample dataset of sentences with their sentiments (1 = positive, 0 = negative)
data = [
    ("I love this product, it works great!", 1),
    ("This is the best purchase I have ever made.", 1),
    ("Absolutely fantastic service and amazing quality!", 1),
    ("I am very happy with my order, will buy again.", 1),
    ("This is a horrible experience.", 0),
    ("I hate this so much, it broke on the first day.", 0),
    ("Worst product I have ever used, total waste of money.", 0),
    ("I am disappointed with this product, it didn't work as expected.", 0)
]

# Separate sentences and labels
sentences = [pair[0] for pair in data]
labels = np.array([pair[1] for pair in data])

# Split dataset into training and testing sets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# Text Preprocessing
# Tokenization, removing stopwords and converting text into numerical data using CountVectorizer

# Instead of using a set, use a list for stop_words
stop_words = stopwords.words('english') # Changed this line to create a list

# Initialize CountVectorizer (this will convert text into a bag-of-words representation)
vectorizer = CountVectorizer(stop_words=stop_words)

# Fit the vectorizer on the training data and transform both training and test sets
X_train = vectorizer.fit_transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

# Initialize the Naive Bayes Classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict sentiments for the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Test the model with new sentences
test_sentences = ["I am happy to comment!", "This is a terrible product."]
test_X = vectorizer.transform(test_sentences)

# Predict sentiments for new sentences
predictions = nb_classifier.predict(test_X)

# Output predictions
for sentence, sentiment in zip(test_sentences, predictions):
    print(f"Sentence: '{sentence}' => Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
```

# Program2 

### Create a basic machine translation system using the K-Nearest Neighbors (KNN) algorithm. Perform the following task in python:
### Build a small bilingual dictionary with sentence pairs in English and French.
### Implement a KNN-based approach to translate new English sentences into French by finding the closest matches in the dictionary.
### Use cosine similarity on vectorized representations of sentences to find the nearest neighbor(s).


```py
!pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Sample bilingual dictionary
english_sentences = [
    "hello", "how are you", "good morning", "good night", "thank you",
    "see you later", "what is your name", "my name is John", "where is the library",
    "I like to read books"
]

french_sentences = [
    "bonjour", "comment ça va", "bonjour", "bonne nuit", "merci",
    "à plus tard", "quel est ton nom", "mon nom est John", "où est la bibliothèque",
    "j'aime lire des livres"
]
vectorizer = TfidfVectorizer()
english_vectors = vectorizer.fit_transform(english_sentences)
def knn_translate(input_sentence, k=1):
    input_vector = vectorizer.transform([input_sentence])

    # Compute cosine similarity between the input sentence and all sentences in the dictionary
    similarities = cosine_similarity(input_vector, english_vectors).flatten()

    # Get indices of the top-k similar sentences
    top_k_indices = similarities.argsort()[-k:][::-1]

    # Retrieve and display the French translations for the most similar sentences
    translations = [french_sentences[i] for i in top_k_indices]
    return translations
# Test sentences
test_sentences = ["good evening", "where is the library", "thank you very much"]

# Translate each test sentence
for sentence in test_sentences:
    translations = knn_translate(sentence, k=1)  # Use k=1 for the closest translation
    print(f"English: {sentence} -> French: {translations[0]}")
```

# Program3 

### Build a Siamese network to perform Sentence similarity detection. Your task is to:
### Design and implement a Siamese network using LSTM layers to compare pairs of sentences.
### Use cosine similarity as a metric to evaluate how two sentences are similar.
### Train the model on a sample dataset of pairs of similar and dissimilar sentences , and test it by providing similarity scores for at least two sentence pairs.


```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Sentence pairs (labels: 1 for similar, 0 for dissimilar)
sentence_pairs = [
    ("How are you?", "How do you do?", 1),
    ("How are you?", "What is your name?", 0),
    ("What time is it?", "Can you tell me the time?", 1),
    ("What is your name?", "Tell me the time?", 0),
    ("Hello there!", "Hi!", 1),
]

# Separate into two sets of sentences and their labels
sentences1 = [pair[0] for pair in sentence_pairs]
sentences2 = [pair[1] for pair in sentence_pairs]
labels = np.array([pair[2] for pair in sentence_pairs])

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
max_len = 100  # Max sequence length
X1 = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=max_len)

# Input layers for two sentences
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

# Embedding layer
embedding_dim = 1000
embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)

# Shared LSTM layer
shared_lstm = Bidirectional(LSTM(512))

# Process the two inputs using the shared LSTM
encoded_1 = shared_lstm(embedding(input_1))
encoded_2 = shared_lstm(embedding(input_2))

# Calculate the L1 distance between the two encoded sentences
def l1_distance(vectors):
    x, y = vectors
    return K.abs(x - y)

l1_layer = Lambda(l1_distance)
l1_distance_output = l1_layer([encoded_1, encoded_2])

# Add a dense layer for classification (similar/dissimilar)
output = Dense(1, activation='sigmoid')(l1_distance_output)

# Create the Siamese network model
siamese_network = Model([input_1, input_2], output)
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
siamese_network.summary()

# Train the model
siamese_network.fit([X1, X2], labels, epochs=12, batch_size=2)

# Test with a new sentence pair
test_sentences1 = ["How are you?"]
test_sentences2 = ["How do you do?"]

test_X1 = pad_sequences(tokenizer.texts_to_sequences(test_sentences1), maxlen=max_len)
test_X2 = pad_sequences(tokenizer.texts_to_sequences(test_sentences2), maxlen=max_len)

# Predict similarity
similarity = siamese_network.predict([test_X1, test_X2])
print(f"Similarity Score: {similarity[0][0]}")
```

# Program 4 

### Using a pre-trained transformer model , create a simple language translation application that translates English sentences into French. Your solution should:
### Load the pre-trained model and tokenizer.
### Include a function to translate user-provided sentences from English to French.
### Display at least three translated sentences as examples.


```py
!pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer for English-to-French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

import torch

def translate_text(text: str, max_length: int = 40) -> str:
    # Tokenize the input text and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate translation using the model
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated IDs back to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
# Sample sentences in English
english_sentences = [
    "Hello, how are you?",
    "This is an experiment in machine translation.",
    "Transformers are powerful models for natural language processing tasks.",
    "Can you help me with my homework?",
    "I love learning new languages."
]

# Translate each sentence
for sentence in english_sentences:
    translation = translate_text(sentence)
    print(f"Original: {sentence}")
    print(f"Translated: {translation}\n")
```

# Program - 5

### Perform  the preprocessing steps applied to the text corpus and train a Word2Vec model.After training the Word2Vec model, find and report the embedding for any  word. Identify the two words that are most similar to that word based on the trained model, along with their similarity scores.

```py
# Import necessary libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample corpus
corpus = [
    "Natural language processing is a field of artificial intelligence.",
    "It enables computers to understand human language.",
    "Word embedding is a representation of words in a dense vector space.",
    "Gensim is a library for training word embeddings in Python.",
    "Machine learning and deep learning techniques are widely used in NLP."
]

# Preprocess the text: Tokenize, remove punctuation and stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# Apply preprocessing to the corpus
processed_corpus = [preprocess_text(sentence) for sentence in corpus]

# Train a Word2Vec model
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=2, min_count=1, sg=1)  # sg=1 uses Skip-gram

# Save the model for future use
model.save("word2vec_model.model")

# Test the model by finding the embedding of a word
word = "vector"
if word in model.wv:
    print(f"Embedding for '{word}':\n{model.wv[word]}")
else:
    print(f"'{word}' not found in vocabulary.")

# Find similar words
similar_words = model.wv.most_similar(word, topn=2)
print(f"Words similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")
```

# Program 6 

###  Implement Auto Correction of words by calculating edit distance and test for 4 sample words


```py
# Install NLTK if needed
!pip install nltk

import nltk
nltk.download('words')
from nltk.corpus import words
import re
from collections import Counter

# Use the NLTK words corpus as our vocabulary
word_list = words.words()
word_freq = Counter(word_list)  # Count frequencies, though here it's a simple corpus with each word appearing once

# Define a set of all known words
WORD_SET = set(word_list)

# Define a function to calculate minimum edit distance
def edit_distance(word1, word2):
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                dp[i][j] = j  # Cost of insertions
            elif j == 0:
                dp[i][j] = i  # Cost of deletions
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change cost
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])  # Substitution
    return dp[-1][-1]

# Define a function to calculate word probability
def word_probability(word, N=sum(word_freq.values())):
    return word_freq[word] / N if word in word_freq else 0

# Suggest corrections based on edit distance and probability
def autocorrect(word):
    # If the word is correct, return it as is
    if word in WORD_SET:
        return word

    # Find candidate words within an edit distance of 1 or 2
    candidates = [w for w in WORD_SET if edit_distance(word, w) <= 2]

    # Choose the candidate with the highest probability
    corrected_word = max(candidates, key=word_probability, default=word)

    return corrected_word

# Test the function with common misspellings
test_words = ["speling", "korrect", "exampl", "wrld"]

for word in test_words:
    print(f"Original: {word} -> Suggested: {autocorrect(word)}")
```
