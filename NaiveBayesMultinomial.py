#NAIVE BAYES - fake news detection
!pip install wordcloud

import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

# Sample data and CSV
df = pd.DataFrame({
  'text': ['Cure for cancer found', 'Aliens landed in NYC', 'Govt launches new policy',
           'Click here to win money', 'NASA confirms water', 'You won a lottery',
           'COVID vaccine works', 'Secret to weight loss'],
  'label': [1, 0, 1, 0, 1, 0, 1, 0]
})
df.to_csv("fake_news.csv", index=False)

# Load, vectorize, split
df = pd.read_csv("fake_news.csv")
X = TfidfVectorizer().fit_transform(df['text'])
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.3)
model = MultinomialNB().fit(X_train, y_train)

# Predict and score
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 2))

# Word clouds
for label, title in zip([1, 0], ["Real", "Fake"]):
    text = ' '.join(df[df['label'] == label]['text'])
    plt.imshow(WordCloud().generate(text))
    plt.axis("off")
    plt.title(f"{title} News")
    plt.show()
