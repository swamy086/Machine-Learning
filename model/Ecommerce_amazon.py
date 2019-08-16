import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
df_train = pd.read_csv(r'C:\Users\kajah\Desktop\2901c100-b-Dataset\Dataset\train.csv')
df_test =  pd.read_csv(r'C:\Users\kajah\Desktop\2901c100-b-Dataset\Dataset\test.csv')

def clean_text(data_frame):
    corpus = []
    for x in range(0,len(data_frame)):
        text = data_frame['Review Text'][x] +' '+ data_frame['Review Title'][x]
        text = text.lower()
        text = re.sub('[0-9",.?#\'!<>;`~$%*)(:’“?\/[\]]',' ',text)
        text = text.split()
        text = [word for word in text if len(word) != 1]
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        corpus.append(text)
    return corpus
        
train_corpus = clean_text(df_train)

cv = CountVectorizer(max_features = 1500)
X_train = cv.fit_transform(train_corpus).toarray()
Y_train = df_train.iloc[:,2]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

test_corpus = clean_text(df_test)
X_test = cv.fit_transform(test_corpus).toarray()
Y_pred = classifier.predict(X_test)

from pandas import DataFrame
pred = DataFrame(Y_pred)
pred.to_csv(r'C:\Users\kajah\Desktop\result.csv')


#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train,Y_train)
#
#Y_pred = classifier.predict(X_test)
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, Y_train)
## Predicting the test set results
#Y_pred = classifier.predict(X_test)
 
