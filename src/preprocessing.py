from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
from sklearn.preprocessing import LabelEncoder

import config as c
import utils as u


def preprocess(df):
    def token(text):
        st = LancasterStemmer()
        txt = nltk.word_tokenize(text.lower())
        return [st.stem(word) for word in txt]

    top_speakers = df.groupby([c.TARGET]).size(
    ).loc[df.groupby([c.TARGET]).size() > 2000]

    main_char_lines = df.loc[df[c.TARGET].isin(
        top_speakers.index.values)]

    main_char_lines['Line'] = [line.replace(
        '\n', '') for line in main_char_lines['Line']]

    # stop = set(stopwords.words("english"))
    cv = CountVectorizer(  # lowercase=True,
        tokenizer=token,  # stop_words=stop, # token_pattern=u'(?u)\b\w\w+\b',
        analyzer=u'word', min_df=4)

    X = cv.fit_transform(main_char_lines['Line'].tolist()).toarray()

    le = LabelEncoder()
    y = le.fit_transform(main_char_lines[c.TARGET])

    u.dump(cv, c.PATH_VECTORIZER)
    u.dump(le, c.PATH_ENCODER)

    return X, y
