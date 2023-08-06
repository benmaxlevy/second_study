import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def split():
    df = pd.read_csv("all.csv")

    df["text"] = df["text"].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    df["text"] = df["text"].str.replace("[\\n]", "", regex=True)
    df["text"] = df["text"].str.replace("[\\r]", "", regex=True)

    df_male = df[df["text"].str.contains('[0-9][0-9](m|(( y\/o| year old)) (dude|boy|guy|man|male))', regex=True)]
    df_male.to_excel("male.xlsx")
    df_female = df[df["text"].str.contains('[0-9][0-9](f|(( y\/o| year old)) (woman|lady|girl|female))', regex=True)]
    df_female.to_excel("female.xlsx")

    print(len(df_male))
    print(len(df_female))
    print(df_male.head())
    print(df_female.head())

    return df_male, df_female


def lda(df_male, df_female):
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    tf_male = tf_vectorizer.fit_transform(df_male["text"])
    tf_feature_names_male = tf_vectorizer.get_feature_names_out()

    tf_female = tf_vectorizer.fit_transform(df_female["text"])
    tf_feature_names_female = tf_vectorizer.get_feature_names_out()

    no_topics = 10

    # Run LDA
    lda_male = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf_male)
    lda_female = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf_female)

    return lda_male, lda_female, tf_feature_names_male, tf_feature_names_female


def nmf():
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_male = tfidf_vectorizer.fit_transform(df_male["text"])
    tfidf_feature_names_male = tfidf_vectorizer.get_feature_names_out()

    tfidf_female = tfidf_vectorizer.fit_transform(df_female["text"])
    tfidf_feature_names_female = tfidf_vectorizer.get_feature_names_out()

    no_topics = 30

    # Run NMF
    nmf_male = NMF(n_components=no_topics, random_state=1, l1_ratio=.5, max_iter=1000000000).fit(tfidf_male)
    nmf_female = NMF(n_components=no_topics, random_state=1, l1_ratio=.5, max_iter=1000000000).fit(tfidf_female)

    """"
    doc_topic_male = nmf_male.transform(tfidf_male)

    print("===male===")
    male_topics = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for n in range(doc_topic_male.shape[0]):
        topic_most_pr = doc_topic_male[n].argmax()
        male_topics[int(topic_most_pr)] = male_topics[int(topic_most_pr)]+1

    print(male_topics)

    doc_topic_female = nmf_male.transform(tfidf_male)
    print("===female===")
    female_topics = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for n in range(doc_topic_male.shape[0]):
        topic_most_pr = doc_topic_female[n].argmax()
        female_topics[int(topic_most_pr)] = female_topics[int(topic_most_pr)] + 1
    print(female_topics)
    """

    return nmf_male, nmf_female, tfidf_feature_names_male, tfidf_feature_names_female


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(
            "Topic %d:" % (topic_idx))
        print(
            " ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == "__main__":
    df_male, df_female = split()
    lda_male, lda_female, tf_feature_names_male, tf_feature_names_female = lda(df_male, df_female)

    print("===male===")
    display_topics(lda_male, tf_feature_names_male, 15)

    print("===female===")
    display_topics(lda_female, tf_feature_names_female, 15)