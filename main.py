import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def main():
    df = pd.read_csv("suicidewatch.csv")
    df = df[df["text"] != "[removed]"]
    df = df[df["text"] != "deleted post"]
    df["text"] = df["text"].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

    df_male = df[df["text"].str.contains('[0-9][0-9]m', regex=True)]
    df_female = df[df["text"].str.contains('[0-9][0-9]f', regex=True)]
    print(len(df_male))
    print(len(df_female))
    print(df_male.head())
    print(df_female.head())

    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_male = tfidf_vectorizer.fit_transform(df_male["text"])
    tfidf_feature_names_male = tfidf_vectorizer.get_feature_names_out()

    # NMF is able to use tf-idf
    tfidf_female = tfidf_vectorizer.fit_transform(df_female["text"])
    tfidf_feature_names_female = tfidf_vectorizer.get_feature_names_out()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(df["text"])
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    no_topics = 30

    # Run NMF
    nmf_male = NMF(n_components=no_topics, random_state=1, l1_ratio=.5, max_iter=1000000000).fit(tfidf_male)
    nmf_female = NMF(n_components=no_topics, random_state=1, l1_ratio=.5, max_iter=1000000000).fit(tfidf_female)

    # Run LDA
    #lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(
            "Topic %d:" % (topic_idx))
            print(
            " ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))

    no_top_words = 15
    print("===male===")
    display_topics(nmf_male, tfidf_feature_names_male, no_top_words)

    print("===female===")
    display_topics(nmf_female, tfidf_feature_names_female, no_top_words)
    #display_topics(lda, tf_feature_names, no_top_words)

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
if __name__ == "__main__":
    main()
