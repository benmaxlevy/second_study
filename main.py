import re, math

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearnex import patch_sklearn
patch_sklearn()


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
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))

    tf_male = tf_vectorizer.fit_transform(df_male["text"])
    tf_feature_names_male = tf_vectorizer.get_feature_names_out()

    tf_female = tf_vectorizer.fit_transform(df_female["text"])
    tf_feature_names_female = tf_vectorizer.get_feature_names_out()

    no_topics = 45

    # def get_umass_score(dt_matrix, i, j):
    #     zo_matrix = (dt_matrix > 0).astype(int)
    #     col_i, col_j = zo_matrix[:, i], zo_matrix[:, j]
    #     col_ij = col_i + col_j
    #     col_ij = (col_ij == 2).astype(int)
    #     Di, Dij = col_i.sum(), col_ij.sum()
    #     return math.log((Dij + 1) / Di)
    #
    # def get_topic_coherence(dt_matrix, topic, n_top_words):
    #     indexed_topic = zip(topic, range(0, len(topic)))
    #     topic_top = sorted(indexed_topic, key=lambda x: 1 - x[0])[0:n_top_words]
    #     coherence = 0
    #     for j_index in range(0, len(topic_top)):
    #         for i_index in range(0, j_index - 1):
    #             i = topic_top[i_index][1]
    #             j = topic_top[j_index][1]
    #             coherence += get_umass_score(dt_matrix, i, j)
    #     return coherence
    #
    # def get_average_topic_coherence(dt_matrix, topics, n_top_words):
    #     total_coherence = 0
    #     for i in range(0, len(topics)):
    #         total_coherence += get_topic_coherence(dt_matrix, topics[i], n_top_words)
    #     return total_coherence / len(topics)

    # for n_topics in range(140, 250, 5):
    #     lda_male = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=0, n_jobs=-1).fit(
    #         tf_male)
    #     avg_coherence = get_average_topic_coherence(tf_male, lda_male.components_, 15)
    #     print("===male===")
    #     print(str(n_topics) + " " + str(avg_coherence))
    #
    # for n_topics in range(5, 250, 5):
    #     lda_female = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=0, n_jobs=-1).fit(
    #         tf_female)
    #     avg_coherence = get_average_topic_coherence(tf_female, lda_male.components_, 15)
    #     print("===female===")
    #     print(str(n_topics) + " " + str(avg_coherence))

    lda_male = LatentDirichletAllocation(n_components=no_topics, max_iter=10, random_state=0, n_jobs=-1).fit(
        tf_male)

    lda_female = LatentDirichletAllocation(n_components=no_topics, max_iter=10, random_state=0, n_jobs=-1).fit(
             tf_female)

    return lda_male, lda_female, tf_feature_names_male, tf_feature_names_female


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(
            "Topic %d:" % (topic_idx))
        print(
            " ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == "__main__":
    # df_male, df_female = split()
    df_male = pd.read_excel("results/male.xlsx")
    df_female = pd.read_excel("results/female.xlsx")

    lda_male, lda_female, tf_feature_names_male, tf_feature_names_female = lda(df_male, df_female)

    print("===male===")
    display_topics(lda_male, tf_feature_names_male, 50)

    print("===female===")
    display_topics(lda_female, tf_feature_names_female, 50)