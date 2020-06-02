#Building a movie content based recommendation system
import pandas as pd
import  numpy as np

credits=pd.read_csv("tmdb_5000_credits.csv")
movies_df=pd.read_csv("tmdb_5000_movies.csv")
# print(credits.head())

credits_column_renamed=credits.rename(index=str, columns={"movie_id":"id"})
movies_df_merge=movies_df.merge(credits_column_renamed, on="id")
# print(movies_df_merge.head())

movies_cleaned_df=movies_df_merge.drop(columns=['homepage','title_x','title_y','status','production_countries'])
# print(movies_cleaned_df.head())
# print(movies_cleaned_df.info())

#content based system based on movie summary
# print(movies_cleaned_df.head(1)['overview'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words="english")

#filling NANs with empty string
movies_cleaned_df['overview']=movies_cleaned_df['overview'].fillna('')

#fitting the TF-IDF on the 'overview' text
tfv_matrix=tfv.fit_transform(movies_cleaned_df['overview'])
# print(tfv_matrix.shape)

from sklearn.metrics.pairwise import sigmoid_kernel
sig=sigmoid_kernel(tfv_matrix,tfv_matrix)
# print(sig[0])

#Reverse mapping of indices and movie titles
indices=pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
# print(indices)

def give_rec(title, sig=sig):
    #get the index corresponding to original_title
    idx=indices[title]
    # print(idx)

    #get the pairwise similarity scores
    sig_scores=list(enumerate(sig[idx]))

    #sort the movies
    sig_scores=sorted(sig_scores, key=lambda x:x[1], reverse=True)

    #scores of the 10 most similar movies
    sig_scores=sig_scores[1:11]

    #Movie indices
    movie_indices=[i[0] for i in sig_scores]

    #top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]


print(give_rec('Spy Kids'))