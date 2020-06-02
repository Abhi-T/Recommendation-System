import pandas as pd
import numpy as np

#downloaded from kaggle
credits=pd.read_csv("tmdb_5000_credits.csv")
movies_df=pd.read_csv("tmdb_5000_movies.csv")
# print(credits.shape)
# print(movies_df.shape)

#renaming the movie_id column with id
credits_column_renamed=credits.rename(index=str, columns={"movie_id":"id"})
movies_df_merge=movies_df.merge(credits_column_renamed, on='id')
# print(movies_df_merge.head())

#dropping columns which are not important
movies_cleaned_df=movies_df_merge.drop(columns=['homepage','title_x','title_y','status','production_countries'])
# print(movies_cleaned_df.head())

#we are using weighted average for each movie's avergage rating
# W=(R*V + C*m)/(v+m)
#W=weigted rating,
# R=avg or mean of the rating between 0 to 10, Rating
#v=number of votes for the movie, votes
# m=minimum votes for threshold, like atleast 200 people should have voted for that movie, (we are setting 3000)
# C=the mean vote across the whole dataset (currently 6.9)

v=movies_cleaned_df['vote_count']
R=movies_cleaned_df['vote_average']
C=movies_cleaned_df['vote_average'].mean()
m=movies_cleaned_df['vote_count'].quantile(0.70) # minimum 70% as compared to others

movies_cleaned_df['weighted_average']=((R*v)+(C*m))/(v+m)
# print(movies_cleaned_df.head())

movie_sorted_ranking=movies_cleaned_df.sort_values("weighted_average",ascending=False)
# print(movie_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head())

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
weight_average=movie_sorted_ranking.sort_values("weighted_average",ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10))
plt.xlim(4,10)
plt.title('Best Movies by avg votes', weight='bold')
plt.xlabel('Weighted avg score', weight='bold')
plt.ylabel('Movie Title ', weight='bold')
# plt.show()
# plt.savefig('best_movies.png')

#now we will use popularity
popularity=movie_sorted_ranking.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,6))
ax=sns.barplot(x=popularity['popularity'].head(10), y=popularity['original_title'].head(10), data=popularity)
plt.title('Most Popular by votes', weight='bold')
plt.xlabel('Score of Popularity', weight='bold')
plt.xlabel('Movie Title', weight='bold')
# plt.show()
# plt.savefig('best_popular_movies.png')

#Recommendation based on scaled weighted average and popularity score(priority is given 50% to both)
from sklearn.preprocessing import MinMaxScaler

scaling=MinMaxScaler()
movie_scaled_df=scaling.fit_transform(movies_cleaned_df[['weighted_average', 'popularity']])
movie_normalized_df=pd.DataFrame(movie_scaled_df, columns=['weighted_average', 'popularity'])
# print(movie_normalized_df.head())

movies_cleaned_df[['normalized_weight_average', 'normalized_popularity']]=movie_normalized_df
# print(movies_cleaned_df.head())

movies_cleaned_df['score']=movies_cleaned_df['normalized_weight_average']*0.5 + movies_cleaned_df['normalized_popularity']*0.5
movies_scored_df=movies_cleaned_df.sort_values(['score'], ascending=False)
# print(movies_scored_df[['original_title', 'normalized_weight_average', 'normalized_popularity', 'score']].head())

scored_df=movies_cleaned_df.sort_values('score', ascending=False)
plt.figure(figsize=(16,6))
ax=sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df )
plt.title('Best Rated and Most popular movie', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title ', weight='bold')
plt.show()
plt.savefig('scored_movies.png')
