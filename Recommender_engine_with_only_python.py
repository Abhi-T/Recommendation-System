import numpy as np
import pandas as pd

#get the data
column_names=['user_id', 'item_id', 'rating', 'timestamp']
df=pd.read_csv('Movie.data', sep='\t', names=column_names)
# print(df.head())

#movie titles
movie_titles=pd.read_csv("Movie_Id_Titles")
# print(movie_titles.head())

#merge
df=pd.merge(df, movie_titles, on='item_id')
# print(df.head())

#EDA
#visualize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

#lets create a ratings dataframe with average rating and number of ratings
# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
# print(ratings.head())

ratings['num of ratings']=pd.DataFrame(df.groupby('title')['rating'].count())
# ratings.head()

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
# plt.show()

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
# plt.show()

sns.jointplot(x='rating', y='num of ratings', data=ratings,alpha= 0.5)
# plt.show()

#Recommending similar movies
moviemat=df.pivot_table(index='user_id', columns='title', values='rating')
# print(moviemat.head())

#most rated movie
# print(ratings.sort_values('num of ratings', ascending=False).head(10))

#lets choose 2 movies, star wars and liar liar
starwars_user_ratings=moviemat['Star Wars (1977)']
liarliar_user_ratings=moviemat['Liar Liar (1997)']
# print(starwars_user_ratings.head())

#use corrwith() method to get the correlations between 2 pandas series:
similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar=moviemat.corrwith(liarliar_user_ratings)

#lets clean this by removing NaN values and using a dataframe instead of a series
corr_starwars=pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
# print(corr_starwars.head())

#sort the dataframe by correlation to get the most similar movies
# print(corr_starwars.sort_values('Correlation', ascending=False).head(10))

#filter out movies with less than 100 revies
corr_starwars=corr_starwars.join(ratings['num of ratings'])
# print(corr_starwars.head())

#sort the values
print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())

#now the same for liar liar
corr_liarliar=pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())
