import numpy as np
import pandas as pd
import ast
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(lowercase=True,max_features=5000,stop_words='english')
movie=pd.read_csv('tmdb_5000_movies.csv')
#print(movie.head())
credit=pd.read_csv('tmdb_5000_credits.csv')
#print(credit.head())
movie=movie.merge(credit,on='title')
#print(movie.columns)
movies=movie[['movie_id','title','genres','keywords','overview','cast','crew']]
#print(movies.head())
movies.dropna(inplace=True)
#print(movies.isnull().sum())
def m_list():
    return movies['title']
def convert(obj):
  l=[]
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['overview']=movies['overview'].apply(lambda x: x.split())
def cast(obj):
  l=[]
  count=0
  for i in ast.literal_eval(obj):
    if count!=3:
      l.append(i['name'])
      count+=1
  return l
movies['cast']=movies['cast'].apply(cast)
def crew(obj):
  l=[]
  for i in ast.literal_eval(obj):
    if i['job']== 'Director':
      l.append(i['name'])
      break
  return l
movies['crew']=movies['crew'].apply(crew)
#print(movies.head())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)
vectors=cv.fit_transform(new_df['tags']).toarray()
similarity=cosine_similarity(vectors)
def recommend(movie):
     movie_index=new_df[new_df['title']==movie].index[0]
     movie_list=sorted(list(enumerate(similarity[movie_index])),reverse=True,key=lambda x:x[1])[1:6]
     l=[]
     for i in movie_list:
       l.append(new_df.iloc[i[0]].title)
     return l
