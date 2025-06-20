#!/usr/bin/env python
# coding: utf-8

# In[5]:
import pandas as pd
import numpy as np
from surprise import SVD,Dataset,Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import streamlit as st


# In[ ]:

#Read the data
ratings=pd.read_csv("D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart\ml-100k/u.data",sep='\t',names=['user_id','item_id','rating','timestamp'])
ratings.drop('timestamp',axis=1,inplace=True)

movies=pd.read_csv("D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart\ml-100k/u.item",sep='|',encoding='latin-1',header=None,
                  names=["item_id", "title", "release_date", "video_release", "IMDb", "unknown",
                            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                            "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                            "Thriller", "War", "Western"])
movies=movies.drop(movies[['release_date','video_release','IMDb','unknown']],axis=1)

data=pd.merge(ratings,movies,on="item_id")

#Colaborative filtering
reader=Reader(rating_scale=(1,5))
dataset=Dataset.load_from_df(data[['user_id','item_id','rating']],reader)
trainset,testset=train_test_split(dataset,test_size=0.2,random_state=42)

# Grid Search to find best SVD params
param_grid = {
    'n_factors': [50, 100],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(dataset)

model_cf = gs.best_estimator['rmse']
model_cf.fit(trainset)
prediction_cf=model_cf.test(testset)
print(f"CF RMSE:",rmse(prediction_cf))

#Save the CF model
joblib.dump(model_cf,"D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart/model_cf.pkl")

#Content based filtering using tfidf
#create genre string for each movie
genre_cols = movies.columns.difference(['item_id', 'title'])

movies['genres'] = movies[genre_cols].apply(lambda x: ' '.join(x.index[x == 1]), axis=1)

#Tfidf Vectorization
tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix=tfidf.fit_transform(movies['genres'])
cos_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)

#save content based model components
joblib.dump(tfidf,"D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart/tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix,"D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart/tfidf_matrix.pkl")
joblib.dump(cos_sim,"D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart/cos_sim.pkl")
movies.to_csv("D:\DataScience\ProjectsPractice\HybridRecommender_Coldstart/movies.csv",index=False)

#Hybrid Recommender(CF,CBF combined)
def hybrid_recommender(user_id,title,top_n=5,alpha=0.5):
    #get index and item Id
    idx=movies[movies['title']==title].index[0]
    item_id=movies.iloc[idx]['item_id']
    
    #CF Score: Predicted rating
    try:
        cf_score=model_cf.predict(user_id,item_id).est
    except:
        cf_score=2.5 #nuetral fallback
    
    #Content based scores for similar movies
    sim_scores=list(enumerate(cos_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:top_n+1]
    
    recommendations=[]
    for i,sim in sim_scores:
        sim_item_id=movies.iloc[i]['item_id']
        try:
            cf_predict=model_cf.predict(user_id,sim_item_id).est
        except:
            cf_score=2.5
        hybrid_score=alpha*sim+(1-alpha)*(cf_predict/5) #Scale CF to [0,1]
        recommendations.append((movies.iloc[i]['title'],hybrid_score))
    return sorted(recommendations,key=lambda x:x[1],reverse=True)

#Handle cold start recommended
def cold_start_recommend(top_n=5):
    #Popular movies as fallback
    popular=ratings.groupby('item_id').size().sort_values(ascending=False).head(top_n)
    return movies[movies['item_id'].isin(popular.index)][['title']]

#Evaluation
def evaluate_cbf():
    user_avg=ratings.groupby('item_id')['rating'].mean()
    predictions=[]
    truths=[]
    for i in range(100):
        try:
            sim_scores=list(enumerate(cos_sim[i]))
            sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:6]
            avg_rating=np.mean([user_avg.get(movies.iloc[j]['item_id'],2.5) for j,_ in sim_scores])
            predictions.append(avg_rating)
            truths.append(user_avg.get(movies.iloc[i]['item_id'],2.5))
        except:
            continue
    print("CBF Approx RMSE:",np.sqrt(np.mean((np.array(predictions)-np.array(truths))**2)))
    
def recommend_new_item_to_users(new_item_genres, top_n=5):
    """
    Given the genre string of a new item, recommend similar existing items.

    Parameters:
    - new_item_genres: str — genre string of new movie (e.g., 'Action|Adventure|Sci-Fi')
    - top_n: int — number of similar items to return

    Returns:
    - DataFrame of top-N similar existing movie titles
    """
    # Transform genre string to match training data format
    new_item_vector = tfidf.transform([new_item_genres])
    
    # Compute similarity with existing items
    sim_scores = cosine_similarity(new_item_vector, tfidf_matrix).flatten()
    
    # Get indices of top-N similar movies
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    return movies.iloc[top_indices][['title']]

def main():
    st.title("Hybrid Movie Recommender")
    user_id=st.number_input("Enter the user id",min_value=1,value=1)
    movie_list=movies['title'].to_list()
    selected_movie=st.selectbox("Select a movie from list",movie_list)
    
    option=st.radio("Choose Recommendation type",['Hybrid','Collaborative Filtering','Content based Filtering'])
    
    if st.button('Get Recommendations'):
        if option=="Hybrid":
            recs=hybrid_recommender(user_id,selected_movie)
            
        elif option=="Collaborative Filtering":
            recs=[]
            for i in range(len(movies)):
                item_id=movies.iloc[i]['item_id']
                pred=model_cf.predict(user_id,item_id)
                title=movies.iloc[i]['title']
                recs.append((title,pred.est))
            recs=sorted(recs, key=lambda x:x[1],reverse=True)[:5]
        elif option=="Content based Filtering":
            #Get similarity scores for the selected movie
            idx=movies[movies['title']==selected_movie].index[0]
            sim_scores=list(enumerate(cos_sim[idx]))
            sim_scores=sorted(sim_scores, key=lambda x:x[1],reverse=True)[1:6]
            recs=[(movies.iloc[i]['title'],score ) for i,score in sim_scores] #Top 5 similar movies
            
        #Display results
        st.subheader("Recommended movies:")
        for title,score in recs:
            st.write(f"{title},score:{score:.2f}")
            
    st.markdown("------")
    st.subheader("New User? Get Popular movies")
    if st.button("Show Popular Movies"):
        st.dataframe(cold_start_recommend())
        
    st.markdown("------")
    st.subheader("New Movie? Enter the Genre info")
    genre_input=st.text_input("Enter Genres eg: Action|Adventure|Scifi")
    if st.button("Recommend Movies based on Genre"):
        st.dataframe(recommend_new_item_to_users(genre_input))
        
if __name__=='__main__':
    main()
            

