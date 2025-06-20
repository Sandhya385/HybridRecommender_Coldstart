# HybridRecommender_Coldstart
HybridRecommender System

A full-featured hybrid movie recommender system that combines Collaborative Filtering (SVD-based) and Content-Based Filtering (TF-IDF on genres). It includes handling of cold start problems for new users and items, hyperparameter tuning with GridSearchCV, evaluation metrics, and an interactive Streamlit web app.

📁 Project Structure

├── data/
│   └── movies.csv
├── model/
│   ├── model_cf.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   └── cos_sim_matrix.pkl
├── recommender.py        # Main hybrid recommender script
├── app.py                # Streamlit app
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

**Features**

**Collaborative Filtering** using SVD from surprise library

**Content-Based Filtering** using TF-IDF vectorization of genres

**Hybrid Model** that blends both approaches with a tunable parameter alpha

**Cold Start solutions:**

Suggests most popular movies for new users

Suggests similar movies based on genre for new items

**Model Evaluation:**

RMSE for Collaborative Filtering

Approximate RMSE for Content-Based Filtering

**Hyperparameter Tuning** using GridSearchCV

**Deployment** using Streamlit


