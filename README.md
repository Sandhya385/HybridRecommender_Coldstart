# Hybrid Recommender System (Cold Start Handling)

A **full-featured hybrid movie recommender system** that combines:  
- **Collaborative Filtering (SVD-based)**  
- **Content-Based Filtering (TF-IDF on genres)**  

The system also addresses the **cold start problem** for new users and new items. It includes **hyperparameter tuning, evaluation metrics, and an interactive Streamlit web app** for deployment.  

---

## 📂 Project Structure  

```bash
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
```

---

## ✨ Features  

- **Collaborative Filtering**  
  - Uses SVD from the `surprise` library  

- **Content-Based Filtering**  
  - TF-IDF vectorization of movie genres  

- **Hybrid Model**  
  - Blends both approaches with a tunable parameter `alpha`  

- **Cold Start Solutions**  
  - Recommends most popular movies for new users  
  - Suggests similar movies based on genres for new items  

- **Model Evaluation**  
  - RMSE for Collaborative Filtering  
  - Approximate RMSE for Content-Based Filtering  

- **Hyperparameter Tuning**  
  - GridSearchCV for SVD parameters  

- **Deployment**  
  - Streamlit app for interactive recommendations  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Sandhya385/HybridRecommender_Coldstart.git
   cd HybridRecommender_Coldstart
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## 📊 Results  

- Collaborative Filtering RMSE: *<add value>*  
- Content-Based RMSE (approximate): *<add value>*  

*(Optional: add a screenshot of your Streamlit app or sample recommendations here)*  

---

## 🔮 Future Improvements  

- Add support for user ratings in cold start scenarios  
- Integrate more metadata (actors, directors, tags) into content-based filtering  
- Experiment with neural network-based recommenders  


