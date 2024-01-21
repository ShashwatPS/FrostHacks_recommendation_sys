from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("food_ingredients_and_allergens.csv")

df["features"] = (
    df[["Main Ingredient", "Sweetener", "Fat/Oil", "Seasoning", "Allergens"]]
    .astype(str)
    .agg(", ".join, axis=1)
)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["features"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(user_preferences, previous_choices, cosine_sim_matrix, df):
    # Combine user preferences and previous choices
    user_input = user_preferences + ", " + ", ".join(previous_choices)
    user_profile = tfidf_vectorizer.transform([user_input])
    sim_scores = list(cosine_similarity(user_profile, tfidf_matrix).flatten())
    top_indices = sorted(
        range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True
    )[:10]
    return df.iloc[top_indices]["Food Product"].tolist()


# # Example usage
# user_preferences = "Chicken, None, None, Butter, Salt, None"
# previous_choices = ["Chicken Noodle Soup", "Chicken Alfredo"]
# recommendations = get_recommendations(
#     user_preferences, previous_choices, cosine_sim, df
# )

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("recommendation.log")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


class Item(BaseModel):
    user_preferences: str
    previous_choices: list


@app.post("/")
async def query(item: Item):
    recommendations = get_recommendations(
        item.user_preferences, item.previous_choices, cosine_sim, df
    )
    logger.info(
        f"Recommendations for {item.user_preferences} and {item.previous_choices}: {recommendations}"
    )
    return {"recommendations": recommendations}


# uvicorn app:app --host 0.0.0.0 --port 4000
