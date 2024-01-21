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
    )[:5]
    return df.iloc[top_indices]["Food Product"]


# Example usage
user_preferences = "Chicken, None, None, Butter, Salt, None"
previous_choices = ["Chicken Noodle Soup", "Chicken Alfredo"]
recommendations = get_recommendations(
    user_preferences, previous_choices, cosine_sim, df
)

print("Recommended Food Items:")
print(recommendations)

# visualize cosine similarity matrix

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming cosine_sim is your cosine similarity matrix
# cosine_sim_df = pd.DataFrame(cosine_sim)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cosine_sim_df, cmap="coolwarm")
# plt.title("Cosine Similarity Matrix")
# plt.show()
