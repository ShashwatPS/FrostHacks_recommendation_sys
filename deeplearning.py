import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("food_ingredients_and_allergens.csv")

# Encode categorical features
label_encoder = LabelEncoder()
df["Main Ingredient"] = label_encoder.fit_transform(df["Main Ingredient"])
df["Sweetener"] = label_encoder.fit_transform(df["Sweetener"])
df["Fat/Oil"] = label_encoder.fit_transform(df["Fat/Oil"])
df["Seasoning"] = label_encoder.fit_transform(df["Seasoning"])
df["Allergens"] = label_encoder.fit_transform(df["Allergens"])

# Create a binary target variable: 1 if the user likes the food, 0 otherwise
df["Liked"] = (df["Prediction"] == "Contains").astype(int)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Define the neural network model
def create_recommendation_model(num_users, num_items, embedding_dim=50):
    user_input = Input(shape=(1,), name="user_input")
    item_input = Input(shape=(1,), name="item_input")

    user_embedding = Embedding(
        output_dim=embedding_dim, input_dim=num_users + 1, input_length=1
    )(user_input)
    item_embedding = Embedding(
        output_dim=embedding_dim, input_dim=num_items + 1, input_length=1
    )(item_input)

    user_flat = Flatten()(user_embedding)
    item_flat = Flatten()(item_embedding)

    concatenated = Concatenate()([user_flat, item_flat])
    dense1 = Dense(128, activation="relu")(concatenated)
    output = Dense(1, activation="sigmoid")(dense1)

    model = Model(inputs=[user_input, item_input], outputs=output)

    return model


# Get the number of unique users and items
num_users = df["Main Ingredient"].nunique()
num_items = df["Allergens"].nunique()

# Create the model
model = create_recommendation_model(num_users, num_items)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    [train_df["Main Ingredient"], train_df["Allergens"]],
    train_df["Liked"],
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(
    [test_df["Main Ingredient"], test_df["Allergens"]], test_df["Liked"]
)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
