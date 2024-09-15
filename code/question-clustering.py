import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Step 1: Load the dataset
df = pd.read_csv('filepath to the questions')

# Convert to DataFrame for easier handling
#df = pd.DataFrame(questions, columns=["question"])

# Step 2: Preprocess the text (optional)
# You can add your text preprocessing steps here (e.g., lowercasing, stopword removal).
representation_model = KeyBERTInspired()
# Step 3: Create and fit the BERTopic model
topic_model = BERTopic(representation_model = representation_model)
topics, probs = topic_model.fit_transform(df['question'].tolist())

# Step 4: Get the topics and corresponding questions
df['topic'] = topics

# Step 5: Identify the most representative question for each cluster
# We will take the first question in each cluster as the representative one.
# Other methods like centroid or median embeddings can be used for more accurate representation.
representative_questions = df.groupby('topic')['question'].first().reset_index()

# Step 6: Output the representative questions for each cluster
for index, row in representative_questions.iterrows():
    print(f"Cluster {row['topic']}: {row['question']}")