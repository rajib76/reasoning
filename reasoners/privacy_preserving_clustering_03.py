# !/usr/bin/env python

## Algorithm for Clustering and Summarizing Text Data with PII Removal

### Step 1: Load Conversations
# - Collect a dataset of text-based user conversations or preferences.

### Step 2: Remove Personally Identifiable Information (PII)
# - Use a language model to sanitize text data by removing PII (e.g., names, emails, phone numbers).

### Step 3: Extract Text Features
# - Detect language.
# - Count words.

### Step 4: Compute Embeddings
# - Convert text into vector representations using a language model.

### Step 5: Cluster Texts
# - Apply K-means clustering to group similar texts based on embeddings.

### Step 6: Generate Cluster Summaries
# - Summarize grouped texts using a language model.
# - Assign a topic name to each cluster.

### Step 7: Visualize Clusters
# - Reduce dimensions with UMAP and plot clusters.
# - Label clusters with topic names.

### Step 8: Execute Pipeline
# - Apply all steps sequentially and display results.

import os

import matplotlib.pyplot as plt
import openai
import umap.umap_ as umap
from dotenv import load_dotenv
from langdetect import detect
from sklearn.cluster import KMeans

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def load_conversations():
    return [
        "My name is John Doe, and I love sci-fi movies.",
        "I'm looking for recommendations on fantasy novels with rich world-building.",
        "My phone number is 123-456-7890, and I prefer action-packed video games.",
        "I enjoy cooking and experimenting with different cuisines, particularly Italian and Indian.",
        "My email is jane.doe@example.com, and I usually listen to jazz music while working.",
        "I'm a fan of horror movies, but I also like psychological thrillers.",
        "I want to start a fitness routine focused on strength training and endurance.",
        "I like attending live concerts and music festivals in my free time.",
        "I enjoy hiking and spending time outdoors, especially in the mountains.",
        "I prefer minimalist interior design with neutral colors and simple decor."
    ]


def remove_pii_with_model(text):
    prompt = f"Remove any personally identifiable information (PII) such as names, phone numbers, and emails from the following text:\n\n{text}\n\nCleaned Text:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_facet(conversation):
    conversation = remove_pii_with_model(conversation)
    try:
        language = detect(conversation)
    except Exception:
        language = "unknown"
    return {
        "text": conversation,
        "language": language,
        "word_count": len(conversation.split())
    }


def compute_embeddings(facets):
    texts = [facet["text"] for facet in facets]
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings


def cluster_texts(embeddings, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def summarize_text(texts):
    combined_text = " ".join(texts)
    prompt = f"Summarize the following text in a concise paragraph:\n\n{combined_text}\n\nSummary:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


def generate_topic_name(texts):
    prompt = f"Generate a short topic name for the following text cluster:\n\n{texts}\n\nTopic:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def visualize_clusters(embeddings, labels, cluster_topics):
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="viridis", s=100, alpha=0.7,
                          edgecolors='k')

    for i, (x, y) in enumerate(embedding_2d):
        plt.annotate(cluster_topics[labels[i]], (x, y), fontsize=9, alpha=0.75, textcoords="offset points",
                     xytext=(5, 5), ha='right')

    plt.title("2D Visualization of Conversation Clusters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()


def main():
    conversations = load_conversations()
    facets = [extract_facet(conv) for conv in conversations]
    embeddings = compute_embeddings(facets)
    num_clusters = 3
    labels, _ = cluster_texts(embeddings, num_clusters=num_clusters)

    cluster_text_map = {}
    cluster_topics = {}
    for i in range(num_clusters):
        cluster_text_map[i] = [facets[j]["text"] for j in range(len(facets)) if labels[j] == i]

    cluster_summaries = {}
    for cluster_id, texts in cluster_text_map.items():
        cluster_topics[cluster_id] = generate_topic_name(" ".join(texts))
        summary = summarize_text(texts)
        cluster_summaries[cluster_id] = summary
        print(f"Cluster {cluster_id} - {cluster_topics[cluster_id]}:")
        for text in texts:
            print(f"- {text}")
        print(f"Summary: {summary}\n")

    visualize_clusters(embeddings, labels, cluster_topics)


if __name__ == "__main__":
    main()
