from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

memory = [] # for storing (vector , answer) in the form of tuples in this list....

def add_to_memory(question , answer):
    vec = model.encode(question)  # Question gets conevrted into a vector...
    memory.apppend((vec , answer)) # Successfully added to Mem...


def similar_search(question , threshold = 0.85):
    if not memory:
        return None # Shit memo is empty like you :)
    
    similar_vec = model.encode(question)

    best_score = 0
    best_answer = 0

    for vec , ans in memory:
        score = cosine_similarity(similar_vec , vec)

        if score > best_score:
            best_score=  score
            best_answer = ans
    
    if best_score >= threshold:
        return best_answer
    
    return None

def cosine_similarity(a , b):
    return np.dot(a , b) / (np.linalg.norm(a) * np.linalg.norm(b))