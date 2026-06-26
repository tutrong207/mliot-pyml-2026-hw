import numpy as np

cau = ["I wake up at 6am every morning", "My family has four members",
       "I love you and you love me", "This book is so interesting", "My favorite subject is maths",
       "She is cooking breakfast", "Linear algebra is the fundamental math subject for AI",
       "Vietnam has just participated in World Cup 2030"]

vocab = sorted({w for s in cau for w in s.lower().split()})
def to_vector(s):
    v = np.zeros(len(vocab))
    for w in s.lower().split():
        if w in vocab:
            v[vocab.index(w)] += 1
    return v
X = np.array([to_vector(s) for s in cau])
print("Task 1:")
print("Shape of X:", end = "")
print(X.shape)
print(X)

print("Task 2:")
X_mean = np.mean(X,axis=0)
print("Shape X_mean:", end="")
print(X_mean.shape)
print(X_mean)

Y = X - X_mean
print("Shape X - X_mean:", end="")
print(Y.shape)
print(Y)

print("Task 3: Cosine Similarity")
def cosine_similarity(X, Y=None):
    if Y is None:
        Y=X
    Xn= X / np.linalg.norm(X,axis=1,keepdims=True)
    Yn= Y / np.linalg.norm(Y,axis=1,keepdims=True)
    return Xn @ Yn.T


print("Task 4: Top 3 highest similarities")

def search(query, top_k=3):
    Top_sentence = []

    CS=cosine_similarity(Y)

    row, col = np.triu_indices(len(cau), k=1)

    pair_values = CS[row, col]

    sorted_indices = np.argsort(pair_values)
    top_3_indices = sorted_indices[:-4:-1]

    print(f"Most 3 similar pairs:")
    for i in top_3_indices:        
        print(f"- {cau[row[i]]} and {cau[col[i]]} (Cosine points:{pair_values[i]:.4f})")

    print("Task 5:")
    print(f"- Most similar: {cau[row[top_3_indices[0]]]} and {cau[col[top_3_indices[0]]]} (Cosine points:{pair_values[top_3_indices[0]]:.4f})")
    print(f"- Most different: {cau[row[sorted_indices[0]]]} and {cau[col[sorted_indices[0]]]} (Cosine points:{pair_values[sorted_indices[0]]:.4f})")

    return Top_sentence

search(cau, top_k=3)

