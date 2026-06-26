import numpy as np
import matplotlib.pyplot as plt

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

Xc = X-X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
coords = U[:, :2]*S[:2]

print("U shape:", end="")
print(U.shape)

print("S shape:", end="")
print(S.shape)

print("Vt shape:", end="")
print(Vt.shape)

print("coords shape:", end = "")
print(coords.shape)
print(coords)

x_coords = []
y_coords = []

for i in range (0,coords.shape[0]):
    x_coords.append(coords[i][0])
for j in range (0,coords.shape[0]):
    y_coords.append(coords[j][1])

plt.scatter(x_coords, y_coords, color='blue', marker='o')

plt.annotate(f"Câu 1: Wake up", xy=(x_coords[0], y_coords[0]))
plt.annotate(f"Câu 2: Family members", xy=(x_coords[1], y_coords[1]))
plt.annotate(f"Câu 3: I love you", xy=(x_coords[2], y_coords[2]))
plt.annotate(f"Câu 4: Interesing book", xy=(x_coords[3], y_coords[3]))
plt.annotate(f"Câu 5: Maths", xy=(x_coords[4], y_coords[4]))
plt.annotate(f"Câu 6: Cooking", xy=(x_coords[5], y_coords[5]))
plt.annotate(f"Câu 7: Algebra", xy=(x_coords[6], y_coords[6]))
plt.annotate(f"Câu 8: Vietnam", xy=(x_coords[7], y_coords[7]))

plt.show()