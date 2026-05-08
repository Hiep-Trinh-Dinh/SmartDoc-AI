from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
query = "Việt Nam có bao nhiêu tỉnh thành?"
passages = [
    "Việt Nam là một quốc gia nằm ở Đông Nam Á, có 63 tỉnh thành phố trực thuộc trung ương.",
    "Bánh mì là một món ăn đường phố nổi tiếng của Việt Nam.",
    "The capital of France is Paris."
]

pairs = [[query, p] for p in passages]
scores = reranker.predict(pairs)

for p, s in zip(passages, scores):
    print(f"Score: {s:.4f} | Passage: {p}")
