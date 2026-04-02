# Re ranking 
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def rank_docs(query: str, docsRelatedToQuery):
    pairs = [(query, doc["content"]) for doc in docsRelatedToQuery]
    scores = reranker.predict(pairs)

    docs_with_scores = list(zip(docsRelatedToQuery, scores))
    reranked_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked_docs[:3]]

def context_of_rank_docs(top_docs):
    return "\n\n".join([
        f"""[Source {i+1}]
        Page: {doc['metadata'].get('page')}
        Content:
        {doc['content']}"""
        for i, doc in enumerate(top_docs)
    ])
