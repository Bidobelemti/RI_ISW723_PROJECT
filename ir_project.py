import pandas as pd
import numpy as np
from iswd723 import (
    clean_text, remove_stopwords, stemming, filter_tokens,
    build_vocabulary, jaccard_similarity, get_tf, get_df, get_idf, 
    get_tfidf, calculate_cos_similarity, avgdl, basic_index, idf_rsj,
    calculate_scores, bm25_rank_query, make_index_inv, get_binary_vector
)

def preprocess_docs(docs_raw):
    return docs_raw.apply(clean_text).apply(remove_stopwords).apply(stemming).apply(filter_tokens)

class IRProject:

    def __init__ (self, docs, vocab, indice_inv, model_name):
        self.query = None
        self.docs = docs
        self.vocab = vocab
        self.indice_inv = indice_inv
        self.model = model_name
    
    def rank(self):
        raise NotImplementedError("Este método debe implementarse en la subclase.")
    
    def setQuery(self, query):
        self.query = query

class TFIDFRI(IRProject):
    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "tfidf")

        vocab_len = len(self.vocab)

        # 1. TF por documento
        self.tf = self.docs.apply(lambda doc: get_tf(doc, self.vocab, vocab_len))

        # 2. Matriz TF
        self.tf_matrix = np.asarray(self.tf.values.tolist())

        # 3. DF y IDF
        self.dft = get_df(self.tf_matrix)
        self.idf_vector = get_idf(self.docs, self.dft)

        # 4. Matriz TF-IDF final
        self.tfidf_matrix = get_tfidf(self.tf_matrix, self.idf_vector)

    def rank(self):
        # Procesamiento limpio idéntico al usado en el corpus
        clean_q = stemming(remove_stopwords(clean_text(self.query)))

        # TF de la query
        query_tf = get_tf(clean_q, self.vocab, len(self.vocab))

        # TF-IDF de la query
        query_tfidf = get_tfidf(query_tf, self.idf_vector)

        # Similaridad coseno
        self.scores = calculate_cos_similarity(self.tfidf_matrix, query_tfidf)

        # Ordenar descendentemente
        self.ranked_indices = np.argsort(-self.scores)

    def getRankedDocs(self):
        return self.ranked_indices, self.scores[self.ranked_indices]

class BM25RI(IRProject):

    def __init__(self, docs, vocab, indice_inv, k1=1.5, b=0.75):
        super().__init__(docs, vocab, indice_inv, "bm25")
        self.k1 = k1
        self.b = b
        # Tokenizar una sola vez
        self.docs_tok = [doc.split() for doc in docs]
        # Construir índice básico
        self.tf_raw, self.dft = basic_index(self.docs_tok)
        # Calcular IDF RSJ
        self.idf_bm25 = idf_rsj(self.dft, len(self.docs_tok))
        # Longitud promedio
        self.avgdl = avgdl(self.docs_tok)
        # Precomputar scores BM25 de todo el corpus
        self.scores, self.terms = calculate_scores(
            self.docs_tok,
            self.avgdl,
            self.tf_raw,
            self.idf_bm25,
            self.k1,
            self.b
        )

    def rank(self):
        clean_q = stemming(remove_stopwords(clean_text(self.query)))
        # Ranking para esta consulta
        self.ranked_indices, self.similarities = bm25_rank_query(
            clean_q,
            self.terms,
            self.scores
        )

    def getRankedDocs(self):
        return self.ranked_indices, self.similarities

class JaccardRI(IRProject):
    
    def __init__(self, docs, vocab, indice_inv):
        super().__init__(docs, vocab, indice_inv, "jaccard")
        self.binary_matrix = get_binary_vector(self.docs, self.vocab, self.indice_inv)
    
    def rank(self):
        # Preprocesar query
        clean_q = stemming(remove_stopwords(clean_text(self.query)))
        # Vector binario de la query
        self.query_binary_vector = np.zeros(len(self.vocab), dtype=int)
        for word in clean_q.split():
            if word in self.vocab:
                self.query_binary_vector[self.vocab[word]] = 1
        # Calcular scores Jaccard contra cada documento
        self.scores = np.array([
            jaccard_similarity(doc_vector, self.query_binary_vector)
            for doc_vector in self.binary_matrix
        ])
        # Ordenar documentos por similitud
        self.ranked_indices = np.argsort(self.scores)[::-1]

    def getRankedDocs(self):
        # Ya están ordenados en self.ranked_indices
        return self.ranked_indices, self.scores[self.ranked_indices]
    
# =========================================================
if __name__ == "__main__":
    # ------------------ CARGAR CORPUS ------------------
    df = pd.read_csv("data/bbc_news.csv")
    docs_raw = df["description"]
    # ------------------ PREPROCESO ---------------------
    docs = preprocess_docs(docs_raw)
    # ------------------ VOCABULARIO ---------------------
    vocab= build_vocabulary(docs)
    vocab_map = {term: idx for idx, term in enumerate(vocab)}
    # ------------------ ÍNDICE INVERTIDO ------------------
    indice_inv = make_index_inv(docs)
    # Ejemplo de consulta
    query = "The war between Syria and Iraq leaves thousands wounded"
    # ------------------ MODELO JACCARD --------------------
    model = 3
    if model == 1:
        jaccard_model = JaccardRI(docs, vocab_map, indice_inv)
        jaccard_model.setQuery(query)
        jaccard_model.rank()
        ranked_indices, scores = jaccard_model.getRankedDocs()
        print("Top documentos por Jaccard:")
        for i in range(5):
            print(f"Doc {ranked_indices[i]} - Score {scores[i]:.4f}")
    elif model == 2:
        # Aquí se podría implementar otro modelo, como BM25
        bm25_model = BM25RI(docs, vocab_map, indice_inv)
        bm25_model.setQuery(query)
        bm25_model.rank()
        ranked_indices, scores = bm25_model.getRankedDocs()
        print("Top documentos por BM25:")
        for i in range(5):
            print(f"Doc {ranked_indices[i]} - Score {scores[i]:.4f}")
    elif model == 3:
        # Aquí se podría implementar otro modelo, como TF-IDF
        tfidf_model = TFIDFRI(docs, vocab_map, indice_inv)
        tfidf_model.setQuery(query)
        tfidf_model.rank()
        ranked_indices, scores = tfidf_model.getRankedDocs()
        print("Top documentos por TF-IDF:")
        for i in range(5):
            print(f"Doc {ranked_indices[i]} - Score {scores[i]:.4f}")
    else:
        print("Modelo no reconocido.")