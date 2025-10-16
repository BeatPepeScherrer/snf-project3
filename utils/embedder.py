from sentence_transformers import SentenceTransformer
import numpy as np


class DocumentEmbedder:
    """
    Create document embeddings from raw text using sentence-level embeddings.
    """
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2", pooling="mean", batch_size=32, target_devices=['cpu']*4):

        self.model = SentenceTransformer(model_name)
        self.pooling = pooling
        self.batch_size = batch_size
        self.target_devices = target_devices

    def _pool_embeddings(self, embeddings):
        if self.pooling == "mean":
            return np.mean(embeddings, axis=0)
        elif self.pooling == "max":
            return np.max(embeddings, axis=0)
        elif self.pooling == "mean_max":
            return np.concatenate([np.mean(embeddings, axis=0), np.max(embeddings, axis=0)])
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def encode_document(self, sentences:list)-> np.ndarray:
        """
        sentences: list of str (pre-split sentences or chunks of a document)
        returns: np.ndarray (document embedding)
        """
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return self._pool_embeddings(sentence_embeddings)

    def encode_documents(self, corpus:list, pool_embeddings = True)->list:
        # flatten all sentences while retaining document indices
        flat_sentences = []
        doc_indices = []
        for i, doc in enumerate(corpus):
            flat_sentences.extend(doc)
            doc_indices.extend([i] * len(doc))

        # start multi-processing
        sentence_embeddings = self.embed_sentences(sentences = flat_sentences)

        # group embeddings per document
        doc_embeddings = [[] for _ in corpus]
        for idx, emb in zip(doc_indices, sentence_embeddings):
            doc_embeddings[idx].append(emb)

        # pool per document if requested
        if pool_embeddings:
            out = [self._pool_embeddings(np.array(embs)) for embs in doc_embeddings]
        else:
            out = doc_embeddings
        return out

    def embed_sentences(self, sentences: list):
        # start multi-processing
        pool = self.model.start_multi_process_pool(target_devices=self.target_devices)

        # encode all sentences in batches
        sentence_embeddings = self.model.encode(
            sentences = sentences,
            pool = pool,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # stop multi-processing
        self.model.stop_multi_process_pool(pool)

        return sentence_embeddings
