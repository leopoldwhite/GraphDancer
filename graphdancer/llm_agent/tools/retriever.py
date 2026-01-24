import os
import pickle
import logging
from typing import List

import faiss
import numpy as np
import sentence_transformers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NODE_TEXT_KEYS = {
    'maple': {
        'paper': ['title'], 'author': ['name'], 'venue': ['name']
    },
    'amazon': {
        'item': ['title'], 'brand': ['name']
    },
    'biomedical': {
        'Anatomy': ['name'], 'Biological_Process': ['name'], 'Cellular_Component': ['name'],
        'Compound': ['name'], 'Disease': ['name'], 'Gene': ['name'], 'Molecular_Function': ['name'],
        'Pathway': ['name'], 'Pharmacologic_Class': ['name'], 'Side_Effect': ['name'], 'Symptom': ['name']
    },
    'legal': {
        'opinion': ['plain_text'], 'opinion_cluster': ['syllabus'],
        'docket': ['pacer_case_id', 'case_name'], 'court': ['full_name']
    },
    'goodreads': {
        'book': ['title'], 'author': ['name'], 'publisher': ['name'], 'series': ['title']
    },
    'dblp': {
        'paper': ['title'], 'author': ['name', 'organization'], 'venue': ['name']
    },
}


class Retriever:
    def __init__(self, args, graph, cache=True, cache_dir=None):
        logger.info("Initializing retriever")

        self.use_gpu = getattr(args, 'faiss_gpu', False)
        self.node_text_keys = getattr(args, 'node_text_keys', NODE_TEXT_KEYS[args.dataset])
        self.model_name = getattr(args, 'embedder_name', 'sentence-transformers/all-mpnet-base-v2')
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        self.graph = graph
        self.cache = getattr(args, 'embed_cache', True)
        self.cache_dir = getattr(args, 'embed_cache_dir', None) or os.getcwd()

        self.reset()

    def reset(self):
        docs, ids, meta_type = self.process_graph()
        save_model_name = self.model_name.split('/')[-1]

        cache_file = os.path.join(self.cache_dir, f'cache-{save_model_name}.pkl')
        if self.cache and os.path.isfile(cache_file):
            embeds, self.doc_lookup, self.doc_type = pickle.load(open(cache_file, 'rb'))
            assert self.doc_lookup == ids
            assert self.doc_type == meta_type
        else:
            embeds = self._encode_docs(docs)
            self.doc_lookup = ids
            self.doc_type = meta_type
            os.makedirs(self.cache_dir, exist_ok=True)
            pickle.dump([embeds, ids, meta_type], open(cache_file, 'wb'))

        self.init_index_and_add(embeds)

    def process_graph(self):
        docs: List[str] = []
        ids: List[str] = []
        meta_type: List[str] = []

        for node_type_key in self.graph.keys():
            node_type = node_type_key.split('_nodes')[0]
            logger.info(f'loading text for {node_type}')
            for nid in self.graph[node_type_key]:
                docs.append(str(self.graph[node_type_key][nid]['features'][self.node_text_keys[node_type][0]]))
                ids.append(nid)
                meta_type.append(node_type)
        return docs, ids, meta_type

    def _encode_docs(self, docs):
        # single-process encode for portability
        return self.model.encode(docs, show_progress_bar=False, convert_to_numpy=True)

    def _initialize_faiss_index(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def init_index_and_add(self, embeds):
        logger.info("Initialize the index...")
        dim = embeds.shape[1]
        self._initialize_faiss_index(dim)
        self.index.add(embeds)

    def reset_index(self):
        if hasattr(self, 'index') and self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []

    def search_single(self, query, topk: int = 10):
        if self.index is None:
            raise ValueError("Index is not initialized")

        query_embed = self.model.encode(query, show_progress_bar=False)

        D, I = self.index.search(query_embed[None, :], topk)
        original_indice = np.array(self.doc_lookup)[I].tolist()[0][0]
        original_type = np.array(self.doc_type)[I].tolist()[0][0]

        return original_indice, self.graph[f'{original_type}_nodes'][original_indice]

