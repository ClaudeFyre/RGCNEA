import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data


def load_entity_ids(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {int(line.strip().split('\t')[0]): line.strip().split('\t')[1] for line in f}
    except Exception as e:
        print(f"Error loading entity ids from {file_path}: {e}")
        return None


def load_features(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading features from {file_path}: {e}")
        return None


def load_triples(file_path):
    try:
        return pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'],
                           dtype={'head': int, 'relation': int, 'tail': int})
    except Exception as e:
        print(f"Error loading triples from {file_path}: {e}")
        return None


def load_alignments(file_path):
    try:
        return pd.read_csv(file_path, sep='\t', header=None, names=['source', 'target'],
                           dtype={'source': int, 'target': int})
    except Exception as e:
        print(f"Error loading alignments from {file_path}: {e}")
        return None


def loaddata(base_path):
    entity_file_1 = f"{base_path}/ent_ids_1"
    entity_file_2 = f"{base_path}/ent_ids_2"
    feature_file_1 = f"{base_path}/id_features_1_embeddings.npy"
    feature_file_2 = f"{base_path}/id_features_2_embeddings.npy"
    triple_file_1 = f"{base_path}/triples_1"
    triple_file_2 = f"{base_path}/triples_2"
    train_alignment_file = f"{base_path}/train.ref"
    test_alignment_file = f"{base_path}/test.ref"

    # 加载数据
    entity_ids_1 = load_entity_ids(entity_file_1)
    entity_ids_2 = load_entity_ids(entity_file_2)
    features_1 = load_features(feature_file_1)
    features_2 = load_features(feature_file_2)
    triples_1 = load_triples(triple_file_1)
    triples_2 = load_triples(triple_file_2)
    train_alignments = load_alignments(train_alignment_file)
    test_alignments = load_alignments(test_alignment_file)

    # 检查是否所有数据都成功加载
    data_loaded = all(x is not None for x in
                      [entity_ids_1, entity_ids_2, features_1, features_2, triples_1, triples_2, train_alignments,
                       test_alignments])

    return entity_ids_1, entity_ids_2, features_1, features_2, triples_1, triples_2, train_alignments, test_alignments


def prepare_data(triples, entity_ids):
    print(f"Original entity count: {len(entity_ids)}")
    print(f"Original triple count: {len(triples)}")

    unique_entities = sorted(set(entity_ids.keys()) | set(triples['head']) | set(triples['tail']))
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_entities)}

    print(f"Total unique entities after merging: {len(unique_entities)}")

    new_triples = triples.copy()
    new_triples['head'] = triples['head'].map(id_map)
    new_triples['tail'] = triples['tail'].map(id_map)

    edge_index = torch.tensor(new_triples[['head', 'tail']].values.T, dtype=torch.long)
    edge_type = torch.tensor(new_triples['relation'].values, dtype=torch.long)

    num_entities = len(id_map)
    num_relations = triples['relation'].nunique()

    print(f"Number of entities: {num_entities}")
    print(f"Number of relations: {num_relations}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge type shape: {edge_type.shape}")
    print(f"Max entity ID in edge_index: {edge_index.max().item()}")
    print(f"Max relation ID in edge_type: {edge_type.max().item()}")

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_entities)
    return data, id_map, num_entities, num_relations



