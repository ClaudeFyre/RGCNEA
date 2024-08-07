import torch
from torch_geometric.loader import DataLoader
from loaddata import *
from model import RGCN, train_rgcn


# Initialize data
base_path = "D:\\LLM\\path\\to\\dbp15k\\DBP15K\\zh_en"
entity_ids_1, entity_ids_2, features_1, features_2, triples_1, triples_2, train_alignments, test_alignments = loaddata(base_path)
# 训练图1和图2的模型
print("Preparing data for graph 1:")
data1, id_map1, num_entities1, num_relations1 = prepare_data(triples_1, entity_ids_1)

print("\nPreparing data for graph 2:")
data2, id_map2, num_entities2, num_relations2 = prepare_data(triples_2, entity_ids_2)

print(f"Graph 1: {num_entities1} entities, {num_relations1} relations")
print(f"Graph 2: {num_entities2} entities, {num_relations2} relations")


print("\nTraining model for graph 1:")
model1 = train_rgcn(data1, num_entities1, num_relations1)

print("\nTraining model for graph 2:")
model2 = train_rgcn(data2, num_entities2, num_relations2)


def align_entities(model1, model2, train_alignments, test_alignments, id_map1, id_map2, top_k=[1, 10, 50]):
    device = next(model1.parameters()).device
    model1.eval()
    model2.eval()

    with torch.no_grad():
        emb1 = model1(data1.edge_index.to(device), data1.edge_type.to(device))
        emb2 = model2(data2.edge_index.to(device), data2.edge_type.to(device))

    # 计算相似度矩阵
    sim_matrix = torch.matmul(emb1, emb2.t())

    # 评估
    def evaluate(alignments):
        source = torch.tensor([id_map1[a] for a in alignments['source'].values], device=device)
        target = torch.tensor([id_map2[b] for b in alignments['target'].values], device=device)

        sim_scores = sim_matrix[source]
        _, top_indices = torch.topk(sim_scores, max(top_k), dim=1)

        hits = {}
        for k in top_k:
            hits[f'Hits@{k}'] = torch.sum(top_indices[:, :k] == target.view(-1, 1)).item() / len(alignments)

        return hits

    print("Evaluating on training alignments:")
    train_results = evaluate(train_alignments)
    print(train_results)

    print("\nEvaluating on testing alignments:")
    test_results = evaluate(test_alignments)
    print(test_results)

# 设置 CUDA 内存分配器的参数
torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 的可用显存
torch.cuda.empty_cache()
# 执行实体对齐
align_entities(model1, model2, train_alignments, test_alignments, id_map1, id_map2)