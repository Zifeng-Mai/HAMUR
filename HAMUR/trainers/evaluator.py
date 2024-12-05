
from typing import Any
import pandas as pd
import torch
import itertools
from sklearn.metrics import ndcg_score
from tqdm import tqdm

class Evaluator():

    def __init__(self, domain_weight) -> None:
        self.domain_weight = domain_weight
        self.metrics = ['HitRate', 'NDCG']
        self.topKs = [5, 10, 15]
        self.max_k = max(self.topKs)
    
    def __call__(self, pred_df: pd.DataFrame) -> Any:
        
        all_logits = []
        all_label = []
        all_domain_id = []

        for user_id, user_df in tqdm(pred_df.groupby('user_id')):
            domain_id = user_df['domain_id'].iloc[0]
            logits = user_df['logits'].tolist()
            label = user_df['label'].tolist()
            all_logits.append(logits)
            all_label.append(label)
            all_domain_id.append(domain_id)
        
        return self.eval(torch.tensor(logits), torch.tensor(label), torch.tensor(domain_id))

    def eval(self, logits, label, domain_id):
        # logits/label: (n_users, 100)
        # domain_id: (n_users)
        result = {}
        result['weighted'] = {}
        for id, weight in self.domain_weight.items():
            domain_result = {}
            index = (domain_id == id).nonzero(as_tuple=False).squeeze()
            domain_logits = logits[index, :] # (domain_users, 100)
            domain_label = label[index, :] # (domain_users, 100)
            _, topk_indices = torch.topk(domain_logits, k=self.max_k, dim=-1)
            for metric, k in itertools.product(self.metrics, self.topKs):
                name = f"{metric}@{k}"
                row_indices = torch.arange(domain_logits.shape[0]).unsqueeze(1).repeat(1, k)
                domain_preds = torch.zeros_like(domain_logits)
                domain_preds[row_indices, topk_indices[:, :k]] = 1
                metric_at_k = getattr(self, f"get_{metric.lower()}")(domain_logits, domain_label, domain_preds, k)
                domain_result[name] = metric_at_k
                if name not in result['weighted']:
                    result['weighted'][name] = 0.
                result['weighted'][name] += metric_at_k * weight
            result[id] = domain_result
        return result

    def get_hitrate(self, logits, labels, preds, k):
        return torch.sum(preds*labels).item() / torch.sum(labels).item()

    def get_ndcg(self, logits, label, preds, k):
        return ndcg_score(label, logits, k=k)
