import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd
from preprocessing import read_data, load_data, device
from autorec import ARDataset, DataLoader, AutoRec
from utils import optim, nn, train_ranking, evaluator
from config import get_config

# Custom dataset for the "No Mask" ablation study
class ARDataset_NoMask(torch.utils.data.Dataset):
    def __init__(self, interaction_matrix):
        self.data = torch.tensor(interaction_matrix, dtype=torch.float32)
        # THE ABLATION: Force the model to treat missing values as actual 0.0 ratings
        self.mask = torch.ones_like(self.data) 

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

def run_experiments():
    config = get_config()
    best_params = config['hyperparameters']
    
    df, num_users, num_items = read_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "Proposed AutoRec": [],
        "Ablation: No Dropout": [],
        "Ablation: Dense MSE (No Mask)": []
    }
    
    fold = 1
    for train_idx, test_idx in kf.split(df):
        print(f"\n" + "="*40)
        print(f"RUNNING FOLD {fold}/5")
        print("="*40)
        
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        _, _, _, train_inter_mat = load_data(train_data, num_users, num_items)
        _, _, _, test_inter_mat = load_data(test_data, num_users, num_items)
        
        test_dataset = ARDataset(test_inter_mat)
        test_iter = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

        # --- MODEL 1: Proposed AutoRec ---
        print("\n[Training Proposed AutoRec]")
        train_dataset = ARDataset(train_inter_mat)
        train_iter = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        
        net = AutoRec(best_params['hidden_dim'], num_items, dropout=0.2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        metrics = train_ranking(net, train_iter, test_iter, nn.MSELoss(), optimizer, 20, device, evaluator, train_inter_mat, test_inter_mat)
        results["Proposed AutoRec"].append([m[-1] for m in metrics[2:]]) # rmse, recall, ndcg, div, nov, cov
        
        # --- MODEL 2: Ablation (No Dropout) ---
        print("\n[Training Ablation: No Dropout]")
        net_no_drop = AutoRec(best_params['hidden_dim'], num_items, dropout=0.0).to(device)
        optimizer_no_drop = optim.Adam(net_no_drop.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        metrics_no_drop = train_ranking(net_no_drop, train_iter, test_iter, nn.MSELoss(), optimizer_no_drop, 20, device, evaluator, train_inter_mat, test_inter_mat)
        results["Ablation: No Dropout"].append([m[-1] for m in metrics_no_drop[2:]])
        
        # --- MODEL 3: Ablation (No Mask) ---
        print("\n[Training Ablation: Dense MSE (No Mask)]")
        train_dataset_nomask = ARDataset_NoMask(train_inter_mat)
        train_iter_nomask = DataLoader(train_dataset_nomask, batch_size=best_params['batch_size'], shuffle=True)
        
        net_no_mask = AutoRec(best_params['hidden_dim'], num_items, dropout=0.2).to(device)
        optimizer_no_mask = optim.Adam(net_no_mask.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        metrics_no_mask = train_ranking(net_no_mask, train_iter_nomask, test_iter, nn.MSELoss(), optimizer_no_mask, 20, device, evaluator, train_inter_mat, test_inter_mat)
        results["Ablation: Dense MSE (No Mask)"].append([m[-1] for m in metrics_no_mask[2:]])
        
        fold += 1

    print("\n\n" + "="*80)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS (AVERAGED)")
    print("="*80)
    
    final_table = []
    for model_name, fold_metrics in results.items():
        avg_metrics = np.mean(fold_metrics, axis=0)
        final_table.append({
            "Configuration": model_name,
            "RMSE": avg_metrics[0],
            "Recall@5": avg_metrics[1],
            "NDCG@5": avg_metrics[2],
            "Diversity": avg_metrics[3],
            "Novelty": avg_metrics[4],
            "Coverage": avg_metrics[5]
        })
        
    df_results = pd.DataFrame(final_table)
    print(df_results.to_markdown(index=False, floatfmt=".3f"))

if __name__ == '__main__':
    run_experiments()