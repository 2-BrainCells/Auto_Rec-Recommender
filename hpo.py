import os, sys, yaml, torch, optuna
from optuna.trial import TrialState
import optuna.visualization as vis
from preprocessing import read_data, split_data, load_data, device
from autorec import ARDataset, DataLoader, AutoRec
from utils import optim, nn, train_ranking, evaluator
from config import SETTINGS, save_hpo_results

class HiddenPrints:
    def __enter__(self): self._original_stdout, sys.stdout = sys.stdout, open(os.devnull, 'w')
    def __exit__(self, *args): sys.stdout.close(); sys.stdout = self._original_stdout

class OptunaEarlyStoppingCallback:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_value = float('inf')
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        if study.best_value < self.best_value - 1e-4:
            self.best_value, self.no_improvement_count = study.best_value, 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience: study.stop()

def objective(trial):
    # USE SETTINGS YAML FOR SEARCH SPACE
    hidden_dim = trial.suggest_categorical('hidden_dim', SETTINGS['hpo']['search_space']['hidden_dim'])
    lr = trial.suggest_float('lr', SETTINGS['hpo']['search_space']['lr_min'], SETTINGS['hpo']['search_space']['lr_max'], log=True)
    weight_decay = trial.suggest_float('weight_decay', SETTINGS['hpo']['search_space']['weight_decay_min'], SETTINGS['hpo']['search_space']['weight_decay_max'], log=True)
    batch_size = trial.suggest_categorical('batch_size', SETTINGS['hpo']['search_space']['batch_size'])
    split = SETTINGS['data']['split_ratio']

    torch.manual_seed(trial.number)
    df, num_users, num_items = read_data()
    train_data, val_data, _ = split_data(df, split)
    _, _, _, train_inter_mat = load_data(train_data, num_users, num_items)
    _, _, _, val_inter_mat = load_data(val_data, num_users, num_items)

    train_iter = DataLoader(ARDataset(train_inter_mat), batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(ARDataset(val_inter_mat), batch_size=batch_size, shuffle=False)

    net = AutoRec(hidden_dim, num_items, dropout=SETTINGS['training']['dropout_rate']).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    with HiddenPrints():
        metrics = train_ranking(net, train_iter, val_iter, nn.MSELoss(), optimizer, 
                                SETTINGS['hpo']['num_epochs'], device, evaluator, train_inter_mat, val_inter_mat)
        
    val_ndcg = metrics[4] # Index 4 is ndcg
    print(f"Trial {trial.number}: hidden_dim={hidden_dim}, lr={lr:.6f}, batch_size={batch_size}, final_ndcg={val_ndcg[-1]:.6f}")
    return -val_ndcg[-1]

def run():
    study = optuna.create_study(study_name="AutoRec HPO", direction='minimize')
    study.optimize(objective, n_trials=SETTINGS['hpo']['n_trials'], callbacks=[OptunaEarlyStoppingCallback(SETTINGS['hpo']['patience'])])

    best_params = study.best_params
    best_params['split'] = SETTINGS['data']['split_ratio']
    save_hpo_results(best_params, study.best_value)
    
    return best_params