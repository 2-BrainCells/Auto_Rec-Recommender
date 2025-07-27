import os, sys
import yaml
import torch
from preprocessing import read_data, split_data, load_data, device
from autorec import ARDataset, DataLoader, AutoRec
from utils import optim, nn, evaluator, train_ranking, EarlyStoppingCallback
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis
import plotly

class HiddenPrints:
    """
    Context manager to suppress stdout output during training operations.
    Used to reduce verbose output during hyperparameter optimization trials.
    """
    
    def __enter__(self):
        """Redirect stdout to null device."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original stdout."""
        sys.stdout.close()
        sys.stdout = self._original_stdout

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    Defines the search space and trains model with suggested parameters.
    
    Args:
        trial: Optuna trial object containing suggested hyperparameters
        
    Returns:
        Final test RMSE score for the trial
    """
    hidden_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128, 512, 1024])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
    split = trial.suggest_float('split', 0.1, 0.25)

    torch.manual_seed(trial.number)

    df, num_users, num_items = read_data()
    train_data, test_data = split_data(df, split)
    _, _, _, train_inter_mat = load_data(train_data, num_users, num_items)
    _, _, _, test_inter_mat = load_data(test_data, num_users, num_items)

    train_dataset = ARDataset(train_inter_mat)
    test_dataset = ARDataset(test_inter_mat)

    train_iter = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_iter = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    net = AutoRec(hidden_dim, num_items).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    num_epochs = 50

    with HiddenPrints():
        _, _, test_rmse = train_ranking(
            net=net,
            train_iter=train_iter,
            test_iter=test_iter,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            evaluator=evaluator,
            inter_mat=test_inter_mat
        )

    print(f"Trial {trial.number}: hidden_dim={hidden_dim}, lr={lr:.6f}, "
          f"weight_decay={weight_decay:.6f}, batch_size={batch_size}, "
          f"split={split:.3f}, final_rmse={test_rmse[-1]:.6f}")

    return test_rmse[-1]

def create_config_yaml(best_params):
    """
    Create YAML configuration file with optimized hyperparameters.
    
    Args:
        best_params: Dictionary of best hyperparameters found
        
    Returns:
        Configuration dictionary
    """
    filename = "Auto_Rec_best_params"
    config = {
        'model_config': {
            'name': 'AutoRec'
        },
        'hyperparameters': {
            'hidden_dim': best_params['hidden_dim'],
            'lr': best_params['lr'],
            'weight_decay': best_params['weight_decay'],
            'batch_size': best_params['batch_size'],
            'split': best_params['split']
        },
        'training': {
            'num_epochs': 50,
            'device': 'cuda',
            'seed': 42
        },
        'data': {
            'num_workers': 0,
            'shuffle_train': True,
            'shuffle_test': False
        },
        'optimization_results': {
            'total_trials': 100,
            'study_name': 'AutoRec HPO'
        }
    }

    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    print(f"Configuration saved to {filename}")
    return config

from config import save_hpo_results

def run():
    """
    Execute hyperparameter optimization study with early stopping and visualization.
    
    Returns:
        Dictionary of best hyperparameters found during optimization
    """
    early_stopping = EarlyStoppingCallback()
    study = optuna.create_study(study_name="AutoRec HPO", direction='minimize')
    study.optimize(objective, n_trials=100, callbacks=[early_stopping])

    config = save_hpo_results(study.best_params, study.best_value)

    print(f"Best test_rmse loss: {study.best_value}")
    print(f"Best hyperparameters: {study.best_params}")

    history_plot = vis.plot_optimization_history(study)
    history_plot.show()

    param_importance = vis.plot_param_importances(study)
    param_importance.show()

    parallel_plot = vis.plot_parallel_coordinate(study)
    parallel_plot.show()

    return study.best_params
