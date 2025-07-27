import os
from preprocessing import read_data, split_data, load_data, device, validate_no_demographics_used
from autorec import ARDataset, DataLoader, AutoRec, optim, nn
from hpo import run
from utils import train_ranking, evaluator, generate_autorec_recommendations, display_recommendations
from config import get_config
import matplotlib.pyplot as plt
import pandas as pd

def auto_rec_runner():
    """
    Main execution function for AutoRec recommendation system.
    Handles data validation, hyperparameter optimization, model training, and recommendation generation.
    """
    try:
        df_full = pd.read_csv('data/values.csv')
        validate_no_demographics_used(df_full)
    except Exception as e:
        print(f"Validation warning: {e}")

    if os.path.exists("autorec_config.pkl"):
        config = get_config()
        if config['optimization_results']['optimized']:
            print("Using optimized configuration")
            best_params = config['hyperparameters']
        else:
            print("Config exists but not optimized. Running HPO...")
            best_params = run()
            config = get_config()
            best_params = config['hyperparameters']
    else:
        print("No config found. Running HPO...")
        best_params = run()
        config = get_config()
        best_params = config['hyperparameters']

    print(f"Runnable Configs: {best_params}")

    batch_size = best_params["batch_size"]
    df, num_users, num_items = read_data()
    print(df.shape, num_users, num_items)

    train_data, test_data = split_data(df, best_params['split'])
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

    net = AutoRec(best_params["hidden_dim"], num_items)
    print(net)
    net = net.to(device)

    num_epochs = 20
    optimizer = optim.Adam(net.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    loss_fn = nn.MSELoss()

    train_loss, test_loss, test_rmse = train_ranking(
        net=net,
        train_iter=train_iter,
        test_iter=test_iter,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        evaluator=evaluator,
        inter_mat=test_inter_mat,
        early_stopping_patience=7,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.plot(test_rmse, '-o', label='Test RMSE')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training / Testing Loss and Test RMSE over Epochs')
    plt.show()

    print("\n" + "="*50)
    print("GENERATING RECOMMENDATIONS (NO DEMOGRAPHICS USED)")
    print("="*50)

    sample_users = [0, 500, 1103] if test_inter_mat.shape[0] > 10 else [0]
    for user_id in sample_users:
        recommendations = generate_autorec_recommendations(
            model=net,
            interaction_matrix=test_inter_mat,
            device=device,
            user_id=user_id,
            top_k=5
        )
        display_recommendations(recommendations, f"Existing User {user_id}")

    new_user_recs_pop = generate_autorec_recommendations(
        model=net,
        interaction_matrix=test_inter_mat,
        device=device,
        user_id=None,
        new_user_method='popularity',
        top_k=5
    )
    display_recommendations(new_user_recs_pop, "New User (Popular Items)")

    new_user_recs_avg = generate_autorec_recommendations(
        model=net,
        interaction_matrix=test_inter_mat,
        device=device,
        user_id=None,
        new_user_method='average',
        top_k=5
    )
    display_recommendations(new_user_recs_avg, "New User (Average Profile)")

if __name__ == '__main__':
    auto_rec_runner()
