import os
from preprocessing import read_data, split_data, load_data, device, validate_no_demographics_used
from autorec import ARDataset, DataLoader, AutoRec, optim, nn
from hpo import run
from utils import train_ranking, evaluator, generate_autorec_recommendations, display_recommendations, plot_long_tail_distribution
from config import SETTINGS, get_config
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

    train_data, val_data, test_data = split_data(df, best_params['split'])
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

    train_loss, test_loss, test_rmse, test_recall, test_ndcg, test_div, test_nov, test_cov = train_ranking(
        net=net,
        train_iter=train_iter,
        test_iter=test_iter,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        evaluator=evaluator,
        train_matrix=train_inter_mat,
        test_matrix=test_inter_mat,
        early_stopping_patience=7,
    )

    # If not, simply set it to 5 since that's what your evaluator defaults to.
    k_val = [3, 5, 10]

    # 3. Print the final summary block
    print("\n" + "="*50)
    print("🏆 TRAINING COMPLETED - FINAL METRICS")
    print("="*50)
    print(f"Test RMSE: {test_rmse[-1]:.4f}")
    print(f"Recall@{k_val}:  {test_recall[-1]:.4f}")
    print(f"NDCG@{k_val}:    {test_ndcg[-1]:.4f}")
    print(f"Diversity: {test_div[-1]:.4f}")
    print(f"Novelty:   {test_nov[-1]:.4f}")
    print(f"Coverage:  {test_cov[-1]:.4f}")
    
    print("\n" + "="*80)
    print("🏆 MULTI-K EVALUATION SUMMARY (FOR RESEARCH PAPER)")
    print("="*80)
    
    # We evaluate the trained 'net' on K = 3, 5, and 10
    k_values = [3, 5, 10]
    
    # Print the table header
    print(f"{'Metric':<15} | {'K=3':<10} | {'K=5':<10} | {'K=10':<10}")
    print("-" * 55)
    
    # Dictionaries to store results
    results = {'Recall': [], 'NDCG': [], 'Diversity': [], 'Novelty': [], 'Coverage': []}
    final_rmse = 0.0
    
    # Run the evaluator for each K
    for k in k_values:
        _, rmse, recall, ndcg, div, nov, cov = evaluator(
            net, train_inter_mat, test_inter_mat, loss_fn, device, k=k
        )
        final_rmse = rmse # RMSE is the same for all K
        results['Recall'].append(recall)
        results['NDCG'].append(ndcg)
        results['Diversity'].append(div)
        results['Novelty'].append(nov)
        results['Coverage'].append(cov)

    # Print the rows
    print(f"{'Recall@K':<15} | {results['Recall'][0]:<10.4f} | {results['Recall'][1]:<10.4f} | {results['Recall'][2]:<10.4f}")
    print(f"{'NDCG@K':<15} | {results['NDCG'][0]:<10.4f} | {results['NDCG'][1]:<10.4f} | {results['NDCG'][2]:<10.4f}")
    print(f"{'Diversity':<15} | {results['Diversity'][0]:<10.4f} | {results['Diversity'][1]:<10.4f} | {results['Diversity'][2]:<10.4f}")
    print(f"{'Novelty':<15} | {results['Novelty'][0]:<10.4f} | {results['Novelty'][1]:<10.4f} | {results['Novelty'][2]:<10.4f}")
    print(f"{'Coverage':<15} | {results['Coverage'][0]:<10.4f} | {results['Coverage'][1]:<10.4f} | {results['Coverage'][2]:<10.4f}")
    print("-" * 55)
    print(f"{'RMSE (Global)':<15} | {final_rmse:<10.4f} | {final_rmse:<10.4f} | {final_rmse:<10.4f}")
    print("="*80 + "\n")

    # --- NEW: Generate the Long Tail Figure for the Paper ---
    plot_long_tail_distribution(
        net=net,
        train_matrix=train_inter_mat,
        test_matrix=test_inter_mat,
        device=device,
        k=5,  # We use 5 as the standard for your paper
        save_path="autorec_long_tail_figure.png"
    )
    # plt.figure(figsize=(8, 6))
    # plt.plot(train_loss, label='Train Loss')
    # plt.plot(test_loss, label='Test Loss')
    # plt.plot(test_rmse, '-o', label='Test RMSE')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.title('Training / Testing Loss and Test RMSE over Epochs')
    # plt.show()

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
            top_k=SETTINGS['evaluation']['top_k']
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
