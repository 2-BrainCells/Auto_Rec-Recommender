# AutoRec System Workflow

Your codebase is structured to run both as a headless evaluation script for academic research (`main.py` / `paper_exp.py`) and as an interactive web dashboard (`app.py`). Here is the step-by-step workflow detailing how data flows from ingestion to final recommendations.

## 1. Data Ingestion & Preprocessing (`preprocessing.py`)
Whether running the CLI or uploading a file to the Streamlit app, the pipeline begins here:
*   **Cleaning:** The raw `values.csv` is loaded. Missing or unknown interactions (`'NC'`, `'NSU'`, blank spaces) are mapped to `NaN`.
*   **Debiasing:** The first 12 columns (demographic data) are explicitly dropped to ensure recommendations are purely behavior-based.
*   **Filtering & Scaling:** Users with more than 40% missing data are removed. Ratings (0-5 scale) are divided by 5.0 to map them into a `[0, 1]` range, which is easier for the neural network to learn.
*   **Matrix Construction:** The data is transformed into a User-Item interaction matrix and split into Train, Validation, and Test sets.

## 2. Configuration & Hyperparameter Optimization (`main.py`, `hpo.py`)
Before the model trains, it needs parameters (like learning rate, hidden dimension size).
*   **Config Check:** The system looks for existing configurations (`autorec_config.pkl` or YAML configs).
*   **HPO Search:** If no optimized config is found (or if forced via the UI), `hpo.py` runs a search to test different combinations of parameters (`hidden_dim`, `batch_size`, `lr`) to find the most mathematically optimal setup for your dataset.

## 3. Model Instantiation & Masked Training (`autorec.py`, `utils.py`)
With parameters set, the PyTorch model initializes and begins training.
*   **AutoRec Architecture:** The model creates an Encoder (compresses the sparse user row via Sigmoid), applies Dropout (to prevent overfitting to the sparse data), and a Decoder (expands it back to predict all item scores).
*   **Masked Loss:** The `ARDataset` class dynamically generates a binary mask. During training, the `masked_loss` function ignores all unrated items. The model only updates its weights based on the errors made on items the student *actually* rated.
*   **Early Stopping:** Training (`train_ranking`) monitors the ranking quality (NDCG) at each epoch limit. If the model stops getting better at ranking relevant items, training halts early.

## 4. Academic Evaluation & Benchmarking (`utils.py`, `main.py`, `paper_exp.py`)
Once trained, the model runs a massive evaluation loop designed specifically for your research paper.
*   **Inference Masking:** The model is asked to predict ratings, but the items the student already interacted with during training are masked out to `-999.0` so they are not recommended again.
*   **Metric Calculation:** The `evaluator()` computes standard error (RMSE) and advanced ranking/fairness metrics (Recall@K, NDCG@K, Diversity, Novelty, Coverage).
*   **Visualizations:** It generates academic-ready figures, like the Long Tail Distribution plot (`autorec_long_tail_figure.png`), proving the model's ability to recommend niche items over popular ones.

## 5. Generating Recommendations (`utils.py`)
The system offers multi-pathway recommendation generation:
*   **Existing Users:** Uses the Autoencoder to "fill in the blanks" of their specific spreadsheet row, returning the Top-K unseen items.
*   **New Users (Cold Start):** Falls back to suggesting globally popular items or simulating an "average user profile" if the user has no history.
*   **Preference-based Users:** Converts qualitative UI sliders (e.g., "I like audiobooks 5/5") into a synthetic interaction profile. It uses cosine similarity to find historically similar users and heavily weights those similarities alongside the Autoencoder's predictions.

## 6. Interactive Dashboard (`app.py`)
All backend logic is wrapped into a modern UI using Streamlit:
*   **Upload & Train:** Users can upload custom CSVs, trigger the HPO visually, and watch the training loss decrease via live progress bars.
*   **Interactive Input:** New users can adjust preference sliders for 27 different educational categories to get instant, personalized hybrid recommendations.
*   **Explainability (XAI):** Rather than a black-box suggestion, the UI tells the user *why* an item was recommended (e.g., "Recommended because users who benefited from [X], which you interacted with, also utilized this").
*   **Latent Space Visuals:** Uses t-SNE to take the complex mathematical brain of the Autoencoder and plot it as a 2D scatter graph, visually showing distinct clusters of dyslexic students with similar learning needs.
