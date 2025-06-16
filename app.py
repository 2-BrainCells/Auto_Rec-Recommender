import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from autorec import AutoRec, ARDataset, DataLoader
from preprocessing import split_data, load_data, device
from utils import (train_ranking, evaluator, masked_loss, generate_autorec_recommendations,
                   load_and_use_config, generate_preference_based_recommendations)
from hpo import run as run_hpo
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from config import get_config

st.set_page_config(
    page_title="AutoRec Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AutoRecStreamlitUI:
    def __init__(self):
        self.model = None
        self.interaction_matrix = None
        self.num_users = 0
        self.num_items = 0
        self.trained = False
        self.hpo_config_file = "Auto_Rec_best_params"
        self.pkl_config_file = "autorec_config.pkl"

    @st.cache_data
    def process_uploaded_data(_self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Original data shape:", df.shape)
            st.write("First few rows:")
            st.dataframe(df.head())

            if df.shape[1] < 13:
                st.error("CSV file must have at least 13 columns (user data + item interactions)")
                return None, 0, 0

            df_items = df.iloc[:, 12:]
            df_items = df_items.replace(['NC', 'NSU', ' '], np.nan)
            df_items = df_items.apply(pd.to_numeric, errors='coerce')

            max_nan_ratio = 0.4
            max_nan_allowed = int(df_items.shape[1] * max_nan_ratio)
            df_items = df_items[df_items.isnull().sum(axis=1) <= max_nan_allowed]
            df_items.reset_index(drop=True, inplace=True)

            num_users = len(df_items.index)
            num_items = len(df_items.columns)

            df_processed = df_items.stack().rename_axis(('User', 'Item')).reset_index(name='Score')
            item_labels, _ = pd.factorize(df_processed['Item'])
            df_processed['Item'] = item_labels
            df_processed['Score'] /= 5.0

            st.success(f"✅ Data processed successfully! Users: {num_users}, Items: {num_items}")
            return df_processed, num_users, num_items

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None, 0, 0

    def check_hpo_config(self):
        config = None
        
        if os.path.exists(self.pkl_config_file):
            try:
                config_data = get_config()
                if config_data['optimization_results']['optimized']:
                    config = config_data['hyperparameters']
                    st.success("✅ Found existing HPO configuration (pickle)!")
                    with st.expander("📋 Existing HPO Configuration", expanded=False):
                        config_df = pd.DataFrame([
                            {"Parameter": k, "Value": str(v)}
                            for k, v in config.items()
                        ])
                        st.dataframe(config_df, hide_index=True)
                    return config
            except Exception as e:
                st.warning(f"⚠️ Error loading pickle HPO config: {str(e)}")
        
        if os.path.exists(self.hpo_config_file):
            try:
                config_data = load_and_use_config(self.hpo_config_file)
                config = config_data['hyperparameters']
                st.success("✅ Found existing HPO configuration (YAML)!")
                with st.expander("📋 Existing HPO Configuration", expanded=False):
                    config_df = pd.DataFrame([
                        {"Parameter": k, "Value": str(v)}
                        for k, v in config.items()
                    ])
                    st.dataframe(config_df, hide_index=True)
                return config
            except Exception as e:
                st.warning(f"⚠️ Error loading YAML HPO config: {str(e)}")
        
        st.info("ℹ️ No existing HPO configuration found.")
        return None

    @st.cache_resource
    def run_hyperparameter_optimization(_self, df, num_users, num_items):
        try:
            st.info("🔍 Starting hyperparameter optimization...")
            temp_csv_path = "temp_uploaded_data.csv"
            df.to_csv(temp_csv_path, index=False)

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Running hyperparameter optimization... This may take a while.")
            progress_bar.progress(0.5)

            best_params = run_hpo()
            progress_bar.progress(1.0)
            status_text.text("✅ Hyperparameter optimization completed!")

            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

            progress_bar.empty()
            status_text.empty()

            st.success("✅ HPO completed successfully!")

            with st.expander("📊 Optimized Parameters", expanded=True):
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": str(v)}
                    for k, v in best_params.items()
                ])
                st.dataframe(params_df, hide_index=True)

            return best_params

        except Exception as e:
            st.error(f"Error during HPO: {str(e)}")
            return None

    @st.cache_resource
    def train_model(_self, df, num_users, num_items, training_params, num_epochs=20):
        try:
            st.info(f"Starting training with parameters: {training_params}")
            
            train_data, test_data = split_data(df, training_params.get('split', 0.2))
            _, _, _, train_inter_mat = load_data(train_data, num_users, num_items)
            _, _, _, test_inter_mat = load_data(test_data, num_users, num_items)

            st.info(f"Data split complete. Training matrix shape: {train_inter_mat.shape}")

            train_dataset = ARDataset(train_inter_mat)
            test_dataset = ARDataset(test_inter_mat)

            batch_size = training_params.get('batch_size', 64)
            train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            hidden_dim = training_params.get('hidden_dim', 512)
            model = AutoRec(hidden_dim, num_items).to(device)
            st.info(f"Model initialized with hidden_dim: {hidden_dim}")

            lr = training_params.get('lr', 0.001)
            weight_decay = training_params.get('weight_decay', 1e-4)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.MSELoss()

            progress_bar = st.progress(0)
            status_text = st.empty()

            train_losses = []
            test_losses = []
            test_rmses = []

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                for batch, mask in train_iter:
                    batch, mask = batch.to(device), mask.to(device)
                    optimizer.zero_grad()
                    predictions = model(batch)
                    loss = masked_loss(predictions, batch, mask, loss_fn)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                train_l = total_loss / len(train_iter)
                test_l, rmse = evaluator(model, test_iter, test_inter_mat, loss_fn)
                
                train_losses.append(train_l)
                test_losses.append(test_l)
                test_rmses.append(rmse)

                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_l:.4f}, Test RMSE: {rmse:.4f}")

            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Training completed! Final RMSE: {test_rmses[-1]:.4f}")
            return model, test_inter_mat, train_losses, test_losses, test_rmses

        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            st.exception(e)
            return None, None, [], [], []

    def plot_training_curves(self, train_losses, test_losses, test_rmses):
        if train_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(train_losses, label='Train Loss', color='blue')
            ax1.plot(test_losses, label='Test Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Test Loss')
            ax1.legend()

            ax2.plot(test_rmses, label='Test RMSE', color='green', marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Test RMSE')
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

    def render_sidebar(self):
        st.sidebar.header("🎯 AutoRec Recommender")
        st.sidebar.markdown("---")

        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with user-item interactions"
        )

        if uploaded_file is not None:
            st.sidebar.success("✅ File uploaded successfully!")

            st.sidebar.subheader("🔧 Optimization Settings")
            use_existing_hpo = st.sidebar.checkbox(
                "Use Existing HPO Config",
                value=True,
                help="Use existing hyperparameter optimization results if available"
            )

            force_new_hpo = st.sidebar.checkbox(
                "Force New HPO",
                value=False,
                help="Run new hyperparameter optimization even if existing config is found"
            )

            if force_new_hpo or not use_existing_hpo:
                st.sidebar.subheader("🎛️ Manual Parameters")
                hidden_dim = st.sidebar.selectbox(
                    "Hidden Dimension",
                    [128, 256, 512, 1024],
                    index=2,
                    help="Size of the hidden layer in the autoencoder"
                )

                learning_rate = st.sidebar.slider(
                    "Learning Rate",
                    0.0001, 0.01, 0.001,
                    format="%.4f",
                    help="Learning rate for the optimizer"
                )

                batch_size = st.sidebar.selectbox(
                    "Batch Size",
                    [16, 32, 64, 128],
                    index=2,
                    help="Number of samples per batch"
                )

                test_split = st.sidebar.slider(
                    "Test Split Ratio",
                    0.1, 0.3, 0.2,
                    help="Proportion of data for testing"
                )

                manual_params = {
                    'hidden_dim': hidden_dim,
                    'lr': learning_rate,
                    'batch_size': batch_size,
                    'split': test_split,
                    'weight_decay': 1e-4
                }
            else:
                manual_params = {}

            num_epochs = st.sidebar.slider(
                "Training Epochs",
                5, 50, 20,
                help="Number of training epochs"
            )

            st.sidebar.markdown("---")
            user_type = st.sidebar.radio(
                "Select User Type:",
                ["New User", "Existing User"],
                help="Choose whether you're a new user or want to see recommendations for an existing user"
            )

            return uploaded_file, {
                'use_existing_hpo': use_existing_hpo,
                'force_new_hpo': force_new_hpo,
                'manual_params': manual_params,
                'num_epochs': num_epochs
            }, user_type

        return None, {}, "New User"

    def render_user_input(self, user_type):
        if user_type == "New User":
            st.subheader("👤 New User Profile")
            
            categories = {
                "Audio Book Tools": [0, 1],
                "Color-Coded Text": [2],
                "Assistive Writing Tools": [3, 4],
                "Text Structuring Tools": [5, 6],
                "Pre-made Visual Aids": [7, 8, 9],
                "Digital Books": [10],
                "Digital Tutor": [11],
                "Visual Memory Aids": [12, 13],
                "Multimedia Lesson Recording": [14, 15, 16],
                "Supplementary Research": [16],
                "Personal Reader Support": [17],
                "Self-made Study Aids": [18, 19, 20],
                "Repetition Strategy": [21],
                "Active Reading Markup": [22, 23],
                "Group Study": [24],
                "Tutoring Support": [25],
                "Peer Association": [26],
                "In-person Attendance": [27],
                "Online Lessons": [28],
                "Classroom Support Aids": [29, 30],
                "Note Taking": [31],
                "Lesson Planning": [32],
                "Assessment Adaptation": [33],
                "Written Assessment": [34],
                "Oral Assessment": [35],
                "Individual Assessment": [36],
                "Online Study Resources": [37]
            }

            st.markdown("**📊 Rate your interest in different learning aid categories:**")
            st.markdown("*Use the sliders below to indicate how much you'd like each type of learning aid (1=Not interested, 5=Very interested)*")

            preferences = {}
            col1, col2 = st.columns(2)
            category_items = list(categories.items())
            mid_point = len(category_items) // 2

            with col1:
                for category, item_ids in category_items[:mid_point]:
                    rating = st.slider(
                        f"🎯 {category}",
                        min_value=1, max_value=5, value=3,
                        key=f"pref_{category}",
                        help=f"Rate your interest in {category.lower()}"
                    )
                    preferences[category] = {
                        'rating': rating,
                        'item_ids': item_ids
                    }

            with col2:
                for category, item_ids in category_items[mid_point:]:
                    rating = st.slider(
                        f"🎯 {category}",
                        min_value=1, max_value=5, value=3,
                        key=f"pref_{category}",
                        help=f"Rate your interest in {category.lower()}"
                    )
                    preferences[category] = {
                        'rating': rating,
                        'item_ids': item_ids
                    }

            if st.checkbox("📋 Show Preference Summary", value=False):
                st.subheader("Your Preferences")
                pref_df = pd.DataFrame([
                    {
                        'Category': category,
                        'Interest Level': '⭐' * data['rating'],
                        'Rating': data['rating']
                    }
                    for category, data in preferences.items()
                    if data['rating'] > 3
                ])
                if not pref_df.empty:
                    st.dataframe(pref_df, hide_index=True)
                else:
                    st.info("Adjust sliders above 3 to see your preferences!")

            return {"type": "preferences", "data": preferences}

        else:
            st.subheader("🔍 Existing User")
            if self.num_users > 0:
                user_id = st.number_input(
                    "User ID",
                    min_value=0,
                    max_value=self.num_users - 1,
                    value=0,
                    help=f"Select user ID (0 to {self.num_users - 1})"
                )

                if self.interaction_matrix is not None:
                    user_interactions = np.sum(self.interaction_matrix[user_id] > 0)
                    if user_interactions > 0:
                        avg_rating = np.mean(self.interaction_matrix[user_id][self.interaction_matrix[user_id] > 0])
                        st.info(f"""
                        **User {user_id} Stats:**
                        - Interactions: {user_interactions}
                        - Avg Rating: {avg_rating:.2f}
                        """)

                return {"type": "existing", "data": user_id}
            else:
                st.error("No trained model available")
                return {"type": "existing", "data": 0}

    def generate_recommendations(self, user_input, user_type):
        if self.model is None or self.interaction_matrix is None:
            st.error("Model not trained. Please train the model first.")
            return []

        try:
            if user_type == "New User":
                if user_input["type"] == "preferences":
                    meaningful_prefs = {k: v for k, v in user_input["data"].items()
                                        if v['rating'] != 3}

                    if meaningful_prefs:
                        recommendations = generate_preference_based_recommendations(
                            model=self.model,
                            interaction_matrix=self.interaction_matrix,
                            device=device,
                            preferences=user_input["data"],
                            top_k=10
                        )
                        st.success("🎯 Recommendations based on your preferences!")
                        self.show_preference_impact(user_input["data"], recommendations)
                    else:
                        recommendations = generate_autorec_recommendations(
                            model=self.model,
                            interaction_matrix=self.interaction_matrix,
                            device=device,
                            user_id=None,
                            new_user_method='popularity',
                            top_k=10
                        )
                        st.info("📈 Popular items (no specific preferences detected)")
                else:
                    recommendations = generate_autorec_recommendations(
                        model=self.model,
                        interaction_matrix=self.interaction_matrix,
                        device=device,
                        user_id=None,
                        new_user_method='popularity',
                        top_k=10
                    )
            else:
                recommendations = generate_autorec_recommendations(
                    model=self.model,
                    interaction_matrix=self.interaction_matrix,
                    device=device,
                    user_id=user_input["data"],
                    top_k=10
                )

            return recommendations

        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []

    def show_preference_impact(self, preferences, recommendations):
        st.subheader("🔍 How Your Preferences Influenced Recommendations")

        category = {
            "Audio Book Tools": [0, 1],
            "Color-Coded Text": [2],
            "Assistive Writing Tools": [3, 4],
            "Text Structuring Tools": [5, 6],
            "Pre-made Visual Aids": [7, 8, 9],
            "Digital Books": [10],
            "Digital Tutor": [11],
            "Visual Memory Aids": [12, 13],
            "Multimedia Lesson Recording": [14, 15, 16],
            "Supplementary Research": [17],
            "Personal Reader Support": [18],
            "Self-made Study Aids": [19, 20, 21],
            "Repetition Strategy": [22],
            "Active Reading Markup": [23, 24],
            "Group Study": [25],
            "Tutoring Support": [26],
            "Peer Association": [27],
            "In-person Attendance": [28],
            "Online Lessons": [29],
            "Classroom Support Aids": [30, 31],
            "Note Taking": [32],
            "Lesson Planning": [33],
            "Assessment Adaptation": [34],
            "Written Assessment": [35],
            "Oral Assessment": [36],
            "Individual Assessment": [37],
            "Online Study Resources": [38]
        }

        rec_items = [item_id for item_id, _ in recommendations[:5]]
        impact_data = []

        for category, data in preferences.items():
            if data['rating'] > 3:
                matching_recs = len([item for item in rec_items if item in data['item_ids']])
                if matching_recs > 0:
                    impact_data.append({
                        'Category': category,
                        'Your Rating': '⭐' * data['rating'],
                        'Items Recommended': matching_recs
                    })

        if impact_data:
            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, hide_index=True)
        else:
            st.info("Your recommendations are based on similarity with other users who have similar overall preferences.")

    def display_recommendations_ui(self, recommendations, user_type):
        if not recommendations:
            st.warning("No recommendations available.")
            return

        st.subheader(f"🎯 Top Recommendations for {user_type}")

        rec_df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Item ID': item_id,
                'Predicted Score': f"{score:.4f}",
                'Item Name': f"Learning Aid {item_id}"
            }
            for i, (item_id, score) in enumerate(recommendations)
        ])

        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            scores = [float(score.split(': ')[-1]) if ': ' in score else float(score) for score in rec_df['Predicted Score']]
            fig_bar = px.bar(
                x=rec_df['Rank'],
                y=scores,
                title='Recommendation Scores',
                labels={'y': 'Score', 'x': 'Rank'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                values=scores[:5],
                names=[f"Item {rec_df.iloc[i]['Item ID']}" for i in range(min(5, len(rec_df)))],
                title="Top 5 Items Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    def show_training_status(self):
        if self.trained:
            st.success("✅ Model is trained and ready for recommendations!")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Users", self.num_users)
            with col2:
                st.metric("Items", self.num_items)
            with col3:
                if self.interaction_matrix is not None:
                    density = np.count_nonzero(self.interaction_matrix) / (self.num_users * self.num_items) * 100
                    st.metric("Matrix Density", f"{density:.1f}%")
        else:
            st.warning("⚠️ Model not trained yet. Please train the model first.")

    def run(self):
        st.title("🎯 AutoRec Recommendation System")
        st.markdown("**Upload your data, optimize/train the model, and get personalized recommendations!**")
        st.markdown("---")

        uploaded_file, optimization_settings, user_type = self.render_sidebar()

        if uploaded_file is not None:
            with st.spinner("Processing uploaded data..."):
                df, num_users, num_items = self.process_uploaded_data(uploaded_file)

            if df is not None:
                self.num_users = num_users
                self.num_items = num_items

                st.subheader("🔍 Hyperparameter Configuration")
                training_params = None

                if optimization_settings['use_existing_hpo'] and not optimization_settings['force_new_hpo']:
                    existing_params = self.check_hpo_config()
                    if existing_params:
                        training_params = existing_params
                        st.info("✅ Using existing HPO configuration for training.")

                if training_params is None:
                    if optimization_settings['force_new_hpo'] or st.button("🚀 Run Hyperparameter Optimization", type="primary"):
                        training_params = self.run_hyperparameter_optimization(df, num_users, num_items)
                    elif optimization_settings['manual_params']:
                        training_params = optimization_settings['manual_params']
                        st.info("📝 Using manual parameters for training.")

                if training_params:
                    st.subheader("🤖 Model Training")

                    auto_train_after_hpo = st.checkbox(
                        "Auto-train after HPO",
                        value=True,
                        help="Automatically train model after hyperparameter optimization"
                    )

                    should_train = False
                    if auto_train_after_hpo and not self.trained:
                        should_train = True
                        st.info("🚀 Auto-training model with optimized parameters...")
                    elif st.button("Train Model", type="primary"):
                        should_train = True

                    if should_train:
                        with st.spinner("Training AutoRec model..."):
                            self.model, self.interaction_matrix, train_losses, test_losses, test_rmses = self.train_model(
                                df, num_users, num_items, training_params, optimization_settings['num_epochs']
                            )

                        if self.model is not None:
                            self.trained = True
                            st.success("✅ Model trained successfully!")

                            st.subheader("📈 Training Progress")
                            self.plot_training_curves(train_losses, test_losses, test_rmses)

                            st.session_state.model = self.model
                            st.session_state.interaction_matrix = self.interaction_matrix
                            st.session_state.num_users = self.num_users
                            st.session_state.num_items = self.num_items
                            st.session_state.trained = True
                        else:
                            st.error("❌ Model training failed!")

                    if 'trained' in st.session_state and st.session_state.trained:
                        self.model = st.session_state.model
                        self.interaction_matrix = st.session_state.interaction_matrix
                        self.num_users = st.session_state.num_users
                        self.num_items = st.session_state.num_items
                        self.trained = True
                        st.success("✅ Trained model loaded from session!")

                    if training_params:
                        st.markdown("---")
                        self.show_training_status()

                    if self.trained:
                        st.markdown("---")
                        st.subheader("🎯 Generate Recommendations")
                        user_input = self.render_user_input(user_type)

                        if st.button("Generate Recommendations", type="secondary"):
                            with st.spinner("Generating recommendations..."):
                                recommendations = self.generate_recommendations(user_input, user_type)
                                self.display_recommendations_ui(recommendations, user_type)
                    else:
                        st.info("👆 Please train the model to generate recommendations.")
                else:
                    st.info("🔧 Please configure hyperparameters (run HPO or set manual parameters) before training.")
        else:
            st.info("📁 Please upload a CSV file to get started.")

            st.subheader("📋 Expected CSV Format")
            st.markdown("""
            Your CSV file should have:
            - Rows: Individual users
            - Missing values: Use 'NC', 'NSU', or leave blank
            """)

if __name__ == "__main__":
    app = AutoRecStreamlitUI()
    app.run()
