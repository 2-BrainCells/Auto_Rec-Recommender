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

# Import your modules
from autorec import AutoRec, ARDataset, DataLoader
from preprocessing import split_data, load_data, device
from utils import train_ranking, evaluator, masked_loss, generate_autorec_recommendations, load_and_use_config
from hpo import run as run_hpo
import torch.optim as optim
import torch.nn as nn

# Page configuration
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
    
    @st.cache_data
    def process_uploaded_data(_self, uploaded_file):
        """Process uploaded CSV file"""
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.write("Original data shape:", df.shape)
            st.write("First few rows:")
            st.dataframe(df.head())
            
            # Check if it's the expected format (user-item interactions starting from column 12)
            if df.shape[1] < 13:
                st.error("CSV file must have at least 13 columns (user data + item interactions)")
                return None, 0, 0
            
            # Apply preprocessing similar to your preprocessing.py
            df_items = df.iloc[:, 12:]  # Take columns from 13th onwards
            
            # Replace missing values
            df_items = df_items.replace(['NC', 'NSU', ' '], np.nan)
            df_items = df_items.apply(pd.to_numeric, errors='coerce')
            
            # Filter users with too many missing values
            max_nan_ratio = 0.4
            max_nan_allowed = int(df_items.shape[1] * max_nan_ratio)
            df_items = df_items[df_items.isnull().sum(axis=1) <= max_nan_allowed]
            df_items.reset_index(drop=True, inplace=True)
            
            num_users = len(df_items.index)
            num_items = len(df_items.columns)
            
            # Convert to long format
            df_processed = df_items.stack().rename_axis(('User', 'Item')).reset_index(name='Score')
            
            # Encode items
            item_labels, _ = pd.factorize(df_processed['Item'])
            df_processed['Item'] = item_labels
            
            # Normalize scores
            df_processed['Score'] /= 5.0
            
            st.success(f"✅ Data processed successfully! Users: {num_users}, Items: {num_items}")
            
            return df_processed, num_users, num_items
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None, 0, 0
    
    def check_hpo_config(self):
        """Check if HPO config file exists and load it"""
        if os.path.exists(self.hpo_config_file):
            try:
                config = load_and_use_config(self.hpo_config_file)
                st.success("✅ Found existing HPO configuration!")
                
                # Display the existing config
                with st.expander("📋 Existing HPO Configuration", expanded=False):
                    config_df = pd.DataFrame([
                        {"Parameter": k, "Value": str(v)} 
                        for k, v in config['hyperparameters'].items()
                    ])
                    st.dataframe(config_df, hide_index=True)
                
                return config['hyperparameters']
            except Exception as e:
                st.warning(f"⚠️ Error loading HPO config: {str(e)}")
                return None
        else:
            st.info("ℹ️ No existing HPO configuration found.")
            return None
    
    @st.cache_resource
    def run_hyperparameter_optimization(_self, df, num_users, num_items):
        """Run hyperparameter optimization"""
        try:
            st.info("🔍 Starting hyperparameter optimization...")
            
            # Create a temporary CSV file for the HPO process
            temp_csv_path = "temp_uploaded_data.csv"
            
            # Convert processed df back to the format expected by your preprocessing
            # This is a simplified approach - you might need to adjust based on your exact data format
            df.to_csv(temp_csv_path, index=False)
            
            # Update the preprocessing.py path temporarily (if needed)
            # Or modify the HPO to accept the data directly
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Show HPO progress
            status_text.text("Running hyperparameter optimization... This may take a while.")
            progress_bar.progress(0.5)
            
            # Run HPO
            best_params = run_hpo()
            
            progress_bar.progress(1.0)
            status_text.text("✅ Hyperparameter optimization completed!")
            
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ HPO completed successfully!")
            
            # Display the optimized parameters
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
        """Train the AutoRec model with given parameters"""
        try:
            # Split data
            train_data, test_data = split_data(df, training_params.get('split', 0.2))
            
            # Load interaction matrices
            _, _, _, train_inter_mat = load_data(train_data, num_users, num_items)
            _, _, _, test_inter_mat = load_data(test_data, num_users, num_items)
            
            # Create datasets
            train_dataset = ARDataset(train_inter_mat)
            test_dataset = ARDataset(test_inter_mat)
            
            batch_size = training_params.get('batch_size', 64)
            train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Initialize model
            hidden_dim = training_params.get('hidden_dim', 512)
            model = AutoRec(hidden_dim, num_items).to(device)
            
            lr = training_params.get('lr', 0.001)
            weight_decay = training_params.get('weight_decay', 1e-4)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.MSELoss()
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop with progress updates
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
                
                # Update progress
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_l:.4f}, Test RMSE: {rmse:.4f}")
            
            progress_bar.empty()
            status_text.empty()
            
            return model, test_inter_mat, train_losses, test_losses, test_rmses
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, [], [], []
    
    def plot_training_curves(self, train_losses, test_losses, test_rmses):
        """Plot training curves"""
        if train_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            ax1.plot(train_losses, label='Train Loss', color='blue')
            ax1.plot(test_losses, label='Test Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Test Loss')
            ax1.legend()
            
            # RMSE plot
            ax2.plot(test_rmses, label='Test RMSE', color='green', marker='o')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Test RMSE')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    
    def render_sidebar(self):
        """Render the sidebar with user inputs"""
        st.sidebar.header("🎯 AutoRec Recommender")
        st.sidebar.markdown("---")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with user-item interactions"
        )
        
        if uploaded_file is not None:
            st.sidebar.success("✅ File uploaded successfully!")
            
            # HPO Configuration Section
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
            
            # Training parameters (shown if no HPO or manual override)
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
            
            # User type selection
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
        """Render user input based on type"""
        if user_type == "New User":
            st.subheader("👤 New User Profile")
            st.info("For new users, we'll provide popularity-based recommendations")
            return None
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
                
                # Show user stats if available
                if self.interaction_matrix is not None:
                    user_interactions = np.sum(self.interaction_matrix[user_id] > 0)
                    if user_interactions > 0:
                        avg_rating = np.mean(self.interaction_matrix[user_id][self.interaction_matrix[user_id] > 0])
                        st.info(f"""
                        **User {user_id} Stats:**
                        - Interactions: {user_interactions}
                        - Avg Rating: {avg_rating:.2f}
                        """)
                
                return user_id
            else:
                st.error("No trained model available")
                return 0
    
    def generate_recommendations(self, user_input, user_type):
        """Generate recommendations based on user input"""
        if self.model is None or self.interaction_matrix is None:
            st.error("Model not trained. Please train the model first.")
            return []
        
        try:
            if user_type == "New User":
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
                    user_id=user_input,
                    top_k=10
                )
            
            return recommendations
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def display_recommendations_ui(self, recommendations, user_type):
        """Display recommendations in the UI"""
        if not recommendations:
            st.warning("No recommendations available.")
            return
        
        st.subheader(f"🎯 Top Recommendations for {user_type}")
        
        # Create DataFrame for better display
        rec_df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Item ID': item_id,
                'Predicted Score': f"{score:.4f}",
                'Item Name': f"Learning Aid {item_id}"
            }
            for i, (item_id, score) in enumerate(recommendations)
        ])
        
        # Display as table
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        
        # Visualization
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
    
    def run(self):
        """Main function to run the Streamlit app"""
        # Header
        st.title("🎯 AutoRec Recommendation System")
        st.markdown("**Upload your data, optimize/train the model, and get personalized recommendations!**")
        st.markdown("---")
        
        # Sidebar
        uploaded_file, optimization_settings, user_type = self.render_sidebar()
        
        if uploaded_file is not None:
            # Process data
            with st.spinner("Processing uploaded data..."):
                df, num_users, num_items = self.process_uploaded_data(uploaded_file)
            
            if df is not None:
                self.num_users = num_users
                self.num_items = num_items
                
                # HPO Configuration Logic
                st.subheader("🔍 Hyperparameter Configuration")
                
                training_params = None
                
                # Check for existing HPO config
                if optimization_settings['use_existing_hpo'] and not optimization_settings['force_new_hpo']:
                    existing_params = self.check_hpo_config()
                    if existing_params:
                        training_params = existing_params
                        st.info("✅ Using existing HPO configuration for training.")
                
                # Run new HPO if needed
                if training_params is None:
                    if optimization_settings['force_new_hpo'] or st.button("🚀 Run Hyperparameter Optimization", type="primary"):
                        training_params = self.run_hyperparameter_optimization(df, num_users, num_items)
                    elif optimization_settings['manual_params']:
                        training_params = optimization_settings['manual_params']
                        st.info("📝 Using manual parameters for training.")
                
                # Training section
                if training_params:
                    st.subheader("🤖 Model Training")
                    
                    if st.button("Train Model", type="primary"):
                        with st.spinner("Training AutoRec model..."):
                            self.model, self.interaction_matrix, train_losses, test_losses, test_rmses = self.train_model(
                                df, num_users, num_items, training_params, optimization_settings['num_epochs']
                            )
                        
                        if self.model is not None:
                            self.trained = True
                            st.success("✅ Model trained successfully!")
                            
                            # Plot training curves
                            st.subheader("📈 Training Progress")
                            self.plot_training_curves(train_losses, test_losses, test_rmses)
                            
                            # Store in session state
                            st.session_state.model = self.model
                            st.session_state.interaction_matrix = self.interaction_matrix
                            st.session_state.num_users = self.num_users
                            st.session_state.num_items = self.num_items
                            st.session_state.trained = True
                    
                    # Load from session state if available
                    if 'trained' in st.session_state and st.session_state.trained:
                        self.model = st.session_state.model
                        self.interaction_matrix = st.session_state.interaction_matrix
                        self.num_users = st.session_state.num_users
                        self.num_items = st.session_state.num_items
                        self.trained = True
                        
                        st.success("✅ Trained model loaded from session!")
                    
                    # Recommendations section
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
            
            # Show example format
            st.subheader("📋 Expected CSV Format")
            st.markdown("""
            Your CSV file should have:
            - First 12 columns: User demographic/metadata (optional)
            - Columns 13+: User-item interaction scores (ratings from 1-5)
            - Rows: Individual users
            - Missing values: Use 'NC', 'NSU', or leave blank
            """)

# Run the app
if __name__ == "__main__":
    app = AutoRecStreamlitUI()
    app.run()
