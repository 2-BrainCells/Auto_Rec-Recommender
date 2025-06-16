import os
import pickle
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Python-based configuration management system using pickle files.
    Handles storage and retrieval of model hyperparameters and optimization results.
    """
    
    def __init__(self, config_file: str = "autorec_config.pkl"):
        """
        Initialize configuration manager with specified config file path.
        
        Args:
            config_file: Path to the configuration pickle file
        """
        self.config_file = config_file
        self.default_config = self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration dictionary with standard AutoRec parameters.
        
        Returns:
            Dictionary containing default model and training configurations
        """
        return {
            'model_config': {
                'name': 'AutoRec',
                'model_type': 'autoencoder',
                'use_demographics': False
            },
            'hyperparameters': {
                'hidden_dim': 512,
                'lr': 0.001,
                'weight_decay': 0.0001,
                'batch_size': 64,
                'split': 0.2,
                'dropout': 0.2
            },
            'training': {
                'num_epochs': 50,
                'device': 'cuda',
                'seed': 42
            },
            'optimization_results': {
                'optimized': False,
                'best_value': None
            }
        }

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration dictionary to pickle file.
        
        Args:
            config: Configuration dictionary to save
        """
        with open(self.config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"Configuration saved to {self.config_file}")

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from pickle file or return default if file doesn't exist.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Config file not found. Using default configuration.")
            return self.default_config.copy()

    def update_from_hpo(self, best_params: Dict[str, Any], best_value: float) -> Dict[str, Any]:
        """
        Update configuration with hyperparameter optimization results.
        
        Args:
            best_params: Dictionary of optimized hyperparameters
            best_value: Best objective value achieved during optimization
            
        Returns:
            Updated configuration dictionary
        """
        config = self.load_config()
        config['hyperparameters'].update(best_params)
        config['optimization_results'] = {
            'optimized': True,
            'best_value': best_value,
            'best_params': best_params
        }
        self.save_config(config)
        return config

config_manager = ConfigManager()

def get_config() -> Dict[str, Any]:
    """
    Get current configuration from the global config manager.
    
    Returns:
        Current configuration dictionary
    """
    return config_manager.load_config()

def save_hpo_results(best_params: Dict[str, Any], best_value: float) -> Dict[str, Any]:
    """
    Save hyperparameter optimization results to configuration.
    
    Args:
        best_params: Best hyperparameters found during optimization
        best_value: Best objective value achieved
        
    Returns:
        Updated configuration dictionary
    """
    return config_manager.update_from_hpo(best_params, best_value)
