import os
import pickle
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Python-based configuration management without YAML files
    """
    
    def __init__(self, config_file: str = "autorec_config.pkl"):
        self.config_file = config_file
        self.default_config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'model_config': {
                'name': 'AutoRec',
                'model_type': 'autoencoder'
            },
            'hyperparameters': {
                'hidden_dim': 512,
                'lr': 0.001,  # Using 'lr' to match existing code
                'weight_decay': 0.0001,
                'batch_size': 64,
                'split': 0.2,  # Using 'split' to match existing code
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
        """Save configuration to pickle file"""
        with open(self.config_file, 'wb') as f:
            pickle.dump(config, f)
        print(f"Configuration saved to {self.config_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from pickle file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Config file not found. Using default configuration.")
            return self.default_config.copy()
    
    def update_from_hpo(self, best_params: Dict[str, Any], best_value: float) -> Dict[str, Any]:
        """Update configuration with HPO results"""
        config = self.load_config()
        
        # Update hyperparameters directly (keys already match)
        config['hyperparameters'].update(best_params)
        
        # Update optimization results
        config['optimization_results'] = {
            'optimized': True,
            'best_value': best_value,
            'best_params': best_params
        }
        
        self.save_config(config)
        return config

# Global instance
config_manager = ConfigManager()

def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return config_manager.load_config()

def save_hpo_results(best_params: Dict[str, Any], best_value: float) -> Dict[str, Any]:
    """Save HPO results to configuration"""
    return config_manager.update_from_hpo(best_params, best_value)
