class Config:

    routes = {
        'data_dir': 'train_val_data',  # [Required]
        'save_folder': 'weights',  # [Required]
        'log_dir': 'logs',         # [Required]
    }

    env_params = {
        'random_seed': 1             # [Optional - None]
    }


cfg = Config()
