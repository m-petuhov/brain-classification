class Config:

    routes = {
        'data_dir': './train_val_data',     # [Required]
        'save_dir': './weights',            # [Required]
        'log_dir': './logs',                # [Required]
        'tensorboard_dir': './tensorboard', # [Required]
    }

    train_params = {
        'arch': 'SpAE_001',

        'with_mask': True,
        'batch_size': 19,
        'start_epoch': 1,
        'epochs': 70,
        'weights': None,
        'num_workers': 1
    }

    env_params = {
        'random_seed': 1  # [Optional - None]
    }
