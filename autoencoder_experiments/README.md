# Experiments with autoencoders

**Abbreviation:**
- SpAE - sparse autoencoder
- SpDAE - deep sparse autoencoder
- StAE - stacking autoencoder
- StDAE - deep stacking autoencod

**Notes:**
- Positive (label=1) - healthy
- Negative (label=0) - sick

**Base structure for experiment:**
```bash
.
├── logs         # Directory with logging
│   └── ...
├── net          # Directory with implementation network
│   └── ...
├── tensorboard  # Directory with tensorboard writer
│   └── ...
├── train.py     # Code for train network
├── config.py    # Global variables
├── README.md    # Description experiments and results
└── ...          # Specific dirs or files for experiment
```

## Experiments

- [SpAE_001](./SpAE_001) - 