class PARAM:
    seed = 42  # Random seed
    #preset = "deberta_v3_extra_small_en" # Name of pretrained models
    #sequence_length = 512  # Input sequence length
    #epochs = 3 # Training epochs
    #batch_size = 16  # Batch size
    #scheduler = 'cosine'  # Learning rate scheduler
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())