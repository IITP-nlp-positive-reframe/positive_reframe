config = {
    "run_name": "bart-large-debug",
    "bos_token": "<s>",
    "strategy_token": "<strg>",
    "ref_token": "<refr>",
    "device": 'cuda:0',

    "epochs": 40,
    "batch_size": 32,
    "init_lr": 1e-4,
    "num_beams": 2, # only for evaluation (generate func.)

    "wandb": True,
    "pretrained_path": "/workspace/positive_reframe/ckpt/bart-large-debug/best", # "/workspace/positive_reframe/ckpt/bart-large-finetune-3/best",
    "save_path": "/workspace/positive_reframe/ckpt" # leaf dir name will be run_name
    
}
