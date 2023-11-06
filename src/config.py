config = {
    "run_name": "bart-base-finetune",
    "bos_token": "<s>",
    "strategy_token": "<strg>",
    "ref_token": "<refr>",

    "epochs": 10,
    "batch_size": 32,
    "init_lr": 5e-5,
    "num_beams": 2, # only for evaluation (generate func.)

    "wandb": True,
    "pretrained_path": None,
    "save_path": "/home/haeji/NLP/ckpt" # leaf dir name will be run_name
    
}