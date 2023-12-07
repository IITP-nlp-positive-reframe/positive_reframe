# BART

This directory contains codes that reimplemented the paper _Inducing Positive Perspectives with Text Reframing_.

### Dataset
Put your datasets under `./data`, as in this repository.

### Setting Configuration
You can set the configuration at `src/config.py`, no matter whether you are trying to train the model or test it.
- `"run_name"`: It is the leaf directory's name for the checkpoint, and also the name that is used for wandb logging.
- `"strategy_token"`, `"ref_token"`: These are in fact not used, since our experiment only deals with unconstrained setting from the original paper.
- `"wandb"`: Whether you want to turn on wandb logging or not.
- `"pretrained_path"`: If you want to resume training from a certain checkpoint or if you are trying to run evaluation code with a trained model, you should put the pretrained model's path here. The code brings the model with `from_pretrained` method from **huggingface transformers**.
- `"save_path"`: A path where all the checkpoints are being stored. The model will be saved under this directory, with its `"run_name"`.

### How to run Training
To train the model, you can adjust your configuration at `src/config.py`, and run
```
python src/train.py
```

### How to run Evaluation
To get a full set of evaluation scores, set the path for your trained model at `src/config.py` and run
```
python src/evaluation.py
```
It gives you BLEU score, Rouge scores, BERT score, and the difference in Text blob sentiment scores, as in the original paper.

### Example outputs
Below are some sample outputs of trained models.
For full outputs and evaluation scores, see `./test_eval_e{1|5}.txt`.
- after **1 epoch**
```
input: Always stressing and thinking about loads of things at once need I take it one at a time overload stressed need to rant
pred: I am stressed and thinking about loads of things at once, but I am going to take it one at a time, and then I will be able to relax.
gt: Loads of things on my mind, I need to make a list, prioritise and work through it all calmly and I will feel much better.

input: The list of things I have to do today is forever long. Can I just go back to bed and wake up tomorrow with everything done?
pred: The list of things I have to do today is forever long, but I'm sure I can do it.
gt: Today I have a lot to do. Time for productivity not procrastination.

input: If I don't have a mental breakdown before the end of summer, it will be a miracle..
pred: If I don't have a mental breakdown before the end of summer, it will be a miracle.
gt: I'm going to look after my mental health over the summer, so I don't have a mental breakdown.

input: I just caught myself taking extra deep breaths as if it would somehow help Hallie breathe.
pred: I just caught myself taking extra deep breaths as if it would somehow help Hallie breathe.
gt: Trying to keep breathing. Extra deep
```

- after **5 epochs**
```
input: Always stressing and thinking about loads of things at once need I take it one at a time overload stressed need to rant
pred: I am stressed and thinking about loads of things at once, but I am sure I can get through it.
gt: Loads of things on my mind, I need to make a list, prioritise and work through it all calmly and I will feel much better.

input: The list of things I have to do today is forever long. Can I just go back to bed and wake up tomorrow with everything done?
pred: I have a lot of things to do today, but I'm sure I'll be able to finish it.
gt: Today I have a lot to do. Time for productivity not procrastination.

input: If I don't have a mental breakdown before the end of summer, it will be a miracle..
pred: I hope I don't have a mental breakdown before the end of summer, it will be a miracle.
gt: I'm going to look after my mental health over the summer, so I don't have a mental breakdown.

input: I just caught myself taking extra deep breaths as if it would somehow help Hallie breathe.
pred: I just caught myself taking extra deep breaths as if it would somehow help Hallie breathe. But I'm sure it won't.
gt: Trying to keep breathing. Extra deep
```
