for noise in 2.5 3.2 3.5 4. 5.:
do 
    python experiments/CIFAR10/main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy"
    python experiments/CIFAR10/main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity"
done