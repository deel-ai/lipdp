for noise in 2.5 3.2 3.5 4. 5.:
do 
    python experiments/CIFAR10/main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=30 --cfg.noisify_strategy="global"
    python experiments/CIFAR10/main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=30 --cfg.noisify_strategy="global"
done