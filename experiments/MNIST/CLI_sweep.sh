for noise in 17.5 14.5 11.5 9.0 7.0 6.0:
do 
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
done