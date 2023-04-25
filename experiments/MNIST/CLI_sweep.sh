for noise in 20.0 17.5 15.5 15.0 14.5 13.0 12.5 11.5 10.0 9.0 8.0 7.5 7.0 6.5 6.0 5.0 4.5 4.0 3.5 3.0 2.5 2.0 1.5 1.0:
do 
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="local"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="local"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="local"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="local"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="local"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="local"
done
for noise in 20.0 17.5 15.5 15.0 14.5 13.0 12.5 11.5 10.0 9.0 8.0 7.5 7.0 6.5 6.0 5.0 4.5 4.0 3.5 3.0 2.5 2.0 1.5 1.0:
do 
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="Dense" --cfg.noisify_strategy="global"
    python main_template.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=50  --cfg.architecture="ConvNet" --cfg.noisify_strategy="global"
done