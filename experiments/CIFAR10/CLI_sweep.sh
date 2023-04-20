for noise in 3.2 3.5 4. 5.:
do 
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=10  --cfg.architecture="ConvNet"
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=10  --cfg.architecture="ConvNet"
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHinge" --cfg.opt_iterations=10  --cfg.architecture="ConvNet"
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassKR" --cfg.opt_iterations=10 --cfg.architecture="ConvNet"
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MulticlassHKR" --cfg.opt_iterations=10  --cfg.architecture="ConvNet"
    python main_Architectures.py --cfg.log_wandb="sweep_archi" --cfg.noise_multiplier=$noise --cfg.loss="MAE" --cfg.opt_iterations=10  --cfg.architecture="ConvNet"
done