for epsilon in 7.53 3.0 :
do 
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=25 --cfg.optimizer="Adam"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="TauCategoricalCrossentropy" --cfg.opt_iterations=25 --cfg.optimizer="SGD"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=25 --cfg.optimizer="Adam"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="KCosineSimilarity" --cfg.opt_iterations=25 --cfg.optimizer="SGD"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassHinge" --cfg.opt_iterations=25 --cfg.optimizer="Adam"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassHinge" --cfg.opt_iterations=25 --cfg.optimizer="SGD"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassKR" --cfg.opt_iterations=25 --cfg.optimizer="Adam"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassKR" --cfg.opt_iterations=25 --cfg.optimizer="SGD"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassHKR" --cfg.opt_iterations=25 --cfg.optimizer="SGD"
    python main.py --cfg.log_wandb="sweep_losses" --cfg.epsilon=$epsilon --cfg.loss="MulticlassHKR" --cfg.opt_iterations=25 --cfg.optimizer="Adam"
done