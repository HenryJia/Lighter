# Lighter
A high level library of deep learning tools built on top of PyTorch inspired by PyTorch's Ignite  

This branch of Lighter was developed specifically for the CS350 Datascience project  

In this branch, the RL algorithms specifically are located in lighter/train/steps/  
Each python file inside each implements a single optimisation step of an RL algorithm. They are then iterated and optimised primarily using lighter/train/trainer/rl\_trainer.py  

Other parts of this library of code are from before CS350, though much of it was overhauled and refactored during the development of the CS350 project as it was outdated.  

The commit history and git diff between this branch (dev-cs350) and the master branch of Lighter can be used if you wished to distinguish between what was added for the CS350 project and what existed before in Lighter.  
If you are too unfamiliar with git to do this on your system, Lighter is public on GitHub under the MIT License and you can see the git diff at https://github.com/HenryJia/Lighter/compare/dev-cs350
