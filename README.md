# Insight-Project
My project for Insight DS 2020A

# DeepEngage
DeepEngage is a engagement prediction tool for posts within specific instagram niches.  An implementation for food-related posts can be found [here](deeplearningsolutions.site:5000)

# Core Requirements and Installation Instructions
The core packages required are:
FastAI V1  (conda install -c fastai fastai)
ImageIO (conda install -c conda-forge imageio)
Selenium (conda install -c conda-forge selenium)

Note: there is currently an incompatibility with Pillow and PyTorch, which requires you to downgrade Pillow (conda install pillow=6.1)

Selenium also requires that the path have access to chromedriver.exe, which can be downloaded from [here](https://chromedriver.chromium.org/downloads)
