# DeepEngage
<p align = "center">
<img src="https://github.com/wdlcameron/DeepEngage/blob/master/src-images/logo.png" alt "" width="144" height="144"> 
</p>

DeepEngage is a engagement prediction tool for posts within specific instagram niches.  An implementation for food-related posts can be found [here](http://www.deeplearningsolutions.site)

The final product would accept a group of images (simulated through preset groups in the demo) and the sponsored content creator's username and will select the post with the greatest engagement potential.  The number represents how much engagement it is forecasted to receive relative to the average (e.g. a score of 1.2 is predicted to get 1200 likes if the average for that user is 1000).  The "Recommended Optimizations" pane applies a series of filters to the chosen image and recommends changes if they are expected to  

# Core Requirements and Installation Instructions
The core packages required are:
FastAI V1  (conda install -c fastai fastai)  
ImageIO (conda install -c conda-forge imageio)  
Selenium (conda install -c conda-forge selenium)  

Note: there is currently an incompatibility with Pillow and PyTorch, which requires you to downgrade Pillow (conda install pillow=6.1)

Selenium also requires that the path have access to chromedriver.exe, which can be downloaded from [here](https://chromedriver.chromium.org/downloads)


# Web Scraping
Models are trained using images and tabular data scraped from Instagram.  This process is outlined in the "Web Scraping Pipeline.ipynb" notebook in the main directory


# Model Training

