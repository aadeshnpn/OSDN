# OSDN
Keras implementation for the research paper "Towards Open Set Deep Networks" A Bendale, T Boult, CVPR 2016

Original Implementation: https://github.com/abhijitbendale/OSDN

This repo has Keras wrapper for the above research paper. Full code plus ipython notebook is also avaliable.

```
jupyter notebook Softmax.ipynb
```
or

open notebook with nbviewer by clicking on this link

https://nbviewer.jupyter.org/github/aadeshnpn/OSDN/blob/master/Softmax.ipynb

If you have any question feel free to create an issue.

## How to run the existing code
* Step 1: Train a CNN model for the dataset you choice
* Step 2: Load the trained model
* Step 3: Load the training data you trained the DNN model
* Step 4: Create a mean activation vector (MAV) and perform weibull fit model
* Step 5: Pass the sample to compute openmax and evaluate the output from openmax, original label, and softmax
* Step 6: Test the trained openmax to images from different distribution

## Refer to the `main.py` for detail implementation