# MNIST
My analysis of the MNIST data set, implementing ML algorithms to try and correctly classify handwritten digital images.


An ML project from data cleaning and preparation through to model selection and evaluation. Following my reading of Aurelion Geron's 'Hands on Machine Learning with Scikit-Learn and TensorFlow'

To use:
1. run the MNIST.py file to generate a K-Nearest Neighbors Classifier trained on the MNIST dataset. This is a .pkl model file ready to predict handwritten labels.
2. To classify your own handwritten labels run frontend.py, click the open image button and then select the image you want to be classified. Done.
       i) Write a digit 0-9 on a piece of paper in a bold font (just go over it a few times so it looks more filled in, the classifier may work without this step but its much better with it)
       ii) Try and make sure the digit is written on a white peice of paper and that it fills the majority of the image.
       iii) Make sure no other writing/characters are in the image.

Notes: 
1. Building the model the first time round will take its time. The grid_search function takes time to try several different cross-validations on the KNeighbors Classifier with different parameters. Don't be alarmed if this step takes 2 hours or longer. After running it once the models needed for the frontend should be saved into a models folder and won't need to be trained again.
2. The model has a 97.5% accuracy score on test data from the MNIST dataset (set aside which the model hasn't 'seen'). However this doesn't mean user supplied handwritten digits will be predicted with the same accuracy as the data preparation steps for new images isn't exactly the same as the MNIST dataset and the submitted samples won't be recorded in the same way.

To install dependencies run 'pip install -r requirements.txt' in the shell whilst in the project directory (using a virtual environment virtualvenv)

