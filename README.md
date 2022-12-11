Deep learning fall 2022 final project by group DL24.

Here you will find all the relevant code we have made for our final project as well as a very small subset of our data. All of our data is on the GPU cluster and thus not easily downloaded. The data that is included here is only to show that the code runs properly.

In the folder "code" you will find all the code we have written for this project. Files:
- model.py: This file contains our model as well as a unit test to make sure the model works. This is also where you will find our loss function.
- training.py: This is where you will find our training function, training loop and all the other requirements for training.
- data_treatment.py: This file was used to generate our noisy images. Different levels of noise is added to each image and then saved in the relevant folder along with the mask we create.
- interactive.py: This file lets you play around with our model to test the performance and limitations yourself. Choose an image, draw on it and see what the model outputs. An example of how to use the model can be seen here: https://imgur.com/a/m5IPwt7

In the folder "data" we have 3 subfolders. One contains images used for the unit test (unittest), one contains images used for the interactive.py file (interactive) and the last contains the very small subset of our data as already explained.

Next we have a folder called "models" where we have saved the models we have trained during the project. This includes the best model achieved for each noise level, the pretrained model on whole images, an overfit model and finally a model trained with a low coefficient on neighbourhood loss (measuring difference from neighbouring pixels). This last model is called "low diff".

Finally we have a folder with plots showing losses during training.
