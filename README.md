# ML-Project
 
## CSCI3345 Machine Learning Final Project

We use three models and three datasets to create three TTS models. 

The first model uses a pre-trained BERT model and a custom dataset, which is the whole Bee Movie script read by a roommate for $20.
The second model uses a variational autoencoder and the LJSpeech dataset.
The third model uses transformers with a pretrained LSTM on the VCTK Speech dataset.

# Running Models
Each model can be ran by running it in google colab. Since we do not have access to GPUs on our computers, we did model training and development on google colab. For recording Nick's voice, a mac with custom recording software was also used.

For the VAENAR-TTS model, the training model doesn't need to be ran as the best trained parameters will already be provided. Simply skip over that section and ensure that the path to the pre-trained parameters is valid. You also don't need to mount the Google Drive, as this was used to save model parameters throughout the epochs and track losses. To see the dimensionality of each module, set the variable validation to be True. Since the parameters are too large for them to be uploaded to Github, a link to all of the parameters tested is provided in the google drive.

# Contributions
Ike - Model 1 and custom dataset
Sam - Model 2 and LJSpeech dataset
Luke - Model 3 with VCTK Speech dataset
