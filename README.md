# Deep-Learning-EEG
My first deep learning project, creating a model that identifies hand movement (6 different kind of movements) from eeg data.


Once the eeg data were collected, the first major step was to apply a low pass filter to remove the noise from the eeg data.
After that, what follows is the Standardization of the features in the dataset and finally the application of PCA (Principal compoment Analysis) for dimensionality reduction.
After this 3 steps the data preperation is complete. The next step is the creation of the model. The model that i used is a CNN model. After 8 hours of training the model was able
to produce a 0.78~ AUC score.  

![finalRes](https://github.com/JkDev17/Deep-Learning-EEG/assets/58857454/31c84c47-076e-46f9-81b9-91f4ec2d5579)

