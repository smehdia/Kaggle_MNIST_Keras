# Kaggle_MNIST_Keras
API Model in Keras for Kaggle MNIST competition 

How to run:
 1) Download train.csv and test.csv from Kaggle:
    (https://www.kaggle.com/c/digit-recognizer/data)
    
 2) Move train.csv and test.csv in data directory

    
 3) Run prepare_dataset.py in order to create the dataset
 
 4) Run train.py to train the model
     if you want to see loss plot enter below command in your terminal:
      tensorboard --logdir logs
    click on the server address to see plots online in your web browser
    
 5) Run evaluate_model_on_test_data.py in order to create submission.csv which you could use to upload it in Kaggle
 
 
 *** TRAINING AND VALIDATION ACCURACIES ARE ABOVE 99%
 
-------------------------------------------------------------------------------------------------------------------
Model Graph:

![Alt text](./model.png "Model Architecture")


First Layer Filters Before Training the Model:

![Alt text](./first_layer_filters_before_training.png "Kernels Initialization")

First Layer Filters After Training the Model:

![Alt text](./first_layer_filters.png "Kernels After Training")

Losses and Accuracies: (Blue ---> Validation, Orange ---> Training)

![Alt text](./tensorboard_plots.png "Tensorboard Sample Plots")

