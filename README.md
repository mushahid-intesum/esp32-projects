## Bird Call Identifier on ESP32

This project involves building a tiny bird call identifiers on multiple micro-controllers. 

# Dataset
The [BirdClef 2023](https://www.kaggle.com/competitions/birdclef-2023) data was used to train the models.

The dataset can be prepared and split by running the `make_dataset.ipynb` script. Install the following packages to run the mentioned script

`
pip install pandas matplotlib librosa tqdm numpy
`


# Setup
**Model Training**
Data was trained using the tensorflow library, available in the `train_tf.ipynb` python notebook. Run the following command to install the relevant packages

`
pip install tensorflow ai-edge-litert tensorflow-model-optimization
`

**Deployment**
After training, the model

**Hardware Setup**


# Hardware
The following hardware were used in the project
- 2x ESP32S3 Board
- A PC with GPU for weight updates
