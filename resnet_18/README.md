# Run scripts
Use the following script to train the Neurotransmitter(NT) Classification model (ResNet-18)

<hr>

### Create train and test split
Use this script to create a train test split from the original ground-truth.

`python3 01_split_data.py`

<hr>

### Train NT model
Use this script to train the CNN model using the training data generated.

`python3 02_train.py`

<hr>

### Evaluate model on held-out test set
Use this script to evaluate the model on the held-out test set.

`python3 03_inference_val.py`

<hr>

### Evaluate model on training set (for debugging purposes)
Use this script to evaluate the model on the training set. For debugging purposes.

`python3 03_inference.py`