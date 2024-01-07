# NLP_Praca_Domowa_-towska
Homework with NLP 
# 1.Dataset
The SMS Spam Collection v.1 is a set of SMS messages that have been collected and labeled as either spam or not spam. This dataset contains 5574 English, real, and non-encoded messages. The SMS messages are thought-provoking and eye-catching. There are 2 columns: sms (the text of the SMS message (String)) and label (the label for the SMS message, indicating whether it is ham (0) or spam (1). (String)). There is 5171 unique values.
Link to dataset: https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset

# 2. Model description Roberta-spam Model Fine tunning

This script demonstrates the entire pipeline of fine-tuning a RoBERTa model for spam classification using the Hugging Face Transformers library and the Datasets library. It includes data preparation, fine-tuning configuration, training configuration, trainer initialization, training and testing. Link to model: https://huggingface.co/mshenoda/roberta-spam

* 1. Data Preparation:
In the first step, the previously cleaned SMS data was loaded from a CSV file using the pandas library. The data was transformed into a Dataset object using the datasets library from Hugging Face. The dataset was then split into a training set and a test set, and the split was 10%.

* 2. Fine-tuning Configuration:
The fine-tuning parameters were defined in the code as follows:

model_checkpoint = 'mshenoda/roberta-spam': Selection of the pre-trained RoBERT model for fine-tuning.
batch_size = 16: The size of the batch used in training.
Tokenization of the SMS text was done using the AutoTokenizer class from the transformers library. A transformer function was defined that performs the tokenization of the text using a previously initialized tokenizer.

* 3. Training Configuration:
The RoBERT model was then initialized using the AutoModelForSequenceClassification class, where the number of labels (in this case 2 for binary classification) was passed as an argument. The training parameters were configured using the TrainingArguments class. The key parameters are:

evaluation_strategy='epoch': Evaluation strategy after each epoch.
save_strategy='epoch': Save the model after each epoch.
learning_rate=2e-5: The learning rate.
per_device_train_batch_size=batch_size: Size of the training batch.
per_device_eval_batch_size=batch_size: Size of the test batch.
num_train_epochs=5: Number of training epochs.
weight_decay=0.01: Weight reduction factor.
load_best_model_at_end=True: Loads the best model at the end of training.
metric_for_best_model='accuracy': Metric used to select the best model.

* 4. Trainer initialization:
An object of the Trainer class has been created, using the initialized model, training arguments, training set, test set, tokenizer and a function that calculates metrics (in this case, accuracy).

* 5. Training:
After setting up the model and training parameters, model evaluation was performed on a small subset of the training data using the trainer.evaluate() method. Then training of the model was run using the trainer.train() method.


* 6. Testing the Model:

Finally, the trained model was tested by giving it a sample SMS text.The text was tokenized, converted to PyTorch tensors, and then the model was inferred to determine whether the text was correctly classified as spam or not.

This process resulted in a model capable of correctly classifying SMS as spam or not.



# 3. Instructions
There are 2 jupyter files on the repository. In 1 there is presentation of data distribution, data cleaning, built model using LSTM recurrent network, built model using CNN network and model with pre-trained word embeddings. For all of them, too, functions have been created that use the trained model and return a result for the passed single sentence(s) in the form of a message telling the user whether the text is spam or not.

The second file contains the described model with a fine-tuning of the Robert-spam language model also with a function that uses the trained model and returns the result for the passed single sentence(s) in the form of a message informing the user whether the text is spam or not.

Please fire up the jupiter notebook with 3 models first, as it contains Exploratory Data Analysis and Data Preparation. 




