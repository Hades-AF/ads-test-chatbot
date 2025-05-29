# ADS Data Science Assessment - Chatbot

## Project Overview
The dataset provided is SQuAD v1.1, which contains paragraphs with associated questions and answers. It can be found [here](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset). Your task is to build an extractive question-answering model, which, given a paragraph and a question, returns the span of text in the paragraph that best answers the question.

You may use a pre-trained model (e.g., BERT, DistilBERT) and fine-tune it on this dataset.

The final script (main.py) should:

* Accept a user question in interactive mode.
* Select a context paragraph (either randomly or based on keyword search).
* Use your trained model to return the most likely answer from the paragraph.
* Display the predicted answer and its confidence score.

Additionally, provide a simple post writeup to describe how the AI model was derived. At a minimum it should include:
* The type of model chosen and why
* How the structure of the model layers were derived (if applicable)
* How the dataset was split into training and testing
* Possible improvements or further things to test
* How you prevented over fitting (if applicable)

## Extra Credit
For additional credit:
* Utilize the AI model in a JVM based language, such as Java or Kotlin
* Create a simple GUI for interacting with the AI model
* Persist the user's input and the model output when interacting with the model. 
Can be persisted to a SQLite database (preferred), or a JSON file. Feel free to add extra metadata that could be useful later.
