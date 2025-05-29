from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer, pipeline,
)
from datasets import load_dataset

model_name = "distilbert-base-uncased"
model_dir = "./model_output"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

squad = load_dataset("squad")


def process_answer_tokens(record):
    inputs = tokenizer(
        record["question"],
        record["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
        return_token_type_ids=True
    )

    answer = record["answers"]["text"][0]
    answer_start_index = record["answers"]["answer_start"][0]
    answer_end_index = answer_start_index + len(answer)

    offsets = inputs["offset_mapping"]
    token_types = inputs["token_type_ids"]

    start_token = end_token = 0
    for i in range(len(offsets)):
        start, end = offsets[i]
        type_id = token_types[i]

        if type_id != 1:
            continue

        if start <= answer_start_index < end:
            start_token = i
        if start < answer_end_index <= end:
            end_token = i
            break

    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    inputs.pop("offset_mapping")
    return inputs


training = squad["train"].select(range(200)).map(process_answer_tokens)
validation = squad["validation"].select(range(50)).map(process_answer_tokens)

training_arguments = TrainingArguments(
    output_dir=model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=training,
    eval_dataset=validation,
)

trainer.train()
pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


def describe_model():
    """
    Describe the model type, shape, layers, etc.
    """
    # raise NotImplementedError()


def report_model_performance():
    """
    Report the model's performance on the training and test datasets.
    """
    # raise NotImplementedError()


def interact_with_model():
    """
    Accept user input from the console, in the form of a string, and output the model's predictions and its confidence score.
    Any extra data that you want to output, please do so.
    """
    # raise NotImplementedError()


if __name__ == '__main__':
    describe_model()
    report_model_performance()
    interact_with_model()
