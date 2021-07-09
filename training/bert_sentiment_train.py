from datasets import load_dataset, load_metric
import transformers
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default=None, type=str, required=True)
parser.add_argument("--eval_file", default=None, type=str, required=True)
parser.add_argument("--bert", default=None, type=str, required=True, help="bert model name")
parser.add_argument('--fast_tokenizer', action='store_true', help='use fast tokenizer')
parser.add_argument("--bs", default=16, type=int, help="batch size")
parser.add_argument("--epoch", default=3, type=float, help="number of epochs to train for")
parser.add_argument("--output", default=None, type=str, help="Folder to save the sentiment analysis model to")
args = parser.parse_args()


batch_size=args.bs

# LOAD DATA
dataset = load_dataset('csv', data_files={'train': args.train_file, 'test': args.eval_file}, column_names=['sentence', 'label'])
metric = load_metric('glue', 'sst2')

# PREPROCESS (TOKENIZE) DATA
tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=args.fast_tokenizer)
#tokenizer.enable_padding(length=512)
label2id = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
def preprocess_function(examples):
    result = tokenizer(examples['sentence'], truncation=True, max_length=512)
    result["label"] = [label2id[l] for l in examples["label"]]
    return result

encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

# TRAINING (finetuning) MODEL
model = AutoModelForSequenceClassification.from_pretrained(args.bert, num_labels=3)
trainargs = TrainingArguments(
    "tweet-sentiment",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epoch,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    trainargs,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=None
    )

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

# SAVE MODEL
if args.output:
    trainer.save_model(output_dir=args.output)
    with open(args.output+'/eval_results.txt', 'a') as writer:
        writer.write('\t'.join([args.eval_file, args.bert, str(eval_results['eval_accuracy'])])+'\n')
