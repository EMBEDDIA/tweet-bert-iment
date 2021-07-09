from datasets import load_dataset, load_metric
import transformers
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", default=None, type=str, required=True)
parser.add_argument("--bertsent", default=None, type=str, required=True, help="sentiment-finetuned bert model name")
parser.add_argument('--fast_tokenizer', action='store_true', help='use fast tokenizer')
parser.add_argument("--bs", default=16, type=int, help="batch size")
parser.add_argument("--output", default=None, type=str, help="Where save eval results")
args = parser.parse_args()


batch_size=args.bs

# LOAD DATA
dataset = load_dataset('csv', data_files={'test': args.eval_file}, column_names=['sentence', 'label'])
metric = load_metric('glue', 'sst2')

# PREPROCESS (TOKENIZE) DATA
tokenizer = AutoTokenizer.from_pretrained(args.bertsent, use_fast=args.fast_tokenizer)
label2id = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
def preprocess_function(examples):
    result = tokenizer(examples['sentence'], truncation=True, max_length=512)
    result["label"] = [label2id[l] for l in examples["label"]]
    return result

encoded_dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

# LOADING MODEL
model = AutoModelForSequenceClassification.from_pretrained(args.bertsent, num_labels=3)
trainargs = TrainingArguments(
    "tweet-sentiment",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
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
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=None
    )

# RUNNING EVALUATION
eval_results = trainer.evaluate()
if args.output:
    with open(args.output, 'a') as writer:
        writer.write('\t'.join([args.eval_file, args.bertsent, str(eval_results['eval_accuracy'])])+'\n')
