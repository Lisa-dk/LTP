import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


DATA_PATH = "data"

def load_data(path=DATA_PATH):
    def get_path(set_type, is_args):
        return os.path.join(DATA_PATH, set_type, ("arguments" if is_args else "labels") + "-" + set_type + ".tsv")
    
    data = {}
    
    for set_type in ["test", "training", "validation"]:
        args = pd.read_csv(get_path(set_type, True), sep="\t")
        
        labels = pd.read_csv(get_path(set_type, False), sep="\t")
        labels = labels.drop("Argument ID", axis=1)
        args["Labels"] = [x for x in labels.to_numpy()]
        
        data[set_type] = args
        print(f"Set:{set_type.title()} - Size:{args.shape[0]}")
        
    classes = list(labels.columns)
    
    return data, classes


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute accuracy of predictions"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=True):
    """Compute label-wise and averaged F1-scores"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 2)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'macro-avg-f1score': f1scores['avg-f1-score']}


class MultiLabelTrainer(Trainer):
    """
        A transformers `Trainer` with custom loss computation

        Methods
        -------
        compute_loss(model, inputs, return_outputs=False):
            Overrides loss computation from Trainer class
        """
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def encode_data(data, tokenizer, training="training"):
    def encode(str_input):
        return tokenizer(str_input, truncation=True)

    train_data = data[training]["Premise"]
    train_data = train_data.map(encode)
    for i, x in enumerate(train_data):
        x["labels"] = data[training]["Labels"][i]

    val_data = data["validation"]["Premise"]
    val_data = val_data.map(encode)
    for i, x in enumerate(val_data):
        x["labels"] = data["validation"]["Labels"][i]

    return train_data, val_data


def add_augmented_data(data):
    russian_aug = np.load(os.path.join(DATA_PATH, 'train_aug', 'back_trans_ru_augm.npy'))
    german_aug = np.load(os.path.join(DATA_PATH, 'train_aug', 'back_trans_de_augm.npy'))

    data_counts = data["training"]['Labels'].sum(axis=0)
    mean_freq = np.mean(data_counts)
    min_classes = np.where(data_counts < mean_freq)[0]

    data["training_aug"] = data["training"].copy()

    rows_to_add = []

    for row_idx, row_labels in enumerate(data["training"]['Labels']):
        for flagged_classes in np.where(row_labels==1):
            if np.intersect1d(flagged_classes, min_classes).any():

                new_row = data["training"].iloc[row_idx].copy()
                new_row["Premise"] = german_aug[row_idx] if np.random.random() > 0.5 else russian_aug[row_idx]
                rows_to_add.append(new_row)

    data["training_aug"] = pd.concat([pd.DataFrame(rows_to_add), data["training_aug"]]).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42) 

    print("Loading data.")
    data, classes = load_data()
    print("Data loaded successfully.")

    print("Loading augmented data.")
    add_augmented_data(data)
    print("Augmented data added successfully")

    model_name = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))

    train_data, val_data = encode_data(data, tokenizer, training="training_aug")

    if torch.cuda.is_available():
        model = model.to('cuda')
        print("\nCuda setup was successful.\n")
    else:
        print("\nCould not find cuda!\n")

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='macro-avg-f1score',
        evaluation_strategy="steps",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, classes),
    )

    trainer.train()
    model.save_pretrained("own-" + model_name)