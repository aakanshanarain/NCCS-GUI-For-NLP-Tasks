import pickle

import torch
from simpletransformers.classification import (MultiLabelClassificationArgs, MultiLabelClassificationModel)
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

train_df = pd.read_csv('/Users/aakansha/Desktop/NCCS NLP for Histology Reports/Datasets for Trials/train_data_for_model2.csv')
train_df['Combined Diagnosis'] = train_df['Diagnosis'] + train_df['Gross Description'] \
                                 + train_df['Microscopic Description']
train_df = train_df[['Combined Diagnosis', 'Primary Site of Cancer']]

eval_df = pd.read_csv('/Users/aakansha/Desktop/NCCS NLP for Histology Reports/Datasets for Trials/test_data_for_model2.csv')
eval_df['Combined Diagnosis'] = eval_df['Diagnosis'] + eval_df['Gross Description'] \
                                 + eval_df['Microscopic Description']
eval_df = eval_df[['Combined Diagnosis', 'Primary Site of Cancer']]

eval_metrics_df = pd.DataFrame(columns=['Model Type', 'Model Name', 'Epoch', 'Overall Accuracy', 'Overall AUROC Score',
                                        'Hamming Loss', 'Eval Loss'])
# Multi-hot Encoding for Train Data

# Drop NA rows
train_df = train_df.dropna(subset=['Primary Site of Cancer']).reset_index(drop=True)

# Convert datatype of elements to string
train_df['Primary Site of Cancer'] = train_df['Primary Site of Cancer'].astype('str')

# Put primary sites into lists, separate multiple sites for each report
for (i, row) in train_df.iterrows():
    val = train_df['Primary Site of Cancer'].iloc[i]
    list_separated = val.split(",")
    stripped = [s.strip().upper() for s in list_separated]
    set_list = set(stripped)
    train_df.at[i, 'Primary Site of Cancer'] = set_list

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(train_df['Primary Site of Cancer'])
cols = ["%s" % c for c in mlb.classes_]

# Fit data into binarizer, generate multi-hot encodings
df = pd.DataFrame(mlb.fit_transform(train_df['Primary Site of Cancer']), columns=mlb.classes_)

# Merge original text with multi-hot encodings
train_df = pd.concat([train_df[['Combined Diagnosis']], df], axis=1)

print(cols)
# Generate labels columns as list
count = len(cols)
train_df['labels'] = ''

for (i, row) in train_df.iterrows():
    labels = []
    j = 1
    while j <= count:
        labels.append(train_df.iloc[i].iloc[j])
        j += 1
    tup = tuple(labels)
    train_df.at[i, 'labels'] = tup

train_df = train_df[['Combined Diagnosis', 'labels']]

# Multi-hot Encoding for Test Data

# Drop NA rows
eval_df = eval_df.dropna(subset=['Primary Site of Cancer'])

# Convert datatype of elements to string
eval_df['Primary Site of Cancer'] = eval_df['Primary Site of Cancer'].astype('str')

# Put primary sites into lists, separate multiple sites for each report
for (i, row) in eval_df.iterrows():
    val = eval_df['Primary Site of Cancer'].iloc[i]
    list_separated = val.split(",")
    stripped = [s.strip().upper() for s in list_separated]
    set_list = set(stripped)
    eval_df.at[i, 'Primary Site of Cancer'] = set_list

# Fit data into binarizer, generate multi-hot encodings
eval_df_individual_labels = pd.DataFrame(mlb.transform(eval_df['Primary Site of Cancer']), columns=cols)

# Merge original text with multi-hot encodings
eval_df = pd.concat([eval_df[['Combined Diagnosis']], eval_df_individual_labels], axis=1)

# Generate labels columns as list
eval_df['labels'] = ''

for (i, row) in eval_df.iterrows():
    labels = []
    j = 1
    while j <= count:
        labels.append(eval_df.iloc[i].iloc[j])
        j += 1

    tup = tuple(labels)
    eval_df.at[i, 'labels'] = tup

eval_df = eval_df[['Combined Diagnosis', 'labels']]

for n in [2]:

    curr_epoch = "Epoch" + str(n)

    # Configure model args
    model_args = MultiLabelClassificationArgs(num_train_epochs=n)
    model_args.evaluate_during_training_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.learning_rate = 1e-5
    model_args.manual_seed = 4
    model_args.multiprocessing_chunksize = 5000
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.train_batch_size = 16
    model_args.gradient_accumulation_steps = 2
    model_args.use_multiprocessing = True
    model_args.overwrite_output_dir = True
    model_args.labels_list = cols

    # model
    model_type = "roberta"
    model_name = "roberta-large"

    # Create Transformer Model
    model = MultiLabelClassificationModel(model_type, model_name, num_labels=count, use_cuda=False, args=model_args)

    if __name__ == '__main__':

        # Train the model
        model.train_model(train_df)
        pickle.dump(model, open('model2.pkl', 'wb'))

        prediction_df = eval_df['Combined Diagnosis'].values.tolist()

        # Predict output
        prediction, outputs = model.predict(prediction_df)
        outputs_df = pd.DataFrame(outputs, columns=cols)
        prediction_df = pd.DataFrame(prediction, columns=cols)

        # Save outputs to csv file
        filename_prefix = "/Users/aakansha/Desktop/Model2/" + model_name + "_outputs_df"
        filename = "%s.csv" % filename_prefix
        outputs_df.to_csv(filename)

        # Save true and predicted labels to csv file
        combined_cols_df = pd.concat([eval_df, prediction_df], axis=1)
        filename_prefix = "/Users/aakansha/Desktop/Model2/" + model_name + "_combined_cols_df"
        filename = "%s.csv" % filename_prefix
        combined_cols_df.to_csv(filename)

        # Calculate individual label accuracies
        label_accuracy_df = pd.concat([eval_df_individual_labels, prediction_df], axis=1)
        new_acc_cols_order = np.unique(
            np.array(list(zip(eval_df_individual_labels.columns, prediction_df.columns))).flatten())
        label_accuracy_df = label_accuracy_df[new_acc_cols_order]

        count = len(label_accuracy_df.columns)
        i = 0
        colnames = []
        accuracies = []
        auroc = []

        while i < count:
            actualValue = label_accuracy_df.iloc[:, i]
            predictedValue = label_accuracy_df.iloc[:, i + 1]
            actualValue = actualValue.values
            predictedValue = predictedValue.values
            acc = accuracy_score(actualValue, predictedValue)
            # temporary fix, try-except block will be removed in the future with a more balanced dataset
            try:
                auroc_score = roc_auc_score(actualValue, predictedValue)
            except ValueError:
                auroc_score = 0
            colnames.append(label_accuracy_df.columns[i])
            accuracies.append(acc)
            auroc.append(auroc_score)
            i += 2

        accuracy_auroc_df = pd.DataFrame(list(zip(colnames, accuracies, auroc)),
                                         columns=['Site', 'Accuracy', 'AUROC Score'])

        # Evaluate model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df)

        # Processing for metrics calculation
        eval_df_multi_hot_encodings = []

        for (i, row) in eval_df.iterrows():
            val = eval_df['labels'].iloc[i]
            val_array = np.asarray(val)
            eval_df_multi_hot_encodings.append(val_array)

        eval_df_true = np.array(eval_df_multi_hot_encodings)
        prediction_data = np.array(prediction)

        # Calculate metrics
        overall_acc = accuracy_score(eval_df_true, prediction_data)
        # temporary fix, try-except block will be removed in the future with a more balanced dataset
        try:
            overall_auroc = roc_auc_score(eval_df_true, prediction_data)
        except:
            overall_auroc = 0
        hamming_loss = hamming_loss(eval_df_true, prediction_data)

        target_names = cols
        other_metrics_report = classification_report(eval_df_true, prediction_data, target_names=target_names,
                                                     output_dict=True)
        classification_report_df = pd.DataFrame(other_metrics_report).transpose()

        # Combine individual accuracies to classification report
        classification_report_df = classification_report_df.reset_index()
        accuracy_auroc_df = accuracy_auroc_df.reset_index()
        classification_report_df_final = pd.concat([classification_report_df, accuracy_auroc_df['Accuracy'],
                                                    accuracy_auroc_df['AUROC Score']], axis=1)

        # Save classification report to csv file
        filename_prefix = "/Users/aakansha/Desktop/Model2/" + model_name + "_classification_metrics_df"
        filename = "%s.csv" % filename_prefix
        classification_report_df_final.to_csv(filename)

        # Save other metrics to csv file
        metrics_data = [model_type, model_name, curr_epoch, overall_acc, overall_auroc, hamming_loss,
                        result['eval_loss']]
        df_length = len(eval_metrics_df)
        eval_metrics_df.loc[df_length] = metrics_data
        filename_prefix = "/Users/aakansha/Desktop/Model2/" + model_name + "_eval_metrics_df"
        filename = "%s.csv" % filename_prefix
        eval_metrics_df.to_csv(filename)