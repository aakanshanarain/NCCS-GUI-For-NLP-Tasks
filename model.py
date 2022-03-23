from simpletransformers.classification import (ClassificationArgs, ClassificationModel)
import pandas as pd
import pickle

# ### Read data from files

# In[ ]:


train_df = pd.read_csv('/Users/aakansha/Desktop/NCCS NLP for Histology Reports/Datasets for Trials/train.csv')
train_df['Combined Diagnosis'] = train_df['Diagnosis'] + train_df['Gross Description'] + train_df[
    'Microscgopic Description']
train_df = train_df[['Combined Diagnosis', 'Cancerous?']]

# ### Pre-process data

# In[ ]:


# Capitalize values
train_df['Cancerous?'] = train_df['Cancerous?'].str.upper()

train_df['Cancerous?'] = train_df['Cancerous?'].str.strip()

# Drop all NA rows
train_df = train_df.dropna(subset=['Cancerous?']).reset_index(drop=True)

# ## Roberta-Large

# In[ ]:

for n in [1]:

    curr_epoch = "Epoch" + str(n)

    # Configure model args
    model_args = ClassificationArgs(num_train_epochs=n)
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
    model_args.labels_list = ['YES', 'NO']

    # model
    model_type = "roberta"
    model_name = "roberta-large"

    # Create Transformer Model
    model = ClassificationModel(model_type, model_name, num_labels=2, use_cuda=False, args=model_args)

    if __name__ == '__main__':
        # Train the model
        model.train_model(train_df)
        pickle.dump(model, open('model.pkl', 'wb'))


