# CNN for Aire's target choices

To investigate Aire’s target choices, we built and trained a CNN model with dilated convolutional layers and residual skip connections to distinguish the extended-promoter sequences of Aire-induced and Aire-neutral genes. We followed the pre-training and then fine-tuning paradigm, given the relatively small number of Aire-induced and Aire-neutral genes compared with the typical training sizes employed for large-scale models. Below detailed the required coding environment and script usage.

## Pre-training
### Environment setup
Code for building and pre-training the CNN was developed using Python 3.6 and a variety of well-developed packages. To prepare for the pre-training environment using Anaconda, run the following:
```
conda env create -f pretrain.yml
```
### Scripts
All the utility functions are included in the script `model_main.py`.

`pretrain_data.py` is used to modify the datasets of Basenji2[^1] for the pre-training purpose. Here is an example of the usage:
```
python pretrain_data.py ${data_path}

## Mandatory input:
  ${data_path}: The directory containing the original datasets of Basenji2.
```

`run_pretrain.py` is used to pre-train the CNN model. Example usage:
```
python run_pretrain.py ${tfr_path} ${loss} ${weight}

## Mandatory input:
  ${tfr_path}: The directory containing the modified datasets of Basenji2 for pre-training.
  ${loss}: The id of the loss used for pre-training: 
    ${loss} == 1: tf.keras.losses.Poisson()
    ${loss} == 2: MSEMultinomialPretrain(total_weight = ${weight})
    ${loss} == 3: PoissonMultinomialPretrain(total_weight = ${weight})
  ${weight}: An argument for the loss function.
```
## Fine-tuning
### Environment setup
Code for fine-tuning the CNN was also developed using Python 3.6. To prepare for the fine-tuning environment, install the following packages in a new conda environment:
- tensorflow (2.3.0)
- keras (2.4.3) 
- tensorflow-addons (0.11.1)
- pandas (1.1.0)
- matplotlib (3.3.0)
- scipy (1.4.1)
- scikit-learn (0.23.2)
- pillow (7.2.0)
- h5py (2.10.0)

### Scripts
All the utility functions are included in the scripts `model_main.py` and `utility_functions.py`.

`finetune_data.py` included the functions used to generate the TFRecords for fine-tuning.

`run_finetune.py` is used to fine-tune the CNN model. Example usage:
```
python run_finetune.py ${batch_size} ${tfr_train_path} ${tfr_test_path} \
${tfr_valid_path} ${pretrained_path} ${pretrained_prefix} ${train_sequence_num} \
${valid_sequence_num} ${test_sequence_num} ${finetuned_path} ${finetuned_prefix}

## Mandatory input:
  ${batch_size}: The batch size for fine-tuning.
  ${tfr_train_path}: The directory containing the training set for fine-tuning.
  ${tfr_test_path}: The directory containing the test set for fine-tuning.
  ${tfr_valid_path}: The directory containing the validation set for fine-tuning.
  ${pretrained_path}: The directory containing the saved pre-trained model.
  ${pretrained_prefix}: The name of the saved pre-trained model.
  ${train_sequence_num}: The number of training samples.
  ${valid_sequence_num}: The number of validation samples.
  ${test_sequence_num}: The number of test samples.
  ${finetuned_path}: The directory to save the fine-tuned model.
  ${finetuned_prefix}: The name of the fine-tuned model.
```


[^1]: Kelley, D.R. Cross-species regulatory sequence activity prediction. _PLoS Comput. Biol_ **16**, e1008050 (2020).
