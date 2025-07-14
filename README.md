# Grammar-Error-Correction
> This repository provides a implementation of finetuned Large Language models  for the task of **Grammatical Error Correction (GEC)**. The models are trained to take grammatically incorrect English sentences and output corrected versions without tchaging their semantics and meaning.

## Abstract
Grammatical errors in written communication, particularly among non-native English speakers, are common and can hinder understanding. Deep Learning models has been put to use, but they are not giving that good results. Advanced general based LLMs are capable of understanding the basics language, and they can be used for sentence correction. Here "T5-Base" and "BART-Large" has been employed after finetuning on the train dataset. Here T5-Base, being a encoder-decoder models has given better results.

## Datasets
Standard dataset has been used, which was accumulated as a part of the course. Datasets can be accessed on the datasets folder.

## Methodology
### Training

- Selected **T5-base** and **BART-large** from HuggingFace Transformers.
- Applied  **LoRA** Finetuning for parameter-efficient training, since the dataset is large.
- Read the training M2 files and parse data to generate incorrect and correct sentences.
- For T5 and BaRT Models, data was trained using seq2seq formulation: `"grammar: <input>" → <corrected output>"`
- Several Tests were conducted to decide the report the best values and model.
- Added **beam search decoding** for better grammatical fluency with beam search size of 10.
- Evaluated models using GEC metrics: **BLeu Score** and **Exact Matching Accuracy**.

## Results
- T5-base reached **82.02 BeRT score** on test dataset with strong performance on various error types.
- BaRT large repoted the **77.23 BERT Score** on test dataset.
- LoRA-finetuned variants significantly reduced training time and memory while maintaining performance.
- T5 Models has performed better than BART-Large models.

## Outputs

### Positive Outputs

Input : It has a interesting past . 

Output: It has an interesting past . 

### Negative Outputs

Input: The entery will be free . 

Output: The enterance will be free . 


## Folder Structure
Overview:
- environemnt.yml: Create the conda environemnt given all the libraries.
- src/ : This folder contains all the python  and bash scripts used to run the experiments.
- doc/ : Contains the report for the project
- datasets/: Contains the dataset for running the model
- models/ : Contains all models trained for the program

## Language and Libraries Used
- Programming Language: Python
- Libraries: Pytorch, Numpy, Pandas, scikit-learn, json  

## Set up the Environment
conda env create -f environment.yml
source activate t5-lora-env

## Running the Code
Run the test script and train script
Train Data: Format of m2.
Model path: Path to folder
Output File: Input the file in CSV Format.


For Train script
```bash
python3 main.py --train --m2_file "$path_train_file" --model_path "$path_model_path"

```

For Test Script
```bash
python3 main.py --correct --output_file="$path_submission_file" --model_path="$path_model_path"

```
## References
1. https://blog.stackademic.com/fine-tuning-t5-for-grammar-correction-a-step-by-step-guide-edba96ada787 
2. https://www.kaggle.com/code/aisuko/fine-tuning-t5-small-with-lora

## Developer and License 
This project is a part of the academic course assignment submitted at the end of the course "Deep Learning in NLP - ELL884", taught by the Prof. Tanmoy Chakraborty.

Student Name: Somesh Agrawal