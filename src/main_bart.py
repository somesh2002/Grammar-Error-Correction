import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field,asdict
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sacrebleu
from sklearn.metrics import accuracy_score
from peft import PeftModel

def compute_sacrebleu_and_exact(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    
    bleu = sacrebleu.corpus_bleu(preds, [labels])
    exact_match = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    
    return {
        "bleu": bleu.score,
        "exact_match": exact_match
    }

def decode_dataset(model, tokenizer, dataset, device, max_length=128, batch_size=16):
    model.eval()
    all_preds = []
    all_labels = []

    sources = dataset["source"]
    targets = dataset["target"]

    for i in tqdm(range(0, len(sources), batch_size), desc="Evaluating"):
        batch_sources = sources[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        inputs = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=5
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(decoded_preds)
        all_labels.extend(batch_targets)

    return all_preds, all_labels

def evaluate_bleu_and_exact(model, tokenizer, train_dataset, val_dataset, device="cuda", max_length=128):
    logger.info(" Evaluating training set...")
    train_preds, train_labels = decode_dataset(model, tokenizer, train_dataset, device, max_length)
    train_metrics = compute_sacrebleu_and_exact(train_preds, train_labels)

    logger.info(" Evaluating validation set...")
    val_preds, val_labels = decode_dataset(model, tokenizer, val_dataset, device, max_length)
    val_metrics = compute_sacrebleu_and_exact(val_preds, val_labels)

    print("\n Final Evaluation Metrics:")
    print(f"Train BLEU: {train_metrics['bleu']:.2f} | Train Exact Match: {train_metrics['exact_match']*100:.2f}%")
    print(f"Val   BLEU: {val_metrics['bleu']:.2f} | Val   Exact Match: {val_metrics['exact_match']*100:.2f}%")

    return {
        "train": train_metrics,
        "val": val_metrics
    }


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class GECConfig:
    # Directory paths
    output_dir: str = "./save_model"
    cache_dir: str = "./cache"
    logging_dir: str = "./logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    model_name_or_path: str = "facebook/bart-large"  # or "facebook/bart-large"
    max_target_length: int = 128
    num_train_epochs: int = 7
    batch_size: int = 32
    learning_rate: float = 3e-4

    # Evaluation
    metric_for_best_model: str = "loss"
    load_best_model_at_end: bool = True
    save_total_limit: int = 2
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # LoRA-specific
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: str = "none"
    lora_task_type: str = "SEQ_2_SEQ_LM"
    batch_correct_batch_size: int = 32
    entry_number = "2024AIB2292"


    def to_json(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"Config saved to {save_path}")

    @staticmethod
    def from_json(json_path: str):
        json_path = json_path+"/config.json"
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return GECConfig(**config_dict)

class M2Parser:
    """Parser for M2 formatted GEC data."""

    @staticmethod
    def parse_m2_file(filename: str) -> List[Dict]:
        """
        Parse an M2 file into a list of sentence dictionaries.

        Args:
            filename: Path to M2 file

        Returns:
            List of dictionaries with source and target sentences
        """
        data = []
        current_sentence = {}
        source_sentence = None
        corrections = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('S '):
                    if source_sentence is not None and corrections:
                        current_sentence = {
                            'source': source_sentence,
                            'corrections': corrections
                        }
                        data.append(current_sentence)

                    source_sentence = line[2:]
                    corrections = []

                elif line.startswith('A '):
                    if "noop" in line:
                        continue
                    parts = line[2:].split("|||")
                    if len(parts) >= 3:
                        start_idx = int(parts[0].split()[0])
                        end_idx = int(parts[0].split()[1])
                        error_type = parts[1]
                        correction = parts[2]
                        corrections.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'error_type': error_type,
                            'correction': correction
                        })

        if source_sentence is not None and corrections:
            current_sentence = {
                'source': source_sentence,
                'corrections': corrections
            }
            data.append(current_sentence)

        return data

    @staticmethod
    def apply_corrections(source: str, corrections: List[Dict]) -> str:
        """
        Apply corrections to a source sentence.

        Args:
            source: Source sentence
            corrections: List of correction dictionaries

        Returns:
            Corrected sentence
        """

        tokens = source.split()
        sorted_corrections = sorted(corrections, key=lambda x: (x['start_idx'], x['end_idx']), reverse=True)

        for correction in sorted_corrections:
            start_idx = correction['start_idx']
            end_idx = correction['end_idx']
            corrected_text = correction['correction']

            if start_idx < len(tokens):
                del tokens[start_idx:end_idx]

                if corrected_text.strip():
                    corrected_tokens = corrected_text.split()
                    for i, token in enumerate(corrected_tokens):
                        tokens.insert(start_idx + i, token)

        corrected_sentence = ' '.join(tokens)

        return corrected_sentence
 
class GECorrector:
    """GEC system using the BART model."""

    def __init__(self, config: GECConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device)
        base_model_path = model_path if model_path else config.model_name_or_path
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        if model_path:
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            if config.use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType[config.lora_task_type],
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config.lora_target_modules,
                    bias=config.lora_bias
                )
                self.model = get_peft_model(base_model, lora_config)
            else:
                self.model = base_model

        self.max_length = config.max_target_length
        self.model.to(self.device)

    def load_and_prepare_data(self, train_file: str) -> Tuple[Dataset, Dataset]:
        logger.info("Loading training and validation data")
        
        # Load M2 files
        store_data = M2Parser.parse_m2_file(train_file)
        data, _ = train_test_split(store_data, test_size=0.95, random_state=42)
        train_data = []
        val_data = []               
        training_data, validation_data = train_test_split(data, test_size=0.05, random_state=42)
        tag = "grammar: "
        for entry in training_data:
            source = entry['source']
            corrections = entry.get('corrections', None)
            if corrections  is None:
                continue
            corrected_sentence = M2Parser.apply_corrections(source, corrections)
            train_data.append({
                'source': tag + source,
                'target': corrected_sentence
            })
        for entry in validation_data:
            source = entry['source']
            corrections = entry.get('corrections', None)
            if corrections is  None:
                continue
            corrected_sentence = M2Parser.apply_corrections(source, corrections)
            val_data.append({
                'source': tag + source,
                'target': corrected_sentence
            })

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        return [train_dataset, val_dataset]

    def preprocess(self, entry):
        inputs = self.tokenizer(
            entry["source"], max_length=self.max_length, padding="max_length", truncation=True
        )
        targets = self.tokenizer(
            entry["target"], max_length=self.max_length, padding="max_length", truncation=True
        )
        inputs["labels"] = [
            (label if label != self.tokenizer.pad_token_id else -100) for label in targets["input_ids"]
        ]
        return inputs



    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        
        logger.info("Starting training")
        tokenized_train = train_dataset.map(self.preprocess, remove_columns=["source", "target"])
        tokenized_val = val_dataset.map(self.preprocess, remove_columns=["source", "target"])

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            report_to="none"
        )

        # Step 4: Set up the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        logger.info("Training completed")

        

    def batch_correct(self, sentences: List[str]) -> List[str]:
        num_beams = 10
        self.model.eval()
        inputs = self.tokenizer(
            [f"grammar: {s}" for s in sentences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=num_beams
            )
        logger.info(f"Correcting {len(sentences)} sentences")
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model, tokenizer, and config saved to {path}")

    @classmethod
    def load(cls, path: str, config: Optional[GECConfig] = None):
        return cls(config=config,model_path=path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEC using BART")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--m2_file", type=str, help="Path to M2 file for training")
    parser.add_argument("--correct", action="store_true", help="Correct sentences")
    parser.add_argument("--evaluate", action="store_true", help="Metric Calculation")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--model_path", type=str, default="./t5_lora_gec", help="Path to save/load model")
    args = parser.parse_args()

    config = GECConfig()
    # config = GECConfig.from_json(args.model_path)
    print(config)
    if args.train and args.m2_file:
        corrector = GECorrector(config)
        train_dataset, val_dataset = corrector.load_and_prepare_data(args.m2_file)
        corrector.train(train_dataset, val_dataset)
        corrector.save(args.model_path)
    else:
        corrector = GECorrector.load(args.model_path, config)
        print("Model loaded successfully")
    if args.evaluate and args.m2_file:
        print("Evaluating the model")
        # Load the model and tokenizer
        # Load the M2 file and prepare the dataset
        corrector = GECorrector(config)
        corrector = GECorrector.load(args.model_path, config)
        train_dataset, val_dataset = corrector.load_and_prepare_data(args.m2_file)
        metrics = evaluate_bleu_and_exact(corrector.model, corrector.tokenizer, train_dataset, val_dataset, device=config.device)
        print(metrics)



    if args.correct and args.output_file:
        output_file = args.output_file
        batch_size = 16
        df = pd.read_csv(output_file)
        sentences = df['source'].tolist()
        predictions = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            batch_preds = corrector.batch_correct(batch)
            predictions.extend(batch_preds)
        df['prediction'] = predictions
        df.to_csv(output_file, index=False)
