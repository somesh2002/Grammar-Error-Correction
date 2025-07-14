#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <train_file> <model_path>"
  exit 1
fi

train_file=$1
model_path=$2
path_train_file=$(realpath "$train_file")
path_model_path=$(realpath "$model_path")
if [ ! -f "$path_train_file" ]; then
  echo "Error: Train file not found at $path_train_file"
  exit 1
fi
if [ ! -d "$path_model_path" ]; then
  echo "Error: Model path not found at $path_model_path"
  exit 1
fi
python3 main.py --train --m2_file "$path_train_file" --model_path "$path_model_path"
