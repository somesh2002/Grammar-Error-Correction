#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_path> <submission_file>"
  exit 1
fi

model_path=$1
submission_file=$2
path_submission_file=$(realpath "$submission_file")
path_model_path=$(realpath "$model_path")
if [ ! -f "$path_submission_file" ]; then
  echo "Error: Submission file not found at $path_submission_file"
  exit 1
fi
if [ ! -d "$path_model_path" ]; then
  echo "Error: Model path not found at $path_model_path"
  exit 1
fi

python3 main.py --correct --output_file="$path_submission_file" --model_path="$path_model_path"
