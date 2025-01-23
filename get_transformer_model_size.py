#!/usr/bin/env python3

import torch
from transformers import AutoModel

def get_model_size(model_name: str) -> tuple:
    try:
        model = AutoModel.from_pretrained(model_name)
        num_parameters = sum(p.numel() for p in model.parameters())
        parameter_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        parameter_size_mb = parameter_size_bytes / (1024 ** 2)
        return num_parameters, parameter_size_mb
    except Exception as e:
        raise ValueError(f"Failed to load model: {model_name} with reason {e}")

def main():
    while True:
        user_input = input("Please enter model name (example: bert-base-uncased) or 'quit' to exit the program: ")
        if user_input.lower() == "quit":
            print('Exiting the program...')
            break
        else:
            try:
                model_name = user_input.strip()
                num_parameters, parameter_size_mb = get_model_size(model_name)
                print(f"Number of parameters: {num_parameters:,}")
                print(f"Parameter size: {parameter_size_mb:.2f} MB")
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    main()
