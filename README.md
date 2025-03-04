# STLoader



A Python library for creating input/output windows from time-series CSV data.

## Installation
```bash
pip install stloader

## Installation
```bash
pip install stloader

## Usage 

from stloader import DataLoader

loader = DataLoader(T=5, H=2)
X, y = loader.load_csv("data.csv")


