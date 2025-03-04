# TimeMesh

A Python library for temporal and spatio-temporal data preparation, feature engineering, and windowing.

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


