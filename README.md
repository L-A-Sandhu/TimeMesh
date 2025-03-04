# STLoader

Minimal time-series windowing library.

## Installation
```bash
pip install stloader

## Usage 

from stloader import DataLoader

loader = DataLoader(T=5, H=2)
X, y = loader.load_csv("data.csv")


