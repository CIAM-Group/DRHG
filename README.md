# [AAAI 2025] Destroy and Repair Using Hyper-Graphs for Routing

## dependencies
- python>=3.8
- Pytorch 1.12.1 or 1.13
- numpy==1.23.3
- matplotlib==3.5.2
- tqdm==4.64.1
- pytz==2022.1

## How to use
### Resources 
- training data
The same as that in LEHD 
https://drive.google.com/drive/folders/1LptBUGVxQlCZeWVxmCzUOf9WPlsqOROR?usp=sharing

- testing data
in ./data

### Testing
```bash
cd TSP
# for TSP100 etc
python test.py 
# for TSPlib
python test_tsplib.py 
```

### Training
```bash
cd TSP
python train.py
```

For CVRP, it's similar.

## Acknowledgements
DRHG's code implementation is based on the code of * [POMO]() and * [LEHD](). Thanks to them.


