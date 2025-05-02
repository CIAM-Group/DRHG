# [AAAI 2025] Destroy and Repair Using Hyper-Graphs for Routing

The code can only be used for non-commercial purposes. Please contact the authors if you want to use this code for business matters.
If this repository is helpful for your research, please cite our paper:

```bash
@article{Li_Liu_Wang_Zhang_2025,   
title={Destroy and Repair Using Hyper-Graphs for Routing},  
volume={39},  
number={17}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Li, Ke and Liu, Fei and Wang, Zhenkun and Zhang, Qingfu}, 
year={2025}, 
pages={18341-18349}}
```

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
The same as that in [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).     
you can download the data from  
https://drive.google.com/drive/folders/1LptBUGVxQlCZeWVxmCzUOf9WPlsqOROR?usp=sharing  
or   
https://pan.baidu.com/s/12uxjol_5pAlnm0j4F6D_RQ?pwd=rzja

- finetuning data   
https://drive.google.com/file/d/1aU1Kpqfy2bMgY1lYdQzjdBN87W3dzE9M/view?usp=sharing
  
- testing data  
in ./TSP/data or ./CVRP/data

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
DRHG's code implementation is based on the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver) and [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD). Thanks to them.


