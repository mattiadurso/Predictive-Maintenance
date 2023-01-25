import os

# Brute-force parameter optimization 

for arch  in ["LSTM"]: 
    for layer in [1,2,4]:
        for pen in [0, 1e-3, 1e-5]:
            for lr in [0.01, 0.001, 0.0001]:
                for p in [0, .2, .4, .6]:
                    os.system(f"python train_cmapss.py --arch {arch} --layer {layer} --penalty {pen} --lr {lr}  --drop {p} --epochs 100 --batch 100 --verbose True")
