import os
from torchinfo import summary
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from dataset import *
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(
    ".", "").replace(":", "").replace("-", "")[:-5]

#########################
# LOAD HYPER-PARAMETERS
#########################
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--socket', type=int,  default='29210', help='Socket number')
parser.add_argument('--layers', type=int, required=True, default='2', help='Layers number')
parser.add_argument('--arch', type=str, required=True, help='BASE, LSTM or TRAN')
parser.add_argument('--batch', type=int, default=40, help='Batch size')
parser.add_argument('--sq_len', type=int, default=64, help='Window size')
parser.add_argument('--hn', type=int, default=64, help='Hidden nodes')
parser.add_argument('--epochs', type=int, default=100, help='Epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--penalty', type=float, default=1e-5, help='L2 Penalty')
parser.add_argument('--heads', type=int, default=8, help='Heads')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--drop', type=float, default=False, help='Dropout')
parser.add_argument('--scaling', type=str, default="r", help='Scaling type')

args = parser.parse_args()
socket = args.socket
layers = args.layers
arch = args.arch
batch = args.batch
sq_len = args.sq_len
hidden_nodes = args.hn
epochs = args.epochs
lr = args.lr
penalty = args.penalty
heads = args.heads
drop = args.drop
scaling = args.scaling

fit_verbose = args.verbose

# setting the folder in which save the logs and the best model
exp_dir = f"experiments_CMAPSS/{timestamp}_{socket}_{arch}_{layers}"
if not os.path.exists(exp_dir):
   os.makedirs(exp_dir)

# Report params
with open(exp_dir+"/setup.txt", 'w') as f:
    f.write(f"socket:  {socket}\n")
    f.write(f"arch:  {arch}\n")
    f.write(f"layers:  {layers}\n")
    f.write(f"lr:  {lr}\n")
    f.write(f"batch:  {batch}\n")
    f.write(f"sq_len:  {sq_len}\n")
    f.write(f"epochs:  {epochs}\n")
    f.write(f"penalty:  {penalty}\n")
    f.write(f"dropout: {drop}\n")
    f.write(f"hidden_nodes:  {hidden_nodes}\n")
    if arch == "TRAN":
        f.write(f"heads:  {heads}\n")
f.close()

if fit_verbose:
    print(f"Training of a {arch}-based model, with {layers} layers for {epochs} epochs on the {socket} dataset.")
#########################
# LOAD DATA
#########################
if fit_verbose:
    print("Data loading...")

df_list = []
n_of_series = len(os.listdir(f"../data/cmapss/series"))

# read all the series
for i in range(n_of_series):
    df = pd.read_csv(f"../data/cmapss/series/serie_{i}.csv", index_col=0).reset_index(drop=True)
    df_list.append(df)

# create datasets
datasets_list = []
for j in range(len(df_list)):
    datasets_list.append(CMAPSSWindowDataset(df_list[j], sq_len))


# # creating loaders
loaders_list = []
for k in range(len(datasets_list)):
    loaders_list.append(DataLoader(datasets_list[k], batch_size=batch, shuffle=False, drop_last=True))
        
# splitting train and test set
train_list = loaders_list[:int(len(loaders_list)*.8)]
test_list = loaders_list[int(len(loaders_list)*.8):]

for train_list_len, _ in enumerate(train_list):
    pass
for test_list_len, _ in enumerate(test_list):
    pass

#########################
# MODEL INIT
#########################
def create_model(arch):
    torch.manual_seed(314)
    if arch == "LSTM":
        model = LSTM_Model(input_size=datasets_list[0][0][0].shape[1],
                           hidden_size=hidden_nodes,
                           num_layers=layers,
                           num_classes=1,
                           batch=batch,
                           drop=drop,
                           device=device
                           )
    elif arch == "TRAN":
        model = Trans_Model(input_size=datasets_list[0][0][0].shape[1],
                            hidden_size=hidden_nodes,
                            num_layers=layers,
                            num_classes=1,
                            drop=drop,
                            heads=heads
                            )
    elif arch == "BASE":
        model = BASELINE(input_size=datasets_list[0][0][0].shape[1],
                         num_classes=1,
                         drop=drop)

    return model


# instantiate the model
candidate_model = create_model(arch).to(device)
torch.save(candidate_model.state_dict(), exp_dir+"/best_model")
# print(datasets_list[0][0][0].shape[1])

if fit_verbose:
    summary(candidate_model)

if fit_verbose:
    print("Model created...")


#########################
# FIT FUNCTION
#########################
def fit(model, train_list, test_list, epochs, criterion, optimizer, scheduler, fit_verbose):
    # init outputs
    loss_train_history = []
    val_loss_history = []
    best_loss = np.inf


    # training on one time series
    if fit_verbose:
        print("Start training...")

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss_train = 0
        j = 0
        for serie in tqdm(train_list, disable=not(fit_verbose)):
            for inputs, labels in serie:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradients to zero
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            # saving loss 
            running_loss_train += loss.item()
            j += 1

        # Computing validation
        model.eval()
        running_loss_val = 0
        k = 0
        for serie in tqdm(test_list, disable=not(fit_verbose)):
            for inputs, labels in serie:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Inference + loss + saving loss
                output = model(inputs)
                loss = criterion(output, labels)

            # saving loss 
            running_loss_val += loss.item()
            k += 1

        loss_train_history.append(running_loss_train / j)
        val_loss_history.append(running_loss_val / k)


        # Reduce on Plateau
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(round(val_loss_history[-1], 5))

        # Save losses data
        with open(exp_dir+"/loss.txt", 'a') as f:
            f.write(f"{loss_train_history[-1]}\n")
        f.close()
        with open(exp_dir+"/val_loss.txt", 'a') as f:
            f.write(f"{val_loss_history[-1]}\n")
        f.close()

        # Save epoch, lr
        with open(exp_dir+"/lr.txt", 'a') as f:
            f.write(f"{current_lr}\n")
        f.close()

        # Save model with smallest val_loss (best model)
        if val_loss_history[-1] < best_loss:
            best_loss = val_loss_history[-1]
            torch.save(model.state_dict(), exp_dir+"/best_model")
            if fit_verbose:
                print(f"Best model updated at epoch {epoch+1}, with val loss {best_loss}")

        if fit_verbose:
            if (epoch+1) % 1 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs}, train loss: {round(loss_train_history[-1], 4)}, val loss: {round(val_loss_history[-1], 4)}, lr: {current_lr}")

    if fit_verbose:
        print("End training...")


#########################
# MODEL TRAINING
########################
# Setting loss function
criterion = torch.nn.MSELoss()
# Setting optimizer
optimizer = torch.optim.Adam(candidate_model.parameters(), lr=lr, weight_decay=penalty)
# Setting the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10, cooldown=2, verbose=False)

# Training the model
fit(candidate_model, train_list, test_list, epochs, criterion, optimizer, scheduler, fit_verbose)

# Calculating model parameters
params = sum(p.numel() for p in candidate_model.parameters())
with open(exp_dir+"/setup.txt", 'a') as f:
    f.write(f"total params: {params}")
f.close()

#########################
# FINAL MODEL EVALUATION
#########################

# Load best model
best_model = create_model(arch)
best_model.load_state_dict(torch.load(exp_dir + "/best_model"))

mape_list = []
rsme_list = []
score_list = []

# loading values to scale back predictions
v = np.loadtxt(f"../data/{socket}/series_scale_back_{scaling}.txt")

# Predicting and calculating metrics
for idx, serie in enumerate(test_list):
    if idx == 9 :
        break
    y_pred = predict(best_model, serie, device, sq_len)*361
    y_true = df_list[len(train_list)+idx]["RUL"][sq_len:sq_len + len(y_pred)].reset_index(drop=True)*361


    mape, rsme, score = ts_metrics(y_true, y_pred[1:], verbose=False)
    mape_list.append(mape)
    rsme_list.append(rsme)
    score_list.append(score)

mape_avg = round(np.array(mape_list).mean(), 4)
rsme_avg = round(np.array(rsme_list).mean(), 4)
score_avg = round(np.array(score_list).mean(), 4)


# Save metrics
with open(exp_dir+"/final_metrics.txt", 'a') as f:
    f.write(f"{mape_avg}\n{rsme_avg}\n{score_avg}")
f.close()

print(timestamp, socket)
print((f"MAPE: {mape_avg}\nRSME: {rsme_avg}\nScore: {score_avg}"))
print()


with open(f"results_CMAPSS.txt", "a") as f:
    f.write(f"{timestamp},{socket},{arch},{layers},{penalty},{lr},{drop},{mape_avg},{rsme_avg},{score_avg},{epochs},{params}\n")
f.close()
