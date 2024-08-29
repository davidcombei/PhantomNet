from PhantomNet import PhantomNet
from PhantomNet_dataset import TrainDataset, DevDataset
import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR





def evalASV5(loader, model, device, subset):
#    conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


 #   conf_matrix.threshold = eer_thresh
#    print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh

def evalASV21(loader, model, device, subset):
 #   conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


 #   conf_matrix.threshold = eer_thresh
 #   print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh

def evalASV19(loader, model, device, subset):
  #  conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


 #   conf_matrix.threshold = eer_thresh
  #  print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh


def evalITW(loader, model, device, subset):
   # conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


#    conf_matrix.threshold = eer_thresh
 #   print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh


def evalFoR(loader, model, device, subset):
   # conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


#    conf_matrix.threshold = eer_thresh
#    print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh


def evalMLAAD(loader, model, device, subset):
    #conf_matrix = confusion_matrix(task="binary", num_classes=2).to(device)

    with torch.no_grad():
        model.eval()
        Y = torch.tensor([], device=device)
        Y_hat = torch.tensor([], device=device)
        Y_hat_score = torch.tensor([], device=device)
        num_iters = 0

        for batch in tqdm(loader):
            X, y = batch
            X = torch.as_tensor(X, dtype=torch.float, device=device)
            y = torch.as_tensor(y, dtype=torch.float, device=device)
            y_class = torch.argmax(y, dim=1)
            Y = torch.cat((Y, y_class))

            y_hat = model(X)

            y_hat = nn.Softmax(dim=1)(y_hat)
            Y_hat_score = torch.cat((Y_hat_score, y_hat[:, 1]))
            y_hat_class = torch.argmax(y_hat, dim=1)
            Y_hat = torch.cat((Y_hat, y_hat_class))
            num_iters += 1

    fpr, tpr, thresholds = roc_curve(Y.cpu().detach().numpy(),
                                     Y_hat_score.cpu().detach().numpy(), pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    eer_thresh = torch.Tensor(eer_thresh)


#    conf_matrix.threshold = eer_thresh
 #   print(conf_matrix(Y_hat_score, Y).cpu().detach().numpy())
    print(subset + " EER: ", np.round(eer * 100, 2), "| EER Threshold: ", np.round(eer_thresh.item(), 2))

    return eer, eer_thresh

def train(model, train_loader, val_loaderASV5, val_loaderASV21, val_loaderASV19, val_loaderITW,
          val_loaderFoR, val_loaderMLAAD, optimizer,scheduler, lossfn, num_epochs=25, device='cuda'):
    MAX_EER = 1000
    print("\n*** Starting the training process...")
    for epoch in range(num_epochs):
        model.train()
        tloss = 0
        num_steps = 1
        loop = tqdm(train_loader)
        for batch in loop:
            X, y = batch

            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)

            loss = lossfn(y_hat, y)
            tloss += loss.item() / num_steps
            num_steps += 1

            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=tloss)

        scheduler.step()

        # VALIDATION FOR ALL DATASETS
        eerASV5, eer_threshASV5= evalASV5(val_loaderASV5, model, device, 'ASV5')
        eerASV21, eer_threshASV21= evalASV21(val_loaderASV21, model, device, 'ASV21')
        eerASV19, eer_threshASV19 = evalASV19(val_loaderASV19, model, device, 'ASV19')
        eerITW, eer_threshITW = evalITW(val_loaderITW, model, device, 'ITW')
        eerFoR, eer_threshFoR = evalFoR(val_loaderFoR, model, device, 'FoR')
        eerMLAAD, eer_threshMLAAD = evalMLAAD(val_loaderMLAAD, model, device, 'MLAAD')

        result_str = (f"******* EVALUATION RESULTS FOR EPOCH {epoch + 1} *******\n"
                      f"ASV5 EVAL: EER = {eerASV5 * 100:.2f}, EER Threshold = {eer_threshASV5 * 100:.2f}\n"
                      f"ASV21 EVAL: EER = {eerASV21 * 100:.2f}, EER Threshold = {eer_threshASV21 * 100:.2f}\n"
                      f"ASV19 EVAL: EER = {eerASV19 * 100:.2f}, EER Threshold = {eer_threshASV19 * 100:.2f}\n"
                      f"ITW EVAL: EER = {eerITW * 100:.2f}, EER Threshold = {eer_threshITW * 100:.2f}\n"
                      f"FoR EVAL: EER = {eerFoR * 100:.2f}, EER Threshold = {eer_threshFoR * 100:.2f}\n"
                      f"MLAAD EVAL: EER = {eerMLAAD * 100:.2f}, EER Threshold = {eer_threshMLAAD * 100:.2f}\n"
                      f"{'*' * 40}\n")

        print(result_str)

        log_filename = f"PhantomNet/saved_metrics/PhantomNet_Epoch{epoch + 1}_trial.txt"
        with open(log_filename, 'w') as file:
            file.write(result_str)

        model_save_path = f"PhantomNet/saved_models/PhantomNet_SpoofMode_Epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")







def main():

   print("*** Loading training data...")
   traindataset = TrainDataset(sample_rate=16000,metafile='DATA/PhantomNet_train_metadata.txt')

   print("\n*** Loading ASV5 validation data...")
   devdataset_ASV5 = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/ASV5_dev_metadata.txt')
   print("\n*** Loading ASV21 validation data...")
   devdataset_ASV21 = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/ASV21_dev_metadata.txt')
   print("\n*** Loading ASV19 validation data...")
   devdataset_ASV19 = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/ASV19_dev_metadata.txt')
   print("\n*** Loading ITW validation data...")
   devdataset_ITW = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/metadata_ITW_dev.txt')
   print("\n*** Loading FoR validation data...")
   devdataset_FoR = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/metadata_dev_FoR.txt')
   print("\n*** Loading MLAAD validation data...")
   devdataset_MLAAD = DevDataset(sample_rate=16000,metafile='DATA/PhantomNet_dev_metadata/dev_MLAAD_metadata.txt')

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = PhantomNet(feature_size=1920, num_classes=2, conv_projection=False, use_mode='spoof').to(device)
   model.load_state_dict(
       torch.load("PhantomNet/saved_models/PhantomNet_Extractor_Epoch14_KL:1.7538.pt", map_location=device))
   optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
   scheduler = StepLR(optimizer, step_size=1, gamma=0.9) #updating the lr after each epoch with current lr*0.9 
   criterion = nn.BCEWithLogitsLoss()


   BATCH_SIZE = 32

   trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   valloader_ASV5 = DataLoader(devdataset_ASV5, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   valloader_ASV21 = DataLoader(devdataset_ASV21, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   valloader_ASV19 = DataLoader(devdataset_ASV19, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   valloader_ITW = DataLoader(devdataset_ITW, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   valloader_FoR = DataLoader(devdataset_FoR, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   valloader_MLAAD = DataLoader(devdataset_MLAAD, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


   train(model=model.to(device),
        train_loader=trainloader,
        val_loaderASV5=valloader_ASV5,
        val_loaderASV21=valloader_ASV21,
        val_loaderASV19=valloader_ASV19,
        val_loaderITW=valloader_ITW,
        val_loaderFoR=valloader_FoR,
        val_loaderMLAAD=valloader_MLAAD,
        optimizer=optimizer,
        scheduler=scheduler,
        lossfn=criterion,
        device=device)


   
if __name__ == '__main__':
    main()
