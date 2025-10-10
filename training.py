import os
import yaml
import torch
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from datasets.video_dataloader import get_data_loaders


DATA_DIR = 'SignEase/ASL_Citizen'   # Replace with your data directory
VIDEO_DIR = DATA_DIR + '/videos'
SPLIT_DIR = DATA_DIR + '/splits'
RESULTS_DIR = 'results'   # Place to store the results

EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.001


class TinyModel(torch.nn.Module):    # TODO: Replace with the actual model
    def __init__(self, in_channels=3, num_frames=16, num_classes=10):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))  # pool C,T,H,W -> C,1,1,1
        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, x, mask=None):
        # x: (B, C, T, H, W)
        # If mask is provided, zero out padded frames before pooling
        if mask is not None:
            # mask: (B, T) -> (B, 1, T, 1, 1)
            m = mask[:, None, :, None, None].to(x.dtype)
            x = x * m
        pooled = self.pool(x).reshape(x.shape[0], -1)  # (B, C)
        return self.fc(pooled)
    



class Trainer():   # Class used for creating the model and training it
    def __init__(self, train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, \
        results_dir=RESULTS_DIR, save_output=True):
        """Load in the data, create the model, and make directories to store future plots
        
        Args:
            train_loader (DataLoader): The training dataset
            val_loader (DataLoader): The validation dataset
            test_loader (DataLoader): The testing dataset
            label_to_idx (dict): The mapping of labels (words) to indices
            epochs (int): The number of epochs to train. Defaults to 200.
            batch_size (int): The size of each batch. Defaults to 16.
            lr (float): The learning rate. Defaults to 0.001.
            optimizer (str): The name of the optimizer to be used. Defaults to "SGD".
            weight_decay (float): The weight_decay value to be used in the optimizer. Defaults to 0.001.
            results_dir (str): The file path where the results will be stored
            save_output (bool): Whether or not to output the plots. Defaults to True.
        """
        # Load in the data
        self.train_dataloader, self.val_dataloader, self.test_loader = train_loader, val_loader, test_loader
        self.label_to_idx = label_to_idx
        # Store hyperparameters
        self.epochs, self.batchsize, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.loss_fn = nn.CrossEntropyLoss()   # TODO: Change loss function if needed
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")
        self.model = TinyModel(in_channels=3, num_frames=16, num_classes=max(2, len(label_to_idx) if label_to_idx else 2)).to(self.device)
        if optimizer.lower()=="sgd": self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower()=="adamw": self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower()=="adam": self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Make directories for the results (if they don't already exist)
        self.save_output = save_output 
        n = 1   # Find the next available folder name by incrementing n
        while os.path.exists(f"{results_dir}/Test{n}"): n += 1
        results_dir = f"{results_dir}/Test{n}/" 
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)   # Create the new folder
        print("The Results Will Be Stored At", results_dir)
        
        if self.save_output:
            self.save_model_path = results_dir + "model/"  # where the model is going to save
            os.makedirs(self.save_model_path, exist_ok=True)
        
        hyperparams = {   # TODO: Add any other hyperparameters you want to store
            "epochs": self.epochs,
            "batchsize": self.batchsize,
            "lr": self.lr,
            "optimizer": optimizer,
            "weight_decay": self.weight_decay,
        }
        print(f"The hyperparameters are {hyperparams} and will be stored at {self.save_model_path}config.yaml")
        with open(str(self.save_model_path) + "config.yaml", "w") as f: yaml.dump(hyperparams, f)

    def save(self, train_loss, eval_loss, train_acc, eval_acc, train_top5_acc, eval_top5_acc, eval_epochNum, prev_best_epoch):
        """Makes plots for the the losses and accuracies

        Args:
            train_loss (float): The training losses
            eval_loss (float): The evaluation losses
            train_acc (float): The training accuracies
            eval_acc (float): The evaluation accuracies
            train_acc_top5 (float): The top5 training accuracies
            eval_acc_top5 (float): The top5 evaluation accuracies
            eval_epochNum (list): A list of each epoch used in evaluation
            prev_best_epoch (int): The previous best epoch number
            
        Returns:
            best_epoch (int): The new best epoch number
        """
        def makePlot(firstX, firstY, secondX, secondY, title, label, fileName):
            """Creates the plots"""
            plt.plot(firstX, firstY)
            plt.plot(secondX, secondY)
            if title=="Loss Plot": plt.title(f"{title}: Min Eval_Loss: {round(min(secondY),5)} at epoch {np.argmin(secondY)}")
            elif title=="Accuracy Plot": plt.title(f"{title}: Max Eval_Acc: {round(max(secondY),5)} at epoch {np.argmax(secondY)}")
            plt.legend(["train","eval"])
            plt.xlabel("Epoch")
            plt.ylabel(label)
            plt.savefig(self.results_dir + fileName)
            plt.close()

        if(self.save_output):
            """Make the plots"""
            # Loss plot
            makePlot(range(1, len(train_loss)+1), train_loss, eval_epochNum, eval_loss, title="Loss Plot", label="Mean Loss",
                    fileName="loss" + ".png")
            # Accuracy plots
            makePlot(range(1, len(train_acc)+1), train_acc, eval_epochNum, eval_acc, title="Accuracy Plot",
                    label="Mean Accuracy", fileName="acc" + ".png")
            makePlot(range(1, len(train_acc)+1), train_top5_acc, eval_epochNum, eval_top5_acc, title="Accuracy Plot",
                    label="Mean Top5 Accuracy", fileName="acc_top5" + ".png")
        
        # Save the weights if we have the new best model (on validation first, then training if there is a tie)
        prev_best_val = eval_acc[prev_best_epoch] if prev_best_epoch is not None else 0.
        prev_best_val_top5 = eval_top5_acc[prev_best_epoch] if prev_best_epoch is not None else 0.
        prev_best_train = train_acc[prev_best_epoch] if prev_best_epoch is not None else 0.
        if prev_best_epoch is None: 
            torch.save(self.model.state_dict(), str(self.save_model_path + "best" + ".pt"))
            best_epoch = len(eval_acc) - 1
            print(f"New best model saved at epoch {best_epoch+1} with validation accuracies {eval_acc[-1]}-{eval_top5_acc[-1]} and training accuracies {train_acc[-1]}-{train_top5_acc[-1]}")
        elif eval_acc[-1] > prev_best_val: 
            torch.save(self.model.state_dict(), str(self.save_model_path + "best" + ".pt"))
            best_epoch = len(eval_acc) - 1
            print(f"New best model saved at epoch {best_epoch+1} with validation accuracies {eval_acc[-1]}-{eval_top5_acc[-1]} and training accuracies {train_acc[-1]}-{train_top5_acc[-1]}")
        elif eval_acc[-1]==prev_best_val and eval_top5_acc[-1]>prev_best_val_top5: 
            torch.save(self.model.state_dict(), str(self.save_model_path + "best" + ".pt"))
            best_epoch = len(eval_acc) - 1
            print(f"New best model saved at epoch {best_epoch+1} with validation accuracies {eval_acc[-1]}-{eval_top5_acc[-1]} and training accuracies {train_acc[-1]}-{train_top5_acc[-1]}")
        elif eval_acc[-1]==prev_best_val and eval_top5_acc[-1]==prev_best_val_top5 and train_acc[-1]>prev_best_train: 
            torch.save(self.model.state_dict(), str(self.save_model_path + "best" + ".pt"))
            best_epoch = len(eval_acc) - 1
            print(f"New best model saved at epoch {best_epoch+1} with validation accuracies {eval_acc[-1]}-{eval_top5_acc[-1]} and training accuracies {train_acc[-1]}-{train_top5_acc[-1]}")
        else: best_epoch = prev_best_epoch
        
        return best_epoch

    def eval_helper(self):
        """Evaluates the model on the test set, at a given epoch

        Args:
            epochNum (int): Current epoch number
            train_loss (float): The loss from training
            train_acc (float): The training accuracy

        Returns:
            loss_val (float): The average loss for all instances in the test set
            num_correct (int): The number of teams labelled correctly
            total (int): The number of total teams
            pred_arr (tensor 1-D): The predictions (OFFENSE=1 or DEFENSE=0) for each team
            label_arr (tensor 1-D): The true labels (OFFENSE=1 or DEFENSE=0) for each team
        """
        num_correct, num_top5_correct, total = 0, 0, 0
        losses =[]
        self.model.eval()
        
        with torch.inference_mode():
            for batch in self.val_dataloader:
                videos, filenames, mask, labels = batch   # Load in the data

                # Move everything to the right device
                videos = videos.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
            
                # Pass in the video
                logits = self.model.forward(videos)   # TODO: May need to pass in more things (like the mask)
                
                # Get the loss
                loss = self.loss_fn(logits, labels.long())
                losses.append(loss.item())
                
                # Get accuracy
                predictions = torch.argmax(logits, dim=1)  
                correct = (predictions == labels)
                num_correct += correct.sum().item()

                # Top-5 accuracy
                top5_preds = torch.topk(logits, k=5, dim=1).indices
                top5_correct = top5_preds.eq(labels.unsqueeze(1))
                num_top5_correct += top5_correct.any(dim=1).sum().item()
                total += labels.size(0)
        
        loss_val = np.mean(losses)
        
        return loss_val, num_correct, num_top5_correct, total

    def train_helper(self):
        """Trains the model on the training set for an entire epoch
        
        Returns:
            t_loss (float): The average training loss of all instances in the training set
            num_correct (int): The number of teams labelled correctly
            total (int): The number of total teams
        """
        num_correct, num_top5_correct, total = 0, 0, 0
        losses = []
        self.model.train()
        
        for batch in self.train_dataloader:
            videos, filenames, mask, labels = batch   # Load in the data
            
            # Move everything to the right device
            videos = videos.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            # Pass in the video
            self.optimizer.zero_grad()
            logits = self.model.forward(videos)   # TODO: May need to pass in more things (like the mask)

            # Get the loss and updates
            loss = self.loss_fn(logits, labels.long())
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():  # TODO: Get accuracy (and store it in num_correct and total)
                # Accuracy
                predictions = torch.argmax(logits, dim=1)  
                correct = (predictions == labels)
                num_correct += correct.sum().item()

                # Top-5 accuracy
                top5_preds = torch.topk(logits, k=5, dim=1).indices
                top5_correct = top5_preds.eq(labels.unsqueeze(1))
                num_top5_correct += top5_correct.any(dim=1).sum().item()

                total += labels.size(0)
                
        return np.mean(losses), num_correct, num_top5_correct, total

    def fit(self):
        """Train the entire model for a specified number of epochs

        Returns:
            eval_acc (list): The evaluation accuracies at each evaluation step
        """
        best_epoch = None
        train_loss, train_acc, train_top5_acc = [], [], []
        eval_loss, eval_acc, eval_top5_acc = [], [], []
        evaluation_epochNum = []
        self.progress = tqdm(total=self.epochs, desc='Training', position=0, unit="epochs")
        
        for i in range(1, self.epochs + 1):

            t_loss, t_num_correct, t_num_correct_top5, t_total = self.train_helper()
            t_acc = (t_num_correct / t_total)*100
            t_acc_top5 = (t_num_correct_top5 / t_total)*100
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_top5_acc.append(t_acc_top5)

            e_loss, e_num_correct, e_num_correct_top5, e_total = self.eval_helper()
            e_acc = (e_num_correct / e_total)*100
            e_acc_top5 = (e_num_correct_top5 / e_total)*100
            self.progress.set_description(f'Epoch {i}/{self.epochs} | Train Loss: {t_loss:.4f} | Val Loss: {e_loss:.4f} | Train Acc: {t_acc:.4f} ({int(t_num_correct/2)}/{int(t_total/2)}) | Val Acc: {e_acc:.4f} ({int(e_num_correct/2)}/{int(e_total/2)}) | Train Top5_Acc: {t_acc:.4f} ({int(t_num_correct_top5/2)}/{int(t_total/2)}) | Val Top5_Acc: {e_acc:.4f} ({int(e_num_correct_top5/2)}/{int(e_total/2)})')
            self.progress.update(1)
            
            eval_loss.append(e_loss)
            eval_acc.append(e_acc)
            eval_top5_acc.append(e_acc_top5)
            evaluation_epochNum.append(i)
            best_epoch = self.save(train_loss, eval_loss, train_acc, eval_acc, train_top5_acc, eval_top5_acc, evaluation_epochNum, best_epoch)
        
        return eval_acc



if __name__ == '__main__':

    # If you are debugging, use smaller datasets and fewer epochs (to save time)
    debugging = True  # Set to false when you want to use the whole dataset
    if debugging:
        EPOCHS = 3
        BATCH_SIZE = 4

    # Build datasets and loaders using helper in datasets.video_dataset
    print('Loading datasets')
    train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, batch_size=BATCH_SIZE, debugging=debugging)
    print('Train / Val / Test sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    # Create and fit the model
    trainer = Trainer(train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY)
    trainer.fit()

    # TODO: Make the model (instead of the arbitrary tiny model above)
        # 3D Transformer
        # Segmentation
        # Keypoint/Pose Detection
    # TODO: Add dropout
    # TODO: Weight the loss function to account for class imbalance
    # TODO: Get various accurcy metrics (top-1, top-5, precision, recall, F1, confusion matrix)