import os
import yaml
import torch
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import ViTModel, ViTConfig, ViTMSNModel
from datasets.video_dataloader import get_data_loaders


DATA_DIR = 'SignEase/ASL_Citizen'   # Replace with your data directory
VIDEO_DIR = DATA_DIR + '/videos'
SPLIT_DIR = DATA_DIR + '/splits'
RESULTS_DIR = 'results'   # Place to store the results

STRIDE = 3   # Look at every STRIDE frames (rather than all of them, for computational efficiency)
USE_KEYPOINTS = False
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.001


class TransformerKeyPointModel(nn.Module):
    def __init__(self, num_classes=10, num_frames=54, keypoint_dim=42, use_keypoints=True):
        super().__init__()
        self.use_keypoints = use_keypoints

        # Vision Transformer backbone (shared weights for all frames)
        self.vit = ViTMSNModel.from_pretrained("facebook/vit-msn-small")   # TODO: Load in a bigger ViT
        vit_feat_dim = self.vit.config.hidden_size
        
        # Temporal Transformer (acts on frame features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_feat_dim,
            nhead=4,
            batch_first=True  # (B, T, F)
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Head for classification
        self.head = nn.Linear(vit_feat_dim, num_classes)

    def forward(self, x, keypoints=None, mask=None):    # TODO: Make this work with keypoints;  TODO: Parameter testing (only small values in ViT work for local computer)
        """
        x: (B, C, T, H, W)
        keypoints: (B, T, keypoint_dim)
        mask: (B, T)
        """
        if keypoints is None and self.use_keypoints: raise ValueError("No keypoints passed in")

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # Permute to (B, T, C, H, W)

        vit_features = []
        for t in range(T):   # Apply ViT to each   # TODO: Could remove for loop by stacking batch and temporal dimension; but this requires more memory
            frame = x[:, t]  # (B, C, H, W)
            outputs = self.vit(frame)  # outputs.last_hidden_state shape (B, num_patches+1, D)
            vit_feat = outputs.last_hidden_state[:, 0, :]  # (B, D)
            vit_features.append(vit_feat)
        vit_features = torch.stack(vit_features, dim=1)  # (B, T, D)

        # Optionally concatenate keypoints
        if self.use_keypoints and keypoints is not None:
            # Make sure keypoints shape: (B, T, keypoint_dim)
            vit_features = torch.cat([vit_features, keypoints], dim=-1)  # (B, T, D + keypoint_dim)

        # Pass through the temporal transformer
        temp_features = self.temporal_transformer(vit_features)  # (B, T, F)

        # Optionally apply mask (if provided) before pooling - mask invalid time steps
        if mask is not None:
            # mask: (B, T), True=valid, False=masked
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            temp_features = temp_features * mask_expanded
            pooled = temp_features.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-5)  # (B, F)
        else: pooled = temp_features.mean(dim=1)  # (B, F)

        logits = self.head(pooled)  # (B, num_classes)
        return logits

    

class Trainer():   # Class used for creating the model and training it
    def __init__(self, train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, \
        results_dir=RESULTS_DIR, save_output=True, stride=None, use_keypoints=USE_KEYPOINTS):
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
            stride (int): The stride of the model (only used to store in the yaml file)
            use_keypoints (bool): Whether or not to use the keypoints
        """
        # Load in the data
        self.train_dataloader, self.val_dataloader, self.test_loader = train_loader, val_loader, test_loader
        self.label_to_idx = label_to_idx
        self.stride, self.use_keypoints = stride, use_keypoints
        # Store hyperparameters
        self.epochs, self.batchsize, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.loss_fn = nn.CrossEntropyLoss()   # TODO: Change loss function if needed
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")
        # self.model = TinyModel(in_channels=3, num_frames=16, num_classes=max(2, len(label_to_idx) if label_to_idx else 2)).to(self.device)
        self.model = TransformerKeyPointModel(num_frames=16, num_classes=max(2, len(label_to_idx) if label_to_idx else 2), use_keypoints=self.use_keypoints).to(self.device)
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
            "stride": self.stride,
            "use_keypoints": use_keypoints
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
                logits = self.model.forward(videos, keypoints=None, mask=mask)    # TODO: Pass in keypoints
                
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
            logits = self.model.forward(videos, keypoints=None, mask=mask)    # TODO: Pass in keypoints

            # Get the loss and updates
            loss = self.loss_fn(logits, labels.long())
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
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
    train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, batch_size=BATCH_SIZE, stride=STRIDE, debugging=debugging)
    print('Train / Val / Test sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    # Create and fit the model
    trainer = Trainer(train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, stride=STRIDE, use_keypoints=USE_KEYPOINTS)
    trainer.fit()

    # TODO: Make the model (instead of the arbitrary tiny model above)
        # Transformer
        # Segmentation
        # Keypoint/Pose Detection
    # TODO: Find best pretrained model
    # TODO: Add dropout
    # TODO: Weight the loss function to account for class imbalance
    # TODO: Get various accurcy metrics (top-1, top-5, precision, recall, F1, confusion matrix)