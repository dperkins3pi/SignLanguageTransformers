import os
import time
import yaml
import torch
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision.models as models
from transformers import ViTModel, ViTConfig, ViTMSNModel
from torchvision.models.feature_extraction import create_feature_extractor
from datasets.video_dataloader import get_data_loaders


DATA_DIR = 'SignEase/ASL_Citizen'   # Replace with your data directory
VIDEO_DIR = DATA_DIR + '/videos'
SPLIT_DIR = DATA_DIR + '/splits'
SEGMENTED_DIR = DATA_DIR + '/segmented-videos'
JOINT_DIR = DATA_DIR + '/joint_data'
RESULTS_DIR = 'results'   # Place to store the results

STRIDE = 2   # Look at every STRIDE frames (rather than all of them, for computational efficiency)
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.001
VISION_MODEL = "ResNet18"


class TransformerKeyPointModel(nn.Module):
    def __init__(self, num_classes=10, num_frames=54, keypoint_dim=237, nhead=8, vision_model=VISION_MODEL):
        super().__init__()

        if "resnet" in vision_model.lower():
            # Load pretrained ResNet (e.g., ResNet18)
            if "18" in vision_model.lower(): resnet = models.resnet18(pretrained=True)
            elif "34" in vision_model.lower(): resnet = models.resnet34(pretrained=True)
            elif "50" in vision_model.lower(): resnet = models.resnet50(pretrained=True)
            # Create a new Conv2d for grayscale input (instead of 3D)
            rgb_weights = resnet.conv1.weight.data  # shape: (64, 3, 7, 7)
            new_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight.data = rgb_weights.mean(dim=1, keepdim=True) # Average the weights across the RGB channels (axis=1)
            resnet.conv1 = new_conv1   # Replace the first conv layer in the model
            self.vision_model = create_feature_extractor(resnet, return_nodes={'avgpool': 'features'})
            if "18" in vision_model.lower() or "34" in vision_model.lower(): self.feature_dim = 512
            elif "50" in vision_model.lower() or "101" in vision_model.lower() or "152" in vision_model.lower(): self.feature_dim = 2048
        else:
            pass
            # Vision Transformer backbone (shared weights for all frames)
            # self.vit = ViTMSNModel.from_pretrained("facebook/vit-msn-small")   # TODO: Load in a bigger ViT

        # Joint model
        self.keypoint_dim = keypoint_dim
        self.attn = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
        self.joint_model = nn.Sequential(nn.Linear(keypoint_dim, self.feature_dim), nn.GELU())

        # Transformer on the combined features
        self.temporal_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.feature_dim*3, nhead=nhead, batch_first=True),num_layers=2)

        # Head for classification
        self.head = nn.Sequential(nn.Linear(self.feature_dim*3, num_classes*2), nn.GELU(), nn.Linear(num_classes*2, num_classes))

    def forward(self, videos, segmented_videos, joints, mask=None):    # TODO: Make this work with keypoints;  TODO: Parameter testing (only small values in ViT work for local computer)
        """
        x: (B, C, T, H, W)
        keypoints: (B, T, keypoint_dim)
        mask: (B, T)
        """
        # Extract the dimensions
        B, T, H, W = videos.shape
        _, _, D = joints.shape

        # Pass in the gray scale video through the vision model
        video_features = []
        for t in range(T):   # Apply ViT to each   # TODO: Could remove for loop by stacking batch and temporal dimension; but this requires more memory
            frame = videos[:, t].unsqueeze(1)  # (B, 1, H, W)
            features = self.vision_model(frame)['features'].flatten(1)  # Batch x feature_dim
            video_features.append(features)
        video_features = torch.stack(video_features, dim=1)  # (B, T, feature_dim)

        # Pass in the segmented video through the vision model
        segmented_features = []
        for t in range(T):   # Apply ViT to each   # TODO: Could remove for loop by stacking batch and temporal dimension; but this requires more memory
            frame = segmented_videos[:, t].unsqueeze(1)  # (B, 1, H, W)
            features = self.vision_model(frame)['features'].flatten(1)  # Batch x feature_dim
            segmented_features.append(features)
        segmented_features = torch.stack(segmented_features, dim=1)  # (B, T, feature_dim)

        # Pass in joint features through a transformer
        joint_features = []
        for t in range(T):   # Apply ViT to each   # TODO: Could remove for loop by stacking batch and temporal dimension; but this requires more memory
            coordinates = joints[:, t].view(B, 79, 3)  # (B, 79, 3) as there are x,y,z coordinates for each joint
            attn_out, _ = self.attn(coordinates, coordinates, coordinates)  # outputs shape: (batch_size, seq_len, keypoint_dim)
            features = self.joint_model(attn_out.reshape(B,self.keypoint_dim))
            joint_features.append(features)
        joint_features = torch.stack(joint_features, dim=1)  # (B, T, feature_dim)

        # Concatenate features, pass through temporal transformer, and then linear layers
        all_features = torch.cat([video_features, segmented_features, joint_features], dim=2)  # Shape: (batch_size, time, 1536)
        out = self.temporal_transformer(all_features)
        pooled = out.mean(dim=1)
        logits = self.head(pooled)

        return logits

    

class Trainer():   # Class used for creating the model and training it
    def __init__(self, train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, \
        results_dir=RESULTS_DIR, save_output=True, stride=None):
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
        """
        # Load in the data
        self.train_dataloader, self.val_dataloader, self.test_loader = train_loader, val_loader, test_loader
        self.label_to_idx = label_to_idx
        self.stride = stride
        # Store hyperparameters
        self.epochs, self.batchsize, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.loss_fn = nn.CrossEntropyLoss()   # TODO: Change loss function if needed
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")
        self.model = TransformerKeyPointModel(num_frames=16, num_classes=max(2, len(label_to_idx) if label_to_idx else 2)).to(self.device)
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
                filenames, videos, segmented_videos, joints, mask, labels = batch

                # Move everything to the right device
                videos = videos.to(self.device)
                segmented_videos = segmented_videos.to(self.device)
                joints = joints.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
            
                # Pass in the video
                logits = self.model.forward(videos, segmented_videos, joints, mask=mask)
                
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
            filenames, videos, segmented_videos, joints, mask, labels = batch
            
            # Move everything to the right device
            videos = videos.to(self.device)
            segmented_videos = segmented_videos.to(self.device)
            joints = joints.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            # Pass in the video
            self.optimizer.zero_grad()
            logits = self.model.forward(videos, segmented_videos, joints, mask=mask)

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

            start = time.time()
            t_loss, t_num_correct, t_num_correct_top5, t_total = self.train_helper()
            t_acc = (t_num_correct / t_total)*100
            t_acc_top5 = (t_num_correct_top5 / t_total)*100
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_top5_acc.append(t_acc_top5)

            e_loss, e_num_correct, e_num_correct_top5, e_total = self.eval_helper()
            e_acc = (e_num_correct / e_total)*100
            e_acc_top5 = (e_num_correct_top5 / e_total)*100
            self.progress.set_description(f'Epoch {i}/{self.epochs} | Time {time.time()-start} | Train Loss: {t_loss:.4f} | Val Loss: {e_loss:.4f} | Train Acc: {t_acc:.4f} ({int(t_num_correct/2)}/{int(t_total/2)}) | Val Acc: {e_acc:.4f} ({int(e_num_correct/2)}/{int(e_total/2)}) | Train Top5_Acc: {t_acc:.4f} ({int(t_num_correct_top5/2)}/{int(t_total/2)}) | Val Top5_Acc: {e_acc:.4f} ({int(e_num_correct_top5/2)}/{int(e_total/2)})')
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
    train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, SEGMENTED_DIR, JOINT_DIR, batch_size=BATCH_SIZE, stride=STRIDE, debugging=debugging)
    print('Train / Val / Test sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    # Create and fit the model
    trainer = Trainer(train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, stride=STRIDE)
    trainer.fit()

    # TODO: Find best pretrained model (ViT, ResNet, e.t.c.)
    # TODO; Determine if it is worth using 3 color channels instead of gray scale
    # TODO: Add dropout
    # TODO: Weight the loss function to account for class imbalance