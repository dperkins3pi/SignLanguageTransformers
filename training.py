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


# DATA_DIR = 'SignEase/ASL_Citizen'   # Replace with your data directory
DATA_DIR = 'small_dataset'   # Replace with your data directory
VIDEO_DIR = DATA_DIR + '/videos'
SPLIT_DIR = DATA_DIR + '/splits'
SEGMENTED_DIR = DATA_DIR + '/segmented-videos'
JOINT_DIR = DATA_DIR + '/joint_data'
RESULTS_DIR = 'results'   # Place to store the results

STRIDE = 2   # Look at every STRIDE frames (rather than all of them, for computational efficiency)
EPOCHS = 400
FEATURE_DIM = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 8
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.001
VISION_MODEL = "ResNet18"
USE_ORIGINAL_VIDEOS = False   # If False, only use segmented videos
USE_SEGMENTED_VIDEOS = False
USE_COORDINATES = True
USE_LSTM = True
USE_ATTENTION = False
BIDIRECTIONAL = True
DROPOUT = 0.2


class TransformerKeyPointModel(nn.Module):
    def __init__(self, num_classes=10, num_frames=54, keypoint_dim=237, nhead=8, feature_dim=FEATURE_DIM, vision_model=VISION_MODEL, use_original_videos=USE_ORIGINAL_VIDEOS, use_segmented_videos=USE_SEGMENTED_VIDEOS, use_coordinates=USE_COORDINATES, use_attention=USE_ATTENTION, use_lstm=True, num_lstm_layers=2, batch_first=True, bidirectional=BIDIRECTIONAL, dropout=DROPOUT):
        super().__init__()

        self.use_original_videos = use_original_videos
        self.use_segmented_videos = use_segmented_videos
        self.use_coordinates = use_coordinates
        assert use_original_videos or use_segmented_videos or use_coordinates, "Can't set all parts of model to false"
        self.use_lstm = use_lstm
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p=dropout)

        if "resnet" in vision_model.lower():

            # Load pretrained ResNet (e.g., ResNet18)
            if "18" in vision_model.lower(): 
                if self.use_original_videos: resnet1 = models.resnet18(pretrained=True)
                resnet2 = models.resnet18(pretrained=True)
            elif "34" in vision_model.lower(): 
                if self.use_original_videos: resnet1 = models.resnet34(pretrained=True)
                resnet2 = models.resnet34(pretrained=True)
            elif "50" in vision_model.lower(): 
                if self.use_original_videos: resnet1 = models.resnet50(pretrained=True)
                resnet2 = models.resnet50(pretrained=True)

            # Create a new Conv2d for grayscale input of segmented videos (instead of 3D)
            rgb_weights = resnet2.conv1.weight.data  # shape: (64, 3, 7, 7)
            new_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight.data = rgb_weights.mean(dim=1, keepdim=True) # Average the weights across the RGB channels (axis=1)
            resnet2.conv1 = new_conv1   # Replace the first conv layer in the model

            # Freeze early layers (except the first if using grayscale since the ResNet was trained on RGB images)
            if self.use_original_videos:
                for name, param in resnet1.named_parameters():
                    if name.startswith("layer4") or name.startswith("fc"): param.requires_grad = True
                    else: param.requires_grad = False 
            if self.use_segmented_videos:
                for name, param in resnet2.named_parameters():
                    if name.startswith("layer4") or name.startswith("fc"): param.requires_grad = True
                    elif name.startswith("conv1"): param.requires_grad = True
                    else: param.requires_grad = False 

            # Create feature extractors to get features from the avgpool layer
            if self.use_original_videos: self.vision_model1 = create_feature_extractor(resnet1, return_nodes={'avgpool': 'features'})
            if self.use_segmented_videos: self.vision_model2 = create_feature_extractor(resnet2, return_nodes={'avgpool': 'features'})

            # Extract output dimension
            if self.use_original_videos or self.use_segmented_videos:
                if "18" in vision_model.lower() or "34" in vision_model.lower(): feature_dim = 512
                elif "50" in vision_model.lower() or "101" in vision_model.lower() or "152" in vision_model.lower(): feature_dim = 2048
                self.resnet_projection = nn.Linear(feature_dim, self.feature_dim)  # Or whatever dimension is appropriate

        else:
            pass
            # Vision Transformer backbone (shared weights for all frames)
            # self.vit = ViTMSNModel.from_pretrained("facebook/vit-msn-small")   # TODO: Load in a bigger ViT

        # Joint model
        if self.use_coordinates:
            self.keypoint_dim = keypoint_dim
            if self.use_attention: self.attn = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
            self.joint_model = nn.Sequential(nn.Linear(keypoint_dim, self.feature_dim), self.dropout, nn.GELU())

        # Transformer on the combined features
        # if self.use_original_videos: self.temporal_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.feature_dim*3, nhead=nhead, batch_first=True),num_layers=2)
        # else: self.temporal_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.feature_dim*2, nhead=nhead, batch_first=True),num_layers=2)
        input_dim = self.feature_dim * (int(self.use_original_videos) + int(self.use_segmented_videos) + int(self.use_coordinates))
        if self.use_lstm: self.temporal_model = nn.LSTM(input_size=input_dim, hidden_size=self.feature_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=bidirectional)
        else: self.temporal_model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True), num_layers=2)

        # Head for classification
        if self.use_lstm and self.bidirectional: self.head = nn.Sequential(nn.Linear(input_dim*2, num_classes*2), self.dropout, nn.GELU(), nn.Linear(num_classes*2, num_classes))
        else: self.head = nn.Sequential(nn.Linear(input_dim, num_classes*2), self.dropout, nn.GELU(), nn.Linear(num_classes*2, num_classes))

    def forward(self, videos, segmented_videos, joints, mask=None): 
        """
        x: (B, C, T, H, W)
        keypoints: (B, T, keypoint_dim)
        mask: (B, T)
        """
        # Extract the dimensions
        B, T, H, W = segmented_videos.shape
        _, _, D = joints.shape

        # Pass in the gray scale video through the vision model
        if self.use_original_videos:
            # Combine batch and time, bring channels forward for ResNet
            frames = videos.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, 3, H, W)
            frames = frames.view(B * T, 3, H, W)                 # (B*T, 3, H, W)
            features = self.vision_model1(frames)['features'].flatten(1)  # (B*T, feature_dim)
            features = self.resnet_projection(features)
            features = self.dropout(features)
            video_features = features.view(B, T, -1)
            # video_features = []   # Use these lines if the above causes a memory issue
            # for t in range(T):   # Apply ViT to each
            #     frame = videos[:, t].permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            #     features = self.vision_model1(frame)['features'].flatten(1)  # Batch x feature_dim
            #     video_features.append(features)
            # video_features = torch.stack(video_features, dim=1)  # (B, T, feature_dim)

        # Pass in the segmented video through the vision model
        if self.use_segmented_videos:
            frames = segmented_videos.view(B * T, 1, H, W)  # Combine batch and time, add single channel
            features = self.vision_model2(frames)['features'].flatten(1)  # (B*T, feature_dim)
            features = self.resnet_projection(features)
            features = self.dropout(features)
            segmented_features = features.view(B, T, -1)  # Reshape back to (B, T, feature_dim)
            # segmented_features = []   # Use these lines if the above causes a memory issue
            # for t in range(T):   # Apply ViT to each
            #     frame = segmented_videos[:, t].unsqueeze(1)  # (B, 1, H, W)
            #     features = self.vision_model2(frame)['features'].flatten(1)  # Batch x feature_dim
            #     segmented_features.append(features)
            # segmented_features = torch.stack(segmented_features, dim=1)  # (B, T, feature_dim)

        if self.use_coordinates:
            if self.use_attention:
                coordinates = joints.view(B * T, 79, 3)  # (B*T, 79, 3)
                attn_out, _ = self.attn(coordinates, coordinates, coordinates)  # (B*T, 79, 3)  # Pass through attention (assuming batch_first=True)
                attn_out_flat = attn_out.reshape(B * T, -1)  # (B*T, 79*3 = 237)    Flatten output of attention for joint model
            else: attn_out_flat = joints
            features = self.joint_model(attn_out_flat)  # (B*T, feature_dim)  # Pass through joint model (should accept (batch, D))
            joint_features = features.view(B, T, -1)   # Reshape back to (B, T, feature_dim)
            # joint_features = []  # Use these lines if the above causes a memory issue
            # for t in range(T):   # Apply ViT to each
            #     coordinates = joints[:, t].view(B, 79, 3)  # (B, 79, 3) as there are x,y,z coordinates for each joint
            #     attn_out, _ = self.attn(coordinates, coordinates, coordinates)  # outputs shape: (batch_size, seq_len, keypoint_dim)
            #     features = self.joint_model(attn_out.reshape(B,self.keypoint_dim))
            #     joint_features.append(features)
            # joint_features = torch.stack(joint_features, dim=1)  # (B, T, feature_dim)

        # Concatenate features, pass through temporal transformer, and then linear layers
        if self.use_coordinates:
            if self.use_original_videos and self.use_segmented_videos: all_features = torch.cat([video_features, segmented_features, joint_features], dim=2)  # Shape: (batch_size, time, 1536)
            elif self.use_segmented_videos: all_features = torch.cat([segmented_features, joint_features], dim=2)
            elif self.use_original_videos: all_features = torch.cat([video_features, joint_features], dim=2)
            else: all_features = joint_features
        else:
            if self.use_original_videos and self.use_segmented_videos: all_features = torch.cat([video_features, segmented_features], dim=2)  # Shape: (batch_size, time, 1536)
            elif self.use_segmented_videos: all_features = segmented_features
            elif self.use_original_videos: all_features = video_features
            else: all_features = joint_features

        if self.use_lstm: out, _ = self.temporal_model(all_features)
        else: out = self.temporal_model(all_features)
        pooled = out.mean(dim=1)
        logits = self.head(pooled)

        return logits

    
class Trainer():   # Class used for creating the model and training it
    def __init__(self, train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, feature_dim=FEATURE_DIM,\
        results_dir=RESULTS_DIR, save_output=True, stride=None, pretrained_weights_path=None, use_original_videos=USE_ORIGINAL_VIDEOS, use_segmented_videos=USE_SEGMENTED_VIDEOS, use_coordinates=USE_COORDINATES, use_attention=USE_ATTENTION, use_lstm=USE_LSTM, bidirectional=BIDIRECTIONAL, dropout=DROPOUT):
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
            use_original_videos (bool): Whether or not to use the original videos along with the segmented ones
        """
        # Load in the data
        self.train_dataloader, self.val_dataloader, self.test_dataloader = train_loader, val_loader, test_loader
        self.label_to_idx = label_to_idx
        self.stride = stride
        self.use_lstm = use_lstm
        self.feature_dim = feature_dim
        self.use_original_videos = use_original_videos
        self.use_segmented_videos = use_segmented_videos
        self.use_coordinates = use_coordinates
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = max(2, len(label_to_idx) if label_to_idx else 2)
        # Store hyperparameters
        self.epochs, self.batchsize, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.loss_fn = nn.CrossEntropyLoss()   # TODO: Change loss function if needed
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps")
        else: self.device = torch.device("cpu")
        self.model = TransformerKeyPointModel(num_frames=16, num_classes=self.num_classes, use_lstm=use_lstm, feature_dim=self.feature_dim, use_original_videos=self.use_original_videos, use_segmented_videos=self.use_segmented_videos, use_coordinates=self.use_coordinates, use_attention=self.use_attention, bidirectional=self.bidirectional, dropout=self.dropout).to(self.device)
        if pretrained_weights_path is not None: self.model.load_state_dict(torch.load(pretrained_weights_path, map_location=self.device))
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
            "use_original_videos": self.use_original_videos,
            "use_segmented_videos": self.use_segmented_videos,
            "use_coordinates": self.use_coordinates,
            "use_lstm": self.use_lstm,
            "use_attention": self.use_attention,
            "bidirectional": self.bidirectional,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout,
            "num_classes": self.num_classes
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
        torch.save(self.model.state_dict(), str(self.save_model_path + "last" + ".pt"))
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
        """Evaluates the model on the evaluation set, at a given epoch

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
            num_batches = len(self.val_dataloader)
            start = time.time()
            for j, batch in enumerate(self.val_dataloader):
                if not torch.cuda.is_available() and j % 50 == 0: 
                    end = time.time()
                    print(f"Batch number: {j}/{num_batches} at {end-start} seconds")
                    start = end
                if self.use_original_videos: filenames, videos, segmented_videos, joints, mask, labels = batch
                else: filenames, segmented_videos, joints, mask, labels = batch

                # Move everything to the right device
                if self.use_original_videos: videos = videos.to(self.device)
                else: videos = None
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
    
    def test_helper(self):
        """Evaluates the model on the test set, at the end

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
            for batch in self.test_dataloader:
                if self.use_original_videos: filenames, videos, segmented_videos, joints, mask, labels = batch
                else: filenames, segmented_videos, joints, mask, labels = batch

                # Move everything to the right device
                if self.use_original_videos: videos = videos.to(self.device)
                else: videos = None
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
        
        num_batches = len(self.train_dataloader)
        start = time.time()
        for j, batch in enumerate(self.train_dataloader):
            if not torch.cuda.is_available() and j % 50 == 0: 
                end = time.time()
                print(f"Batch number: {j}/{num_batches} at {end-start} seconds")
                start = end
            if self.use_original_videos: filenames, videos, segmented_videos, joints, mask, labels = batch
            else: filenames, segmented_videos, joints, mask, labels = batch

            # Move everything to the right device
            if self.use_original_videos: videos = videos.to(self.device)
            else: videos = None
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
            # print(f"[{i}] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB | Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            t_loss, t_num_correct, t_num_correct_top5, t_total = self.train_helper()
            # print(f"[{i}] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB | Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            t_acc = (t_num_correct / t_total)*100
            t_acc_top5 = (t_num_correct_top5 / t_total)*100
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_top5_acc.append(t_acc_top5)

            # print(f"[{i}] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB | Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            e_loss, e_num_correct, e_num_correct_top5, e_total = self.eval_helper()
            # print(f"[{i}] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB | Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            e_acc = (e_num_correct / e_total)*100
            e_acc_top5 = (e_num_correct_top5 / e_total)*100
            self.progress.set_description(f'Epoch {i}/{self.epochs} | Time {time.time()-start} | Train Loss: {t_loss:.4f} | Val Loss: {e_loss:.4f} | Train Acc: {t_acc:.4f} ({int(t_num_correct)}/{int(t_total)}) | Val Acc: {e_acc:.4f} ({int(e_num_correct)}/{int(e_total)}) | Train Top5_Acc: {t_acc:.4f} ({int(t_num_correct_top5)}/{int(t_total)}) | Val Top5_Acc: {e_acc:.4f} ({int(e_num_correct_top5)}/{int(e_total)})')
            self.progress.update(1)
            
            eval_loss.append(e_loss)
            eval_acc.append(e_acc)
            eval_top5_acc.append(e_acc_top5)
            evaluation_epochNum.append(i)
            best_epoch = self.save(train_loss, eval_loss, train_acc, eval_acc, train_top5_acc, eval_top5_acc, evaluation_epochNum, best_epoch)
        
        print(f"The model finished training with eval acc {eval_acc[best_epoch]} and top5 eval accuraacy {eval_top5_acc[best_epoch]}")
        print(f"The model is saved at {self.save_model_path}")
        self.model.load_state_dict(torch.load(f"{self.save_model_path}best.pt", map_location=self.device))
        t_loss, t_num_correct, t_num_correct_top5, t_total = self.test_helper()
        print(f"Final test loss: {t_loss}")
        print(f"Final test accuracy: {t_num_correct/t_total}")
        print(f"Final test top5 accuracy: {t_num_correct_top5/t_total}")

        return eval_acc



if __name__ == '__main__':

    # # If you are debugging, use smaller datasets and fewer epochs (to save time)
    # if not torch.cuda.is_available():
    #     print("CUDA is not avaiable; so we will debug")
    #     debugging = True  # Set to false when you want to use the whole dataset
    #     EPOCHS = 3
    #     BATCH_SIZE = 4
    # else:
    #     print("Using Cuda") 
    #     debugging = False
    debugging = False

    # Build datasets and loaders using helper in datasets.video_dataset
    print('Loading datasets')
    train_loader, val_loader, test_loader, label_to_idx = get_data_loaders(VIDEO_DIR, SPLIT_DIR, SEGMENTED_DIR, JOINT_DIR, batch_size=BATCH_SIZE, stride=STRIDE, debugging=debugging, use_original_videos=USE_ORIGINAL_VIDEOS)
    print('Train / Val / Test sizes:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    # Create and fit the model
    # PRETRAINED_WEIGHTS_PATH = "current_weights.pt"    # If you want to start with pretrained weights, set the path here; else set to None
    PRETRAINED_WEIGHTS_PATH = None
    trainer = Trainer(train_loader, val_loader, test_loader, label_to_idx, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, optimizer=OPTIMIZER, weight_decay=WEIGHT_DECAY, \
        stride=STRIDE, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, use_original_videos=USE_ORIGINAL_VIDEOS, use_lstm=USE_LSTM)
    trainer.fit()

    # TODO: Find best pretrained model (ViT, ResNet, e.t.c.)
    # TODO; Determine if it is worth using 3 color channels instead of gray scale
    # TODO: Add dropout
    # TODO: Weight the loss function to account for class imbalance