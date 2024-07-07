import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_path = os.path.abspath(args.dataset_path)
    print(f"Dataset path: {dataset_path}")

    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )

    # train_setの定義
    try:
        train_set = loader.get_train_dataset()
        train_set_size = len(train_set)
        print(f"train_set size: {train_set_size}")
        if train_set_size >= 3900:
            train_set = random.sample(list(train_set), 3900)
        else:
            print(f"Warning: train_set size is smaller than 3900. Using full train_set of size {train_set_size}")
            train_set = list(train_set)
    except Exception as e:
        print(f"Error loading train dataset: {e}")
        return

    try:
        test_set = loader.get_test_dataset()
        test_set_size = len(test_set)
        print(f"test_set size: {test_set_size}")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    collate_fn = train_collate

    train_batch_size = min(args.data_loader.train.batch_size, train_set_size)
    test_batch_size = min(args.data_loader.test.batch_size, test_set_size)

    print(f"train_batch_size: {train_batch_size}, test_batch_size: {test_batch_size}")

    # データローダーの作成
    try:
        print("Creating DataLoader...")
        train_data = DataLoader(train_set,
                                batch_size=train_batch_size, shuffle=args.data_loader.train.shuffle,
                                collate_fn=collate_fn,
                                drop_last=False)
        test_data = DataLoader(test_set,
                               batch_size=test_batch_size,
                               shuffle=args.data_loader.test.shuffle,
                               collate_fn=collate_fn,
                               drop_last=False)
        print("DataLoader created successfully.")
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return

    # モデルの作成
    try:
        model = EVFlowNet(args.train).to(device)
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    # オプティマイザーの作成
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
        print("Optimizer created successfully.")
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        return

    # 学習率スケジューラーの追加
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    print("Scheduler created successfully.")

    # トレーニングの開始
    model.train()
    try:
        for epoch in range(args.train.epochs):
            total_loss = 0
            print(f"on epoch: {epoch+1}")
            for i, batch in enumerate(tqdm(train_data)):
                batch: Dict[str, Any]
                event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
                ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
                try:
                    flow = model(event_image)  # [B, 2, 480, 640]
                    loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
                    print(f"batch {i} loss: {loss.item()}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA out of memory. Trying to allocate less memory...")
                        torch.cuda.empty_cache()
                        train_batch_size = max(1, train_batch_size // 2)  # バッチサイズを半分にして再試行
                        print(f"Reduced train_batch_size to {train_batch_size}")
                        train_data = DataLoader(train_set,
                                                batch_size=train_batch_size, shuffle=args.data_loader.train.shuffle,
                                                collate_fn=collate_fn,
                                                drop_last=False)
            print(f'Epoch {epoch+1} completed. Loss: {total_loss / len(train_data)}')
            scheduler.step()
    except Exception as e:
        print(f"Error during training: {e}")
        return

    print("Training completed.")

    # モデルの保存
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    # 予測の開始
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("Predicting on test set...")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            if "event_volume" in batch:
                event_image = batch["event_volume"].to(device)
                batch_flow = model(event_image)  # [1, 2, 480, 640]
                flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
            else:
                print("'event_volume' キーは存在しません。")
        print("Prediction completed.")

<<<<<<< HEAD
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)
=======
    # submissionファイルの保存
    file_name = "submission.npy"
    try:
        save_optical_flow_to_npy(flow, file_name)
        print(f"Submission saved to {file_name}")
    except Exception as e:
        print(f"Error saving submission: {e}")
>>>>>>> 29bbad8fa5a1e6b9ab2648678e986f563b8eaac7

if __name__ == "__main__":
    main()
