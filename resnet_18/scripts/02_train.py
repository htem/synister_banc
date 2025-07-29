#import gunpowder as gp
import sys
import torchio as tio
import zarr
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import os
import random
from loss import FocalLoss
from resnet_18 import resnet_18_model, resnet_2_1_model
import warnings
import gc


""" 02_train.py is used to train the Neurotransmitter model

All code inspired from the Funke Lab's Neurotransmitter work 
https://github.com/funkelab/synister/tree/dev
"""


class ZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_array, locations, ids, nt_gt_list, cell_type_gt_list,  shape=(160, 160, 14), weight_sampler=None, transform=None, 
        voxel_size=None, train=False, aug_transforms=None, super_aug_transforms=None, super_super_aug_transforms=None,):
        """
        Torch dataset getting locations from a zarr array.

        Parameters
        ----------
        zarr_array : zarr.core.Array
            Zarr array containing the data.
        locations : np.ndarray
            Array of locations to extract from the zarr array, assumed to be in the same order as the zarr array.
            So, if the zarr array is organized as (z, y, x), the locations should be in the same order.
        ids : np.ndarray
            Array of identifiers for the locations.
        nt_gt_list: np.ndarray
            Array of gt labels for the locations
        shape : tuple
            Shape of the output array. The dataset will return a crop, centered at the location, of this shape.
        transform : callable
            Transform to apply to the data.
        voxel_size : tuple
            Voxel size of the data. If None, we assume isotropic data, and locations in voxel coordinates.
            If not None, we assume locations in world coordinates - the locations will be rounded to the nearest voxel.
        """
        self.zarr_array = zarr_array
        self.locations = locations
        self.ids = ids
        assert len(locations) == len(ids)
        self.voxel_size = voxel_size
        self.nt_gt_list = nt_gt_list
        self.cell_type_gt_list = cell_type_gt_list
        self.shape = shape
        self.transform = transform
        self.train = train
        self.aug_transforms = aug_transforms
        self.super_aug_transforms = super_aug_transforms
        self.super_super_aug_transforms = super_super_aug_transforms
        self.weight_sampler = weight_sampler

    def __len__(self):
        return len(self.locations)

    def _voxel_location(self, location):
        """ Convert a location in world coordinates to voxel coordinates. """
        if True: #self.voxel_size is None:
            return np.array(location).astype(int)
        return np.round(location / self.voxel_size).astype(int)

    def __getitem__(self, idx):
        center = self._voxel_location(self.locations[idx])

        if self.train: # Apply some location jitter
            center[0] += random.randint(-10, 10)
            center[1] += random.randint(-10, 10)
            center[2] += random.randint(-1, 1)

        corner = center - (np.array(self.shape) // 2)

        # synapse id
        this_id = int(self.ids[idx])

        # cell type of synapse
        this_cell_type = self.cell_type_gt_list[idx]

        # neurotransmitter type
        this_gt = int(self.nt_gt_list[idx])

        # 3d local cutout
        array = np.array(
            self.zarr_array[
                int(corner[0]) : int(corner[0] + self.shape[0]),
                int(corner[1]) : int(corner[1] + self.shape[1]),
                int(corner[2]) : int(corner[2] + self.shape[2]),
            ]
        )[np.newaxis]

        # Apply augmentations ONLY in TRAINING. Add further augmentations depending on the class
        if self.train: 
            if this_gt in [1,4]:
                if self.super_super_aug_transforms:
                    array = self.super_super_aug_transforms(array)
            elif this_gt in [5,6,7]:
                if self.super_aug_transforms:
                    array = self.super_aug_transforms(array)
            else:
                if self.aug_transforms:
                    array = self.aug_transforms(array)

        if self.transform: # general normalization
            array = self.transform(array)

        if idx % 100 == 0: # Garbage collector
            gc.collect()

        return array, this_id, this_gt, this_cell_type


# class Transform:
#     def __init__(self, mean, std, max_value=255):
#         self.mean = mean
#         self.std = std
#         self.max_value = max_value
#     def __call__(self, sample):
#         return torch.from_numpy(
#             ((sample / self.max_value) - self.mean) / self.std
#         ).float()


def train(
    model=None,
    valmodel=None,
    experiment_dir: Path = None,
    file_path: str = None,
    test_file_path: str = None,
    num_iterations=100000,
    batch_size=64,
    save_every=50000,
    snapshot_every=50000,
    log_every=100,
    input_shape=(160, 160, 14),
    lr=1e-4,
    num_cache_workers=12,
    point_id=None,
    nt_name=None,
    # Data
    container: str = None,
    dataset: str = None,
    voxel_size=(4, 4, 45),
    num_transmitters: int = 8,
    coordinate_order: str = "xyz",
    num_workers: int = 8,
):
    """
    Train a classifier model on the synapse data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    file_path : str
        Path to the feather file containing the ground truth data.
    log_dir : str
        Directory to save the logs.
    num_iterations : int
        Number of iterations to train the model.
    batch_size : int
        Batch size.
    save_every : int
        Save the model every `save_every` iterations.
    snapshot_every : int
        Save the snapshots every `snapshot_every` iterations.
    log_every : int
        Log the training every `log_every` iterations.
    input_shape : tuple
        Shape of the input to the model.
    voxel_size : tuple
        Voxel size of the input data.
    lr : float
        Learning rate.
    coordinate_order : str
        Order of the coordinates to use, e.g. "zyx" or "xyz".
    """

    experiment_dir = Path(experiment_dir)

    # Directories
    log_dir = experiment_dir / "logs"
    checkpoint_dir = experiment_dir / "checkpoints"
    # Make directories
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Name the checkpoints
    checkpoint_basename = str(checkpoint_dir / "model")
    # Summary writer
    summary_writer = SummaryWriter(log_dir=log_dir)

    # Metadata
    coordinate_order = "xyz"
    voxel_size = voxel_size
    input_size = np.array(input_shape) * np.array(voxel_size)
    cleft_size_thresh = 5

    logging.info("Reading data...")
    df = pd.read_feather(file_path)
    df = df[df['size'] >= cleft_size_thresh] # Tuning parameter.
    locations = df[list(coordinate_order)].values
    point_ids = df[point_id].values
    nt_gt_list = df["neurotransmitter"].values
    cell_type_gt_list = df["cell_type"].values

    # Load the validation ground truth
    valdf = pd.read_feather(test_file_path)
    valdf = valdf[valdf['size'] >= cleft_size_thresh]
    val_locations = valdf[list(coordinate_order)].values
    val_point_ids = valdf[point_id].values
    val_nt_gt_list = valdf["neurotransmitter"].values
    val_cell_type_gt_list = valdf["cell_type"].values

    # Make sure that we have the right number of transmitters
    assert len(df["neurotransmitter"].unique()) == num_transmitters

    logging.info(f"Input shape (voxels): {input_shape}")
    logging.info(f"Voxel size: {voxel_size}")
    logging.info(f"Input size (nm): {input_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nt_list_sorted = ['acetylcholine', 'dopamine', 'gaba', 'glutamate', 'histamine', 'octopamine', 'serotonin', 'tyramine']
    class_counts = []
    for nt in nt_list_sorted:
        class_counts.append(len(df[df[nt_name]==nt]))
    class_counts = np.array(class_counts)
    num_classes = len(class_counts)

    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts + 1e-9)

    # Overwrite alpha weights used (Manual Approach)
    weights = np.array([1.0,1.1,1.0,1.0,0.9,1.0,1.1,1.0])

    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logging.info(f"Alpha weights: {alpha_weights}")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.train()
    num_gpus = torch.cuda.device_count()

    z = zarr.open(container, mode="r")[dataset]

    assert isinstance(z, zarr.core.Array)
    # Get datatype
    datatype = z.dtype
    max_value = np.iinfo(datatype).max

    transform = tio.Compose([ # This rescales/normalizes
        tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 255)),  # First scale to [0, 1]
        tio.Lambda(lambda x: ((x - 0.5) / 0.5).float())  # Then apply your normalization
    ])

    # Different sets of augmentations during training
    aug_transforms = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), translation=3, degrees=(0,0,90), p=0.4),
        tio.RandomFlip(axes=(0,1), p=0.3),
        tio.RandomNoise(std=0.05, p=0.2),
    ])
    super_aug_transforms = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), translation=3, degrees=(0,0,180), p=0.5),
        tio.RandomFlip(axes=(0,1), p=0.5),
        tio.RandomNoise(std=0.05, p=0.5),
    ])
    super_super_aug_transforms = tio.Compose([
        tio.RandomAffine(scales=(0.8, 1.2), translation=5, degrees=(0,0,180), p=0.8),
        tio.RandomFlip(axes=(0,1), p=0.5),
        tio.RandomNoise(std=0.05, p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    ])

    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001,)
    scaler = GradScaler()

    logging.info(f"Number of classes: {num_transmitters}")
    logging.info("Setting up pipeline...")

    print('Initial stats:', len(locations), len(point_ids), len(nt_gt_list))

    # Assign the corresponding class weight to each sample, Calculate class counts in the training set
    class_sample_counts = np.array(
        [np.sum(nt_gt_list == i) for i in range(num_classes)] # num_classes = 8
    )
    class_weights = 1. / (class_sample_counts + 1e-9)
    sample_weights = np.array([class_weights[t] for t in nt_gt_list])
    sample_weights_tensor = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    zarr_dataset = ZarrDataset(
        z, locations, point_ids, nt_gt_list=nt_gt_list, cell_type_gt_list=cell_type_gt_list, shape=input_shape, transform=transform, voxel_size=voxel_size, train=True,
        aug_transforms=aug_transforms, super_aug_transforms=super_aug_transforms, super_super_aug_transforms=super_super_aug_transforms,
    )

    val_zarr_dataset = ZarrDataset(
        z, val_locations, val_point_ids, nt_gt_list=val_nt_gt_list, cell_type_gt_list=val_cell_type_gt_list, shape=input_shape, transform=transform, voxel_size=voxel_size
    )

    dataloader = torch.utils.data.DataLoader(
        zarr_dataset,
        batch_size=16,
        drop_last=False,
        sampler=sampler, # Use sampler
        num_workers=8,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_zarr_dataset,
        batch_size=16,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers * num_gpus,
    )

    epoch = 0 #60000 + 1 #400000+1 #80000 + 1
    for _ in range(400000000):
        for inputs, ids, labels, cell_types in tqdm(dataloader, total=len(dataloader)):
            model.train()

            running_total = 0
            running_correct = 0
            running_loss = 0

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 1, 4, 2, 3)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Report training accuracy and loss
            if epoch % 500 == 0:
                if summary_writer:
                    summary_writer.add_scalar(
                        "accuracy", running_correct / running_total, epoch
                    )
                    summary_writer.add_scalar(
                        "loss", running_loss, epoch
                    )

            epoch += 1
            if epoch % 10000 == 0: # Exit if epoch reaches limit for validation step
                break

        # Report testing accuracy
        if epoch % 10000 == 0 and epoch != 0:
            model.eval()

            incorrect_by_cell_type = {}
            eval_running_total = 0
            eval_running_correct = 0
            eval_running_loss = 0
            per_nt_total = [0,0,0,0,0,0,0,0]
            per_nt_correct = [0,0,0,0,0,0,0,0]
            with torch.no_grad():
                for inputs, ids, labels, cell_types in tqdm(val_dataloader, total=len(val_dataloader)):
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.permute(0, 1, 4, 2, 3)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    eval_running_loss += loss.item()

                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    eval_running_total += labels.size(0)
                    eval_running_correct += (predicted == labels).sum().item()

                    for i in range(len(labels)):
                        if labels[i] == predicted[i]:
                            per_nt_correct[labels[i]] += 1
                        per_nt_total[labels[i]] += 1

                    # Count incorrect predictions
                    incorrect_mask = predicted != labels
                    for i in range(len(labels)):
                        if incorrect_mask[i]:
                            ct = cell_types[i]
                            if ct not in incorrect_by_cell_type:
                                incorrect_by_cell_type[ct] = 0
                            incorrect_by_cell_type[ct] += 1

                if summary_writer:
                    summary_writer.add_scalar(
                        "valaccuracy", eval_running_correct / eval_running_total, epoch
                    )
                    summary_writer.add_scalar(
                        "valloss", eval_running_loss / len(val_dataloader), epoch
                    )
                per_nt_acc = [per_nt_correct[i]/per_nt_total[i] for i in range(len(per_nt_total))]
                print('valaccuracy per nt:', per_nt_acc)
                print('valaccuracy:', eval_running_correct / eval_running_total)
                print('valloss:', eval_running_loss / len(val_dataloader))

            # Save checkpoint
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f'model_checkpoint_{str(epoch)}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_checkpoint_{str(epoch)}.pth'))

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Remove logging from gunpowder
    logging.getLogger("gunpowder").setLevel(logging.WARNING)
    warnings.filterwarnings('ignore')

    # Load GPU and get device information
    device = "cuda" if torch.cuda.is_available() else "cpu"
    i=0
    print(f"  Device Name: {torch.cuda.get_device_name(i)}")
    print(f"  Capability: {torch.cuda.get_device_capability(i)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

    # Load model and checkpoint if needed
    model = resnet_18_model(num_classes=8)
    model = torch.compile(model)

    # Restart from checkpoint if needed
    #model.load_state_dict(torch.load("/n/data3_vast/hms/neurobio/htem2/users/kd193/banc_neurotransmitter_pred/synister/results/resnet_18_new_v9/checkpoints/model_checkpoint_60000.pth", map_location=device))

    # Train the model
    train(
        file_path = cfg.gt.train,
        test_file_path = cfg.gt.val,
        model=model,
        valmodel=None,
        point_id=cfg.validate.point_id,
        nt_name=cfg.validate.nt_name,
        **cfg.train,
    )


if __name__ == "__main__":
    main()
