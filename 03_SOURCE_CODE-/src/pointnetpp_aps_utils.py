
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


CLASSES = ["boxes", "cylinders", "spheres"]
LABEL_MAP = {"boxes": 0, "cylinders": 1, "spheres": 2}
LABEL_NAME = {0: "box", 1: "cylinder", 2: "sphere"}


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_xyz(xyz):
    xyz = xyz.astype(np.float32)
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale
    return xyz


def random_rotation_matrix(rng):
    ax, ay, az = rng.uniform(0, 2 * np.pi, size=3)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


def make_point_record(xyz, normals):
    xyz = normalise_xyz(xyz)

    normals = normals.astype(np.float32)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    dummy = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, normals, dummy], axis=1)


def sample_box(n_points, rng):
    lx, ly, lz = rng.uniform(0.6, 2.0, size=3)

    face_areas = np.array([
        ly * lz, ly * lz,
        lx * lz, lx * lz,
        lx * ly, lx * ly
    ])
    face_probs = face_areas / face_areas.sum()
    faces = rng.choice(6, size=n_points, p=face_probs)

    xyz = np.zeros((n_points, 3), dtype=np.float32)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    for i, face in enumerate(faces):
        x = rng.uniform(-lx / 2, lx / 2)
        y = rng.uniform(-ly / 2, ly / 2)
        z = rng.uniform(-lz / 2, lz / 2)

        if face == 0:
            x = lx / 2
            n = [1, 0, 0]
        elif face == 1:
            x = -lx / 2
            n = [-1, 0, 0]
        elif face == 2:
            y = ly / 2
            n = [0, 1, 0]
        elif face == 3:
            y = -ly / 2
            n = [0, -1, 0]
        elif face == 4:
            z = lz / 2
            n = [0, 0, 1]
        else:
            z = -lz / 2
            n = [0, 0, -1]

        xyz[i] = [x, y, z]
        normals[i] = n

    R = random_rotation_matrix(rng)
    xyz = xyz @ R.T
    normals = normals @ R.T

    return make_point_record(xyz, normals)


def sample_cylinder(n_points, rng):
    radius = rng.uniform(0.4, 1.0)
    height = rng.uniform(0.8, 2.2)

    side_area = 2 * np.pi * radius * height
    cap_area = np.pi * radius * radius
    probs = np.array([side_area, cap_area, cap_area])
    probs = probs / probs.sum()

    parts = rng.choice(3, size=n_points, p=probs)

    xyz = np.zeros((n_points, 3), dtype=np.float32)
    normals = np.zeros((n_points, 3), dtype=np.float32)

    for i, part in enumerate(parts):
        theta = rng.uniform(0, 2 * np.pi)

        if part == 0:
            z = rng.uniform(-height / 2, height / 2)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            n = [np.cos(theta), np.sin(theta), 0]
        else:
            r = radius * np.sqrt(rng.uniform(0, 1))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            if part == 1:
                z = height / 2
                n = [0, 0, 1]
            else:
                z = -height / 2
                n = [0, 0, -1]

        xyz[i] = [x, y, z]
        normals[i] = n

    R = random_rotation_matrix(rng)
    xyz = xyz @ R.T
    normals = normals @ R.T

    return make_point_record(xyz, normals)


def sample_sphere(n_points, rng):
    radius = rng.uniform(0.6, 1.2)

    u = rng.uniform(-1, 1, size=n_points)
    theta = rng.uniform(0, 2 * np.pi, size=n_points)

    x = radius * np.sqrt(1 - u ** 2) * np.cos(theta)
    y = radius * np.sqrt(1 - u ** 2) * np.sin(theta)
    z = radius * u

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    normals = xyz / radius

    R = random_rotation_matrix(rng)
    xyz = xyz @ R.T
    normals = normals @ R.T

    return make_point_record(xyz, normals)


def generate_clean_primitive_dataset(output_root, n_per_class=200, n_points=1000, seed=100):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    generators = {
        "boxes": sample_box,
        "cylinders": sample_cylinder,
        "spheres": sample_sphere,
    }

    for class_folder, generator in generators.items():
        class_dir = output_root / class_folder
        class_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(class_dir.glob("*.csv"))
        if len(existing_files) >= n_per_class:
            print(f"{class_folder}: already exists, skipping generation.")
            continue

        for i in range(n_per_class):
            data = generator(n_points, rng)
            file_path = class_dir / f"{class_folder[:-1]}_{i:04d}.csv"
            np.savetxt(file_path, data, delimiter=",", fmt="%.8f")


def make_noisy_variant_from_clean(clean_root, output_root, noise_level=0.025, seed=200):
    clean_root = Path(clean_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for class_folder in CLASSES:
        clean_class_dir = clean_root / class_folder
        output_class_dir = output_root / class_folder
        output_class_dir.mkdir(parents=True, exist_ok=True)

        clean_files = sorted(clean_class_dir.glob("*.csv"))

        for file_path in clean_files:
            data = np.loadtxt(file_path, delimiter=",").astype(np.float32)
            noisy = data.copy()

            xyz_noise = rng.uniform(
                low=-noise_level,
                high=noise_level,
                size=noisy[:, 0:3].shape
            ).astype(np.float32)

            noisy[:, 0:3] = noisy[:, 0:3] + xyz_noise

            output_path = output_class_dir / file_path.name
            np.savetxt(output_path, noisy, delimiter=",", fmt="%.8f")


def collect_dataset_dataframe(root_dir, variant, role, source):
    root_dir = Path(root_dir)
    records = []

    for class_folder in CLASSES:
        class_dir = root_dir / class_folder
        files = sorted(class_dir.glob("*.csv"))

        for file_path in files:
            records.append({
                "path": str(file_path),
                "class_folder": class_folder,
                "class_name": LABEL_NAME[LABEL_MAP[class_folder]],
                "label": LABEL_MAP[class_folder],
                "variant": variant,
                "role": role,
                "source": source,
                "file_name": file_path.name
            })

    return pd.DataFrame(records)


class APSPointCloudDataset(Dataset):
    def __init__(self, dataframe, n_points=1000, use_normals=True, normalise=True):
        self.df = dataframe.reset_index(drop=True).copy()
        self.n_points = n_points
        self.use_normals = use_normals
        self.normalise = normalise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])

        data = np.loadtxt(path, delimiter=",").astype(np.float32)

        xyz = data[:, 0:3]

        if self.normalise:
            xyz = normalise_xyz(xyz)

        if self.use_normals:
            normals = data[:, 3:6]

            norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
            norm_len[norm_len == 0] = 1.0
            normals = normals / norm_len

            points = np.concatenate([xyz, normals], axis=1)
        else:
            points = xyz

        if points.shape[0] >= self.n_points:
            choice = np.random.choice(points.shape[0], self.n_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.n_points, replace=True)

        points = points[choice, :]

        points = torch.tensor(points, dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(label, dtype=torch.long)

        return points, label


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)

    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)

    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape

    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)

        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points
            )

        new_points = new_points.permute(0, 3, 2, 1)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = torch.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]

        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetPPClassifier(nn.Module):
    def __init__(self, num_classes=3, normal_channel=True):
        super().__init__()

        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.30,
            nsample=32,
            in_channel=6 if normal_channel else 3,
            mlp=[64, 64, 128],
            group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.60,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        if self.normal_channel:
            xyz = x[:, 0:3, :]
            normals = x[:, 3:6, :]
        else:
            xyz = x[:, 0:3, :]
            normals = None

        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(l3_points.size(0), 1024)

        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x
