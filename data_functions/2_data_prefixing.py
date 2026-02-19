"""
Dataset utilities for Event Log Prediction

This module handles:
1. Loading preprocessed event log data
2. Creating prefix sequences for training
3. Batching and padding
4. Data augmentation (optional)

TODO OPTIMIZATION NOTES:
+ Prefix generation: Creates multiple training examples from each case
X Caching: Stores processed sequences to speed up training -- FAISS
? Augmentation: Helps prevent overfitting on small datasets
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import joblib


class EventLogDataset(Dataset):
    """
    Creates prefix sequences from event logs:
    - For a case with events [A, B, C, D, E]
    - Creates prefixes: [A], [A,B], [A,B,C], [A,B,C,D]
    - Each prefix predicts the next activity and remaining time
    """

    def __init__(
        self,
        data_path: str,
        min_prefix_length: int = 2,
        max_prefix_length: int = 20,
        max_sequence_length: int = 20,
        augment: bool = False,
        augment_prob: float = 0.1,
    ):
        """
        Args:
            data_path: Path to preprocessed CSV file
            min_prefix_length: Minimum events before prediction
            max_prefix_length: Maximum prefix length to consider
            max_sequence_length: Maximum sequence length for padding
            augment: Whether to apply data augmentation
            augment_prob: Probability of augmentation
        """
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length
        self.max_sequence_length = max_sequence_length
        self.augment = augment
        self.augment_prob = augment_prob

        # Load data
        print(f"Data Path: {data_path}")
        self.df = pd.read_csv(data_path)

        # Feature columns (from data_preproc.py preprocessing pipeline)
        self.feature_columns = [
            # Time deltas
            'accumulated_time', 'remaining_time',
            # Event-level temporal features (9)
            'day_of_month', 'day_of_week', 'hour_of_day',
            'min_of_hour', 'sec_of_min', 'week_of_year',
            'month_of_year', 'day_of_year', 'secs_within_day',
            # Activity statistics
            'avg_duration_activity', 'std_duration_activity',
            # Time cycle features
            'hour_sin', 'hour_cos',
            # Business context
            'is_business_hours',
            # Workload features
            'concurrent_cases', 'workload_ratio',
            # Case dynamics
            'velocity', 'acceleration',
        ]

        # Generate prefix sequences

        self.sequences = self._generate_prefixes()
        print(f"Generated {len(self.sequences)} prefix sequences")

    def _generate_prefixes(self) -> List[Dict]:
        """
        For each case, creates multiple prefixes of increasing length.
        Each prefix is used to predict the next activity and remaining time.

        Returns:
            List of sequence dictionaries
        """
        sequences = []

        # Group by case
        for case_id, case_df in self.df.groupby('case_id'):
            case_df = case_df.sort_values('timestamp').reset_index(drop=True)
            case_length = len(case_df)

            # Skip very short cases
            if case_length < self.min_prefix_length:
                continue

            # Generate prefixes
            max_prefix = min(case_length - 1, self.max_prefix_length)
            for prefix_length in range(self.min_prefix_length, max_prefix + 1):
                # Extract prefix
                prefix = case_df.iloc[:prefix_length]

                # Target: next activity and remaining time
                next_event = case_df.iloc[prefix_length]
                next_activity = next_event['activity_encoded']
                remaining_time = next_event['remaining_time']

                # Extract features
                activities = prefix['activity_encoded'].values
                features = prefix[self.feature_columns].values

                sequences.append({
                    'case_id': case_id,
                    'prefix_length': prefix_length,
                    'activities': activities,
                    'features': features,
                    'next_activity': next_activity,
                    'remaining_time': remaining_time,
                })

        return sequences

    def _augment_sequence(self, activities: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to prevent overfitting.

        Augmentation techniques (ICCV 2021 Simple Feature Augmentation):
        1. Gaussian noise to features (slight time variations)
        2. Random masking of activities (forces model to be robust)

        Args:
            activities: Activity sequence
            features: Feature sequence

        Returns:
            Augmented activities and features
        """
        if not self.augment or np.random.rand() > self.augment_prob:
            return activities, features

        # Add Gaussian noise to features (small perturbations)
        noise = np.random.randn(*features.shape) * 0.05  # 5% noise
        features = features + noise

        # Random masking (replace random activities with 0)
        if np.random.rand() < 0.3:  # 30% chance of masking
            mask_idx = np.random.choice(len(activities), size=max(1, len(activities) // 5), replace=False)
            activities = activities.copy()
            activities[mask_idx] = 0  # 0 is padding/unknown

        return activities, features

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.

        Returns:
            Dictionary with:
                - activities: Activity sequence (padded)
                - features: Feature sequence (padded)
                - length: Actual sequence length
                - next_activity: Target activity
                - remaining_time: Target remaining time
        """
        seq = self.sequences[idx]

        activities = seq['activities']
        features = seq['features']
        length = len(activities)

        # Apply augmentation
        activities, features = self._augment_sequence(activities, features)

        # Pad or truncate to max_sequence_length
        if length > self.max_sequence_length:
            # Truncate from the beginning (keep most recent events)
            # OPTIMIZATION: Recent events usually more informative
            activities = activities[-self.max_sequence_length:]
            features = features[-self.max_sequence_length:]
            length = self.max_sequence_length
        else:
            # Pad with zeros
            pad_length = self.max_sequence_length - length
            activities = np.pad(activities, (0, pad_length), mode='constant', constant_values=0)
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

        return {
            'activities': torch.LongTensor(activities),
            'features': torch.FloatTensor(features),
            'length': torch.LongTensor([length]),
            'next_activity': torch.LongTensor([seq['next_activity']]),
            'remaining_time': torch.FloatTensor([seq['remaining_time']]),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Stacks sequences into batches, handling variable lengths.
    """
    # Sort by length (longest first) for packed sequences
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    # Stack tensors
    activities = torch.stack([item['activities'] for item in batch])
    features = torch.stack([item['features'] for item in batch])
    lengths = torch.cat([item['length'] for item in batch])
    next_activities = torch.cat([item['next_activity'] for item in batch])
    remaining_times = torch.cat([item['remaining_time'] for item in batch])

    return {
        'activities': activities,
        'features': features,
        'lengths': lengths,
        'next_activities': next_activities,
        'remaining_times': remaining_times,
    }


def create_data_loaders(
    train_path: str,
    test_path: str,
    config,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and test data loaders.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        config: Configuration object

    Returns:
        train_loader, test_loader, num_activities
    """
    # Create datasets
    train_dataset = EventLogDataset(
        train_path,
        min_prefix_length=config.MIN_PREFIX_LENGTH,
        max_prefix_length=config.MAX_PREFIX_LENGTH,
        max_sequence_length=config.MAX_SEQUENCE_LENGTH,
        augment=config.USE_DATA_AUGMENTATION,
        augment_prob=config.AUGMENTATION_PROB,
    )

    test_dataset = EventLogDataset(
        test_path,
        min_prefix_length=config.MIN_PREFIX_LENGTH,
        max_prefix_length=config.MAX_PREFIX_LENGTH,
        max_sequence_length=config.MAX_SEQUENCE_LENGTH,
        augment=False,  # No augmentation for test set
    )

    # Get number of unique activities
    encoder_path = config.DATA_DIR / f"{config.DATASET_NAME}_label_encoder.pkl"
    encoder = joblib.load(encoder_path)
    num_activities = len(encoder.classes_)

    print(f"\nDataset Statistics:")
    print(f"  Number of activities: {num_activities}")
    print(f"  Training sequences: {len(train_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")
    print(f"  Max sequence length: {config.MAX_SEQUENCE_LENGTH}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, test_loader, num_activities


class CaseDataset(Dataset):
    """
    Dataset that returns complete cases (not prefixes).Use: Inference to avoid redundant predictions
    """

    def __init__(
        self,
        data_path: str,
        max_sequence_length: int = 20,
    ):
        """
        Initialize case dataset.

        Args:
            data_path: Path to preprocessed CSV
            max_sequence_length: Maximum sequence length
        """
        self.max_sequence_length = max_sequence_length

        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)

        # Feature columns (from data_preproc.py preprocessing pipeline)
        self.feature_columns = [
            # Time deltas
            'accumulated_time', 'remaining_time',
            # Event-level temporal features (9)
            'day_of_month', 'day_of_week', 'hour_of_day',
            'min_of_hour', 'sec_of_min', 'week_of_year',
            'month_of_year', 'day_of_year', 'secs_within_day',
            # Activity statistics
            'avg_duration_activity', 'std_duration_activity',
            # Time cycle features
            'hour_sin', 'hour_cos',
            # Business context
            'is_business_hours',
            # Workload features
            'concurrent_cases', 'workload_ratio',
            # Case dynamics
            'velocity', 'acceleration',
        ]

        # Extract complete cases
        print("Extracting cases...")
        self.cases = self._extract_cases()
        print(f"Extracted {len(self.cases)} cases")

    def _extract_cases(self) -> List[Dict]:
        """Extract complete cases from dataframe."""
        cases = []

        for case_id, case_df in self.df.groupby('case_id'):
            case_df = case_df.sort_values('timestamp').reset_index(drop=True)

            # Extract data
            activities = case_df['activity_encoded'].values
            features = case_df[self.feature_columns].values
            activity_names = case_df['activity'].values

            cases.append({
                'case_id': case_id,
                'activities': activities,
                'features': features,
                'activity_names': activity_names,
                'length': len(activities),
            })

        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict:
        """Get a complete case."""
        case = self.cases[idx]

        activities = case['activities']
        features = case['features']
        length = len(activities)

        # Pad or truncate
        if length > self.max_sequence_length:
            activities = activities[:self.max_sequence_length]
            features = features[:self.max_sequence_length]
            length = self.max_sequence_length
        else:
            pad_length = self.max_sequence_length - length
            activities = np.pad(activities, (0, pad_length), mode='constant', constant_values=0)
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

        return {
            'case_id': case['case_id'],
            'activities': torch.LongTensor(activities),
            'features': torch.FloatTensor(features),
            'length': torch.LongTensor([length]),
            'activity_names': case['activity_names'],
        }


def get_dataset_statistics(data_path: str) -> Dict:
    """
    Compute dataset statistics for optimization guidance.

    Returns statistics that help choose hyperparameters:
    - Sequence length distribution
    - Activity distribution
    - Temporal patterns

    OPTIMIZATION: Use these stats to set:
    - MAX_SEQUENCE_LENGTH (e.g., 95th percentile)
    - Class weights (for imbalanced activities)
    - Normalization parameters
    """
    df = pd.read_csv(data_path)

    # Case statistics
    case_lengths = df.groupby('case_id').size()

    # Activity statistics
    activity_counts = df['activity_encoded'].value_counts()

    stats = {
        'num_cases': df['case_id'].nunique(),
        'num_events': len(df),
        'num_activities': df['activity_encoded'].nunique(),
        'avg_case_length': case_lengths.mean(),
        'median_case_length': case_lengths.median(),
        'min_case_length': case_lengths.min(),
        'max_case_length': case_lengths.max(),
        'p95_case_length': case_lengths.quantile(0.95),
        'p99_case_length': case_lengths.quantile(0.99),
        'activity_distribution': activity_counts.to_dict(),
        'imbalance_ratio': activity_counts.max() / activity_counts.min(),
    }

    return stats


