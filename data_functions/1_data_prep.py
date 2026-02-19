"""
Data Preprocessing Module for event log attributes

Pipeline steps:
1. Convert XES to CSV
2. Data Engineering (Oyamada et al.) + Feature Engineering (Oyamada et al. + additional stats)
3. Synthetic Data
4. Evaluation splits and scenarios


Before running edit the paths in TODO
"""

import pandas as pd
import numpy as np
import joblib
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from collections import Counter
import pm4py
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: XES TO CSV CONVERSION
# =============================================================================

class XESConverter:
    def __init__(self, raw_csv_dir: str = "raw_csv"):
        self.raw_csv_dir = Path(raw_csv_dir)
        self.raw_csv_dir.mkdir(exist_ok=True)

    def convert(self, xes_filepath: str, dataset_name: str) -> str:
        """
        Args:
            xes_filepath: Path to the XES file
            dataset_name: Name identifier for the dataset

        Returns:
            Path to the saved CSV file
        """
        # print(f"  Loading XES file: {xes_filepath}")
        log = pm4py.read_xes(xes_filepath)
        df = pm4py.convert_to_dataframe(log)

        # Standardize column names
        df = df.rename(columns={
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp'
        })

        # Save as CSV
        csv_path = self.raw_csv_dir / f"{dataset_name}_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved raw CSV: {csv_path}")
        # print(f"  {len(df)} events from {df['case_id'].nunique()} cases")

        return str(csv_path)

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load event log from XES or CSV file.

        Args:
            filepath: Path to XES or CSV file

        Returns:
            DataFrame with columns: case_id, activity, timestamp
        """
        if filepath.endswith('.xes'):
            log = pm4py.read_xes(filepath)
            df = pm4py.convert_to_dataframe(log)
            df = df.rename(columns={
                'case:concept:name': 'case_id',
                'concept:name': 'activity',
                'time:timestamp': 'timestamp'
            })
        else:
            df = pd.read_csv(filepath)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        return df


# =============================================================================
# STEP 2: DATA CLEANING AND ADDITONAL FEATURE COLUMNS
# =============================================================================

class DataCleaner:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None

    def filter_short_cases(self, df: pd.DataFrame, min_length: int = 2) -> pd.DataFrame:
        """
        Filter out cases <= min_length events.

        Short cases don't provide enough context for prediction.

        Args:
            df: Event log DataFrame
            min_length: Minimum number of events per case

        Returns:
            Filtered DataFrame
        """
        case_lengths = df.groupby('case_id').size()
        valid_cases = case_lengths[case_lengths > min_length].index
        filtered_df = df[df['case_id'].isin(valid_cases)].copy()

        removed = len(case_lengths) - len(valid_cases)
        print(f"  Filtered: kept {len(valid_cases)} cases, removed {removed} short cases")
        print(f"  Events remaining: {len(filtered_df)}")
        return filtered_df

    def sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by CaseID and timestamp.

        Ensures events within each case are in chronological order.
        """
        df = df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        # print(f"  Sorted {len(df)} events by case_id and timestamp")
        return df

    def extract_temporal_features(self, df: pd.DataFrame, time_unit: str = 'days') -> pd.DataFrame:
        """

        Extracts:
            accumulated_time: time since case start
            remaining_time:   time until case completion (target)
            Event-level features: day_of_month, day_of_week, hour_of_day,
                min_of_hour, sec_of_min, week_of_year, month_of_year,
                day_of_year, secs_within_day

        Args:
            df: Sorted event log DataFrame
            time_unit: Unit for time deltas ('seconds', 'minutes', 'hours', 'days')

        Returns:
            DataFrame with temporal features added
        """
        #df_copy = df.copy()

        # ----- Event-level features (9) -----
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['min_of_hour'] = df['timestamp'].dt.minute
        df['sec_of_min'] = df['timestamp'].dt.second
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
        df['month_of_year'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['secs_within_day'] = (df['timestamp'].dt.hour * 3600 +
                                 df['timestamp'].dt.minute * 60 +
                                 df['timestamp'].dt.second)


        case_groups = df.groupby('case_id')
        case_start = case_groups['timestamp'].transform('first')
        case_end = case_groups['timestamp'].transform('last')

        df['accumulated_time'] = (df['timestamp'] - case_start).dt.total_seconds()
        df['remaining_time'] = (case_end - df['timestamp']).dt.total_seconds()

        # Convert to specified unit
        time_divisor = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400
        }[time_unit]
        df['accumulated_time'] /= time_divisor
        df['remaining_time'] /= time_divisor

        print(f"  Extracted 9 event-level features + accumulated_time + remaining_time")
        return df

    def extract_activity_stats(self, df: pd.DataFrame, time_unit: str = 'days') -> pd.DataFrame:
        """
        Activity stats:
            avg_duration[activity]: mean duration per activity type
            std_duration[activity]: std deviation of duration per activity type

        These capture how long each activity type typically takes,
        providing the model with activity-level temporal context.

        Args:
            df: DataFrame with timestamp column
            time_unit: Unit for duration calculation

        Returns:
            DataFrame with activity duration stats added
        """
        df = df.copy()
        time_divisor = {
            'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400
        }[time_unit]

        # Compute per-event duration (time to next event within same case)
        df['event_duration'] = df.groupby('case_id')['timestamp'].diff().dt.total_seconds() / time_divisor
        df['event_duration'] = df['event_duration'].fillna(0)

        # avg_duration[activity]: mean duration per activity type
        activity_avg = df.groupby('activity')['event_duration'].transform('mean')
        df['avg_duration_activity'] = activity_avg

        # std_duration[activity]: std deviation of duration per activity type
        activity_std = df.groupby('activity')['event_duration'].transform('std')
        df['std_duration_activity'] = activity_std.fillna(0)

        # Drop intermediate column
        df = df.drop(columns=['event_duration'])

        # print(f"  Extracted activity stats: avg_duration, std_duration per activity")
        return df

    def extract_time_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
       Time cycle features:
            hour_sin: sine encoding of hour (captures cyclical nature)
            hour_cos: cosine encoding of hour

        Cyclical encoding to ensure hours 23 and 0 are close

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with cyclical time features added
        """
        df = df.copy()

        # hour_sin, hour_cos: cyclical encoding of hour
        hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        print(f"  Extracted time cycle features: hour_sin, hour_cos")
        return df

    def extract_business_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        is_business_hours: 1 if weekday (Mon-Fri) and 9:00-17:00, else 0.

        Business hours context helps the model distinguish between events
        processed during peak vs off-peak times.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with is_business_hours column added
        """
        df = df.copy()

        hour = df['timestamp'].dt.hour
        weekday = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

        # is_business_hours: weekday 9-17
        df['is_business_hours'] = ((weekday < 5) & (hour >= 9) & (hour < 17)).astype(int)

        print(f"  Extracted business hours feature")
        return df

    def extract_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute workload features.

        Workloads:
            concurrent_cases: number of active cases at each event's timestamp.
                More concurrent cases = slower processing.
            workload_ratio: ratio of concurrent cases to average concurrent cases.
                Values > 1 indicate above-average load.

        Args:
            df: DataFrame with case_id and timestamp columns

        Returns:
            DataFrame with workload features added
        """
        df = df.copy()

        # Build case start/end lookup
        case_spans = df.groupby('case_id')['timestamp'].agg(['min', 'max'])
        case_starts = case_spans['min'].values
        case_ends = case_spans['max'].values

        # concurrent_cases: count of active cases at each event time
        # A case is active if its start <= event_time <= its end
        concurrent = []
        for _, row in df.iterrows():
            t = row['timestamp']
            count = np.sum((case_starts <= t) & (case_ends >= t))
            concurrent.append(count)

        df['concurrent_cases'] = concurrent
        avg_concurrent = df['concurrent_cases'].mean()

        # workload_ratio: concurrent / average concurrent
        df['workload_ratio'] = df['concurrent_cases'] / avg_concurrent if avg_concurrent > 0 else 1.0

        print(f"  Extracted workload features: concurrent_cases, workload_ratio")
        print(f"    Average concurrent cases: {avg_concurrent:.1f}")
        return df

    def extract_workload_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with case_id and timestamp columns

        Returns:
            DataFrame with workload features added
        """
        df = df.copy()

        # Build case start/end times
        case_spans = df.groupby('case_id')['timestamp'].agg(['min', 'max'])

        # Sweep-line: create events for case start (+1) and case end (-1)
        starts = pd.DataFrame({'time': case_spans['min'], 'delta': 1})
        ends = pd.DataFrame({'time': case_spans['max'], 'delta': -1})
        events = pd.concat([starts, ends]).sort_values('time').reset_index(drop=True)
        events['concurrent'] = events['delta'].cumsum()

        # For each event timestamp, find the concurrent case count
        # Use merge_asof for efficient lookup
        df_sorted = df.sort_values('timestamp').copy()
        events_sorted = events.sort_values('time')

        merged = pd.merge_asof(
            df_sorted[['timestamp']].reset_index(),
            events_sorted.rename(columns={'time': 'timestamp'})[['timestamp', 'concurrent']],
            on='timestamp',
            direction='backward'
        ).set_index('index')

        df['concurrent_cases'] = merged['concurrent'].reindex(df.index).fillna(1).astype(int)
        avg_concurrent = df['concurrent_cases'].mean()

        # workload_ratio: concurrent / average concurrent
        df['workload_ratio'] = df['concurrent_cases'] / avg_concurrent if avg_concurrent > 0 else 1.0

        print(f"  Extracted workload features (fast): concurrent_cases, workload_ratio")
        print(f"    Average concurrent cases: {avg_concurrent:.1f}")
        return df

    def extract_case_dynamics(self, df: pd.DataFrame, time_unit: str = 'days') -> pd.DataFrame:
        """
        TODO Compute case-level dynamics.

        Case dynamics:
            velocity: events_per_time_unit
                Measures how fast events are being processed in a case.
            acceleration: change in velocity over the case.
                Fast events tend to continue fast, slow events tend to stay slow.

        These features capture the momentum of a case's progression.

        Args:
            df: DataFrame sorted by case_id and timestamp
            time_unit: Unit for velocity calculation

        Returns:
            DataFrame with velocity and acceleration columns added
        """
        df = df.copy()
        time_divisor = {
            'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400
        }[time_unit]

        velocities = []
        accelerations = []

        for case_id, case_df in df.groupby('case_id'):
            case_idx = case_df.index
            timestamps = case_df['timestamp'].values

            case_vel = []
            case_acc = []

            for i in range(len(case_df)):
                if i == 0:
                    # First event: no velocity or acceleration yet
                    case_vel.append(0.0)
                    case_acc.append(0.0)
                else:
                    # velocity: events completed / elapsed time
                    elapsed = (timestamps[i] - timestamps[0]) / np.timedelta64(1, 's') / time_divisor
                    vel = (i + 1) / elapsed if elapsed > 0 else 0.0
                    case_vel.append(vel)

                    # acceleration: change in velocity from previous event
                    prev_vel = case_vel[i - 1]
                    acc = vel - prev_vel
                    case_acc.append(acc)

            velocities.extend(case_vel)
            accelerations.extend(case_acc)

        df['velocity'] = velocities
        df['acceleration'] = accelerations

        print(f"  Extracted case dynamics: velocity (events_per_{time_unit}), acceleration")
        return df

    def encode_activities(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          dataset_name: str, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode activity labels as integers and save encoder.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            dataset_name: Name for saving artifacts
            output_dir: Directory to save encoder

        Returns:
            Tuple of (train_df, test_df) with activity_encoded column
        """
        self.label_encoder = LabelEncoder()
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df['activity_encoded'] = self.label_encoder.fit_transform(train_df['activity'])
        test_df['activity_encoded'] = self.label_encoder.transform(test_df['activity'])

        # Save encoder
        encoder_path = output_dir / f"{dataset_name}_label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"  Encoded {len(self.label_encoder.classes_)} unique activities")
        print(f"  Saved encoder: {encoder_path}")

        return train_df, test_df

    def normalize_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           feature_columns: List[str], dataset_name: str,
                           output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standard Scaler

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_columns: List of numerical columns to normalize
            dataset_name: Name for saving artifacts
            output_dir: Directory to save scaler

        Returns:
            Tuple of (train_df, test_df) with normalized features
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        self.scaler = StandardScaler()
        train_df[feature_columns] = self.scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = self.scaler.transform(test_df[feature_columns])

        # Save scaler
        scaler_path = output_dir / f"{dataset_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"  Z-score normalized {len(feature_columns)} features")
        print(f"  Saved scaler: {scaler_path}")

        return train_df, test_df


# =============================================================================
# STEP 3: SYNTHETIC VARIATION DATASETS
# =============================================================================

class SyntheticVariationGenerator:
    """
    Generate synthetic variation datasets for experimental evaluation.

    Assigns domain IDs to cases based on structural properties (entropy,
    case length, process type) and applies entropy-increasing transformations
    to create controlled experimental scenarios.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)

    def compute_case_entropy(self, case_activities: List[str]) -> float:
        """
        Compute Shannon entropy of activity distribution within a case.

        Higher entropy = more diverse/unpredictable activity sequences.
        Lower entropy = more repetitive/predictable sequences.

        Args:
            case_activities: List of activity names in the case

        Returns:
            Shannon entropy value
        """
        if len(case_activities) <= 1:
            return 0.0

        counts = Counter(case_activities)
        total = len(case_activities)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def assign_domain_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TODO Assign domain IDs based on case properties. (Fixed vs dynamic with LLM)

        Domain IDs are assigned based on:
            - Entropy (low, med, high): Shannon entropy of activity distribution
            - Case length (short, medium, long): number of events
            - Process type: combination of entropy and length categories

        These domain IDs enable stratified evaluation and domain-aware
        model training.

        Args:
            df: Event log DataFrame with case_id and activity columns

        Returns:
            DataFrame with domain ID columns added
        """
        df = df.copy()

        # Compute per-case properties
        case_stats = df.groupby('case_id').agg(
            case_length=('activity', 'count'),
            activities=('activity', list)
        )

        # ----- Entropy (low, med, high) -----
        case_stats['entropy'] = case_stats['activities'].apply(self.compute_case_entropy)
        entropy_33 = case_stats['entropy'].quantile(0.33)
        entropy_66 = case_stats['entropy'].quantile(0.66)
        case_stats['entropy_level'] = pd.cut(
            case_stats['entropy'],
            bins=[-np.inf, entropy_33, entropy_66, np.inf],
            labels=['low', 'med', 'high']
        )

        # ----- Case length (short, medium, long) -----
        len_33 = case_stats['case_length'].quantile(0.33)
        len_66 = case_stats['case_length'].quantile(0.66)
        case_stats['length_category'] = pd.cut(
            case_stats['case_length'],
            bins=[-np.inf, len_33, len_66, np.inf],
            labels=['short', 'medium', 'long']
        )

        # ----- Process type: combination of entropy + length -----
        case_stats['process_type'] = (
            case_stats['entropy_level'].astype(str) + '_' +
            case_stats['length_category'].astype(str)
        )

        # Map back to event level
        domain_map = case_stats[['entropy_level', 'length_category', 'process_type']].to_dict()
        df['entropy_level'] = df['case_id'].map(domain_map['entropy_level'])
        df['length_category'] = df['case_id'].map(domain_map['length_category'])
        df['process_type'] = df['case_id'].map(domain_map['process_type'])

        # Numeric domain ID for model input
        process_types = df['process_type'].unique()
        type_to_id = {t: i for i, t in enumerate(sorted(process_types))}
        df['domain_id'] = df['process_type'].map(type_to_id)

        print(f"  Assigned domain IDs based on entropy x case_length")
        print(f"  Entropy levels: {case_stats['entropy_level'].value_counts().to_dict()}")
        print(f"  Length categories: {case_stats['length_category'].value_counts().to_dict()}")
        print(f"  Process types: {len(process_types)} unique combinations")

        return df

    def apply_transformation(self, case_activities: List[str],
                             all_activities: List[str],
                             transformation: str) -> List[str]:
        """
        Apply an entropy-increasing transformation to a case.

        Transformations:
            permute:  swap activity order (pairwise swaps)
            insert:   add random activities from the activity vocabulary
            skip:     skip (remove) some activities from the sequence
            repeat:   repeat some activities in place
            hybrid:   combination of the above transformations

        Args:
            case_activities: Original activity sequence
            all_activities: Full activity vocabulary for insertions
            transformation: Type of transformation to apply

        Returns:
            Transformed activity sequence
        """
        activities = list(case_activities)

        if transformation == 'permute':
            # Swap activity order: randomly swap adjacent pairs
            for i in range(len(activities) - 1):
                if self.rng.random() < 0.3:
                    activities[i], activities[i + 1] = activities[i + 1], activities[i]

        elif transformation == 'insert':
            # Add random activities at random positions
            n_inserts = max(1, len(activities) // 5)
            for _ in range(n_inserts):
                pos = self.rng.randint(0, len(activities))
                new_act = self.rng.choice(all_activities)
                activities.insert(pos, new_act)

        elif transformation == 'skip':
            # Skip (remove) some activities
            if len(activities) > 2:
                n_skip = max(1, len(activities) // 5)
                skip_indices = set(self.rng.sample(range(len(activities)), min(n_skip, len(activities) - 2)))
                activities = [a for i, a in enumerate(activities) if i not in skip_indices]

        elif transformation == 'repeat':
            # Repeat some activities in place
            result = []
            for act in activities:
                result.append(act)
                if self.rng.random() < 0.2:
                    result.append(act)
            activities = result

        elif transformation == 'hybrid':
            # Combination of the above: apply 2-3 random transformations
            transforms = self.rng.sample(['permute', 'insert', 'skip', 'repeat'],
                                         k=self.rng.randint(2, 3))
            for t in transforms:
                activities = self.apply_transformation(activities, all_activities, t)

        return activities

    def generate_synthetic_dataset(self, df: pd.DataFrame,
                                   n_synthetic_cases: Optional[int] = None,
                                   transformation_types: Optional[List[str]] = None
                                   ) -> pd.DataFrame:
        """
        Generate a synthetic variation dataset.

        Creates new cases by applying entropy-increasing transformations
        to existing cases. Each synthetic case records which transformation
        was applied for experimental analysis.

        Args:
            df: Original event log DataFrame
            n_synthetic_cases: Number of synthetic cases to generate
                (defaults to same number as original cases)
            transformation_types: List of transformations to use
                (defaults to all: permute, insert, skip, repeat, hybrid)

        Returns:
            DataFrame with synthetic cases appended (marked with is_synthetic=1)
        """
        if transformation_types is None:
            transformation_types = ['permute', 'insert', 'skip', 'repeat', 'hybrid']

        all_activities = df['activity'].unique().tolist()
        cases = df.groupby('case_id')

        if n_synthetic_cases is None:
            n_synthetic_cases = df['case_id'].nunique()

        # Sample cases to transform
        case_ids = list(cases.groups.keys())
        selected_cases = [self.rng.choice(case_ids) for _ in range(n_synthetic_cases)]

        synthetic_rows = []
        for i, case_id in enumerate(selected_cases):
            case_df = cases.get_group(case_id).copy()
            original_activities = case_df['activity'].tolist()

            # Pick a random transformation
            transformation = self.rng.choice(transformation_types)
            new_activities = self.apply_transformation(
                original_activities, all_activities, transformation
            )

            # Build synthetic case
            new_case_id = f"synthetic_{i}_{transformation}"
            for j, act in enumerate(new_activities):
                row = {
                    'case_id': new_case_id,
                    'activity': act,
                    'is_synthetic': 1,
                    'transformation': transformation,
                }
                # Carry over timestamp with small perturbation if within bounds
                if j < len(case_df):
                    row['timestamp'] = case_df.iloc[j]['timestamp']
                else:
                    # For inserted events, interpolate timestamp
                    last_ts = case_df.iloc[-1]['timestamp']
                    row['timestamp'] = last_ts + pd.Timedelta(seconds=self.rng.randint(1, 3600))

                synthetic_rows.append(row)

        synthetic_df = pd.DataFrame(synthetic_rows)
        synthetic_df['timestamp'] = pd.to_datetime(synthetic_df['timestamp'], utc=True)

        # Mark original data
        df = df.copy()
        df['is_synthetic'] = 0
        df['transformation'] = 'none'

        # Combine
        combined = pd.concat([df, synthetic_df], ignore_index=True)
        print(f"  Generated {n_synthetic_cases} synthetic cases using: {transformation_types}")
        print(f"  Total events: {len(combined)} ({len(df)} original + {len(synthetic_df)} synthetic)")

        return combined


# =============================================================================
# STEP 4: EVALUATION SPLITS AND SCENARIOS
# =============================================================================


class EvaluationSplitter:
    """
    Evaluation splits and scenarios.

    Provides unbiased train-test splitting with no temporal leakage.
    """

    def temporal_split(self, df: pd.DataFrame,
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Unbiased train-test split (no temporal leakage).


        Args:
            df: Event log DataFrame with case_id and timestamp
            test_size: Fraction of cases for the test set

        Returns:
            Tuple of (train_df, test_df)
        """
        case_start_times = df.groupby('case_id')['timestamp'].min()
        split_time = case_start_times.quantile(1 - test_size)

        train_cases = case_start_times[case_start_times <= split_time].index
        test_cases = case_start_times[case_start_times > split_time].index

        train_df = df[df['case_id'].isin(train_cases)].copy()
        test_df = df[df['case_id'].isin(test_cases)].copy()

        print(f"  Temporal split (no leakage):")
        print(f"    Train: {len(train_cases)} cases ({len(train_df)} events)")
        print(f"    Test:  {len(test_cases)} cases ({len(test_df)} events)")
        print(f"    Train date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"    Test date range:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        return train_df, test_df


# =============================================================================
# PIPELINE CONSTRUCTION
# =============================================================================


# All numerical feature columns produced by the pipeline
FEATURE_COLUMNS = [
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
    # # Case dynamics
    # 'velocity', 'acceleration',
]


class EventLogPreprocessor:
    """
    Complete data preprocessing pipeline for event log prediction.

    Orchestrates all four pipeline steps:
        Step 1: XES to CSV conversion
        Step 2: Data cleaning and normalization
        Step 3: Synthetic variation datasets (optional)
        Step 4: Evaluation splits and scenarios
    """

    def __init__(self, output_dir: str = "prep_data", raw_csv_dir: str = "raw_csv",
                 synthetic_dir: str = "synthetic_data", random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.synthetic_dir = Path(synthetic_dir)

        self.converter = XESConverter(raw_csv_dir)
        self.cleaner = DataCleaner()
        self.synth_gen = SyntheticVariationGenerator(random_seed)
        self.splitter = EvaluationSplitter()

        self.scaler = None
        self.label_encoder = None
        self.feature_columns = FEATURE_COLUMNS

    def prepare_data(self, filepath: str, dataset_name: str,
                     test_size: float = 0.2, min_case_length: int = 2,
                     time_unit: str = 'days',
                     generate_synthetic: bool = False,
                     n_synthetic_cases: Optional[int] = None,
                     use_fast_workload: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.

        Args:
            filepath: Path to raw XES or CSV file
            dataset_name: Name for saving outputs
            test_size: Fraction for test set
            min_case_length: Minimum events per case
            time_unit: Unit for time features ('seconds', 'minutes', 'hours', 'days')
            generate_synthetic: Whether to generate synthetic variation data (Step 3)
            n_synthetic_cases: Number of synthetic cases (None = same as original)
            use_fast_workload: Use vectorized workload computation (recommended for large datasets)

        Returns:
            Tuple of (train_df, test_df) with all preprocessing applied
        """
        print("\n" + "=" * 80)
        print(f"DATA PREPROCESSING PIPELINE - {dataset_name}")
        print("=" * 80)

        if filepath.endswith('.xes'):
            print("\n[Step 1] Converting XES to CSV...")
            csv_path = self.converter.convert(filepath, dataset_name)
            filepath = csv_path
        else:
            print("\n[Step 1] Loading CSV (XES conversion skipped)...")

        df = self.converter.load(filepath)
        print(f"  Loaded {len(df)} events from {df['case_id'].nunique()} cases")


        print("\n[Step 2] Data cleaning and normalization...")

        # Filter out cases <= 2 events
        print("\n  [2a] Filtering short cases...")
        df = self.cleaner.filter_short_cases(df, min_case_length)

        # CaseID and timestamp sorting
        print("\n  [2b] Sorting chronologically...")
        df = self.cleaner.sort_chronologically(df)

        # =====================================================================
        # STEP 3: Synthetic Variation Datasets (optional)
        # Synthetic data is saved separately from the cleaned dataset.
        # =====================================================================
        synthetic_train_df = None
        synthetic_test_df = None

        if generate_synthetic:
            print("\n[Step 3] Generating synthetic variation datasets...")
            self.synthetic_dir.mkdir(exist_ok=True)

            # Assign domain IDs based on entropy, case length, process type
            print("\n  [3a] Assigning domain IDs...")
            df = self.synth_gen.assign_domain_ids(df)

            # Generate synthetic cases with entropy-increasing transformations
            print("\n  [3b] Applying entropy-increasing transformations...")
            combined_df = self.synth_gen.generate_synthetic_dataset(df, n_synthetic_cases)

            # Separate synthetic from original for independent processing
            synthetic_only = combined_df[combined_df['is_synthetic'] == 1].copy()
            synthetic_only = self.cleaner.sort_chronologically(synthetic_only)

            print(f"\n  Cleaned dataset:   {df['case_id'].nunique()} cases ({len(df)} events) -> {self.output_dir}/")
            print(f"  Synthetic dataset: {synthetic_only['case_id'].nunique()} cases ({len(synthetic_only)} events) -> {self.synthetic_dir}/")
        else:
            print("\n[Step 3] Synthetic generation skipped")

        # Mark cleaned data
        df['is_synthetic'] = 0
        df['transformation'] = 'none'


        print("\n[Step 4] Evaluation splits...")

        # --- Cleaned dataset split ---
        print("\n  --- Cleaned dataset ---")
        train_df, test_df = self.splitter.temporal_split(df, test_size)

        # --- Synthetic dataset split (if generated) ---
        if generate_synthetic:
            print("\n  --- Synthetic dataset ---")
            synthetic_train_df, synthetic_test_df = self.splitter.temporal_split(synthetic_only, test_size)

        # =====================================================================
        # FEATURE EXTRACTION & NORMALIZATION - Cleaned dataset
        # =====================================================================
        print("\n[Features] Extracting features for cleaned dataset...")
        train_df, test_df = self._extract_and_normalize(
            train_df, test_df, dataset_name, self.output_dir,
            time_unit, use_fast_workload
        )

        # Store references from cleaned dataset
        self.scaler = self.cleaner.scaler
        self.label_encoder = self.cleaner.label_encoder

        # =====================================================================
        # FEATURE EXTRACTION & NORMALIZATION - Synthetic dataset
        # =====================================================================
        if generate_synthetic and synthetic_train_df is not None:
            print("\n[Features] Extracting features for synthetic dataset...")
            synthetic_train_df, synthetic_test_df = self._extract_and_normalize(
                synthetic_train_df, synthetic_test_df,
                f"{dataset_name}_synthetic", self.synthetic_dir,
                time_unit, use_fast_workload
            )

            # Save synthetic data to separate directory
            synth_train_path = self.synthetic_dir / f"{dataset_name}_synthetic_train.csv"
            synth_test_path = self.synthetic_dir / f"{dataset_name}_synthetic_test.csv"
            synthetic_train_df.to_csv(synth_train_path, index=False)
            synthetic_test_df.to_csv(synth_test_path, index=False)

            print(f"\nSaved synthetic train data: {synth_train_path}")
            print(f"Saved synthetic test data:  {synth_test_path}")

        # =====================================================================
        # SAVE CLEANED DATA
        # =====================================================================
        train_path = self.output_dir / f"{dataset_name}_train.csv"
        test_path = self.output_dir / f"{dataset_name}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nSaved cleaned train data: {train_path}")
        print(f"Saved cleaned test data:  {test_path}")
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        print("=" * 80 + "\n")

        return train_df, test_df

    def _extract_and_normalize(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               dataset_name: str, save_dir: Path,
                               time_unit: str, use_fast_workload: bool
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run feature extraction, encoding, and normalization on a train/test pair.

        Used internally to apply the same pipeline to both the cleaned dataset
        and the synthetic dataset independently.

        Args:
            train_df: Training split
            test_df: Test split
            dataset_name: Name prefix for saved artifacts
            save_dir: Directory to save encoder and scaler
            time_unit: Time unit for feature computation
            use_fast_workload: Use vectorized workload computation

        Returns:
            Tuple of (train_df, test_df) with features extracted and normalized
        """
        for label, split_df in [("train", train_df), ("test", test_df)]:
            print(f"\n  --- {label} set ---")

            # Temporal features: accumulated_time, remaining_time, 9 event-level
            split_df = self.cleaner.extract_temporal_features(split_df, time_unit)

            # Activity stats: avg_duration[activity], std_duration[activity]
            split_df = self.cleaner.extract_activity_stats(split_df, time_unit)

            # Time cycle: hour_sin, hour_cos
            split_df = self.cleaner.extract_time_cycle_features(split_df)

            # is_business_hours
            split_df = self.cleaner.extract_business_hours(split_df)

            # Workloads: concurrent_cases, workload_ratio
            if use_fast_workload:
                split_df = self.cleaner.extract_workload_features_fast(split_df)
            else:
                split_df = self.cleaner.extract_workload_features(split_df)

            # Case dynamics: velocity, acceleration
            split_df = self.cleaner.extract_case_dynamics(split_df, time_unit)

            if label == "train":
                train_df = split_df
            else:
                test_df = split_df

        # Encode activities
        print("\n  [Encode] Encoding activities...")
        train_df, test_df = self.cleaner.encode_activities(
            train_df, test_df, dataset_name, save_dir
        )

        # Z-score normalization: better outlier handling
        print("\n  [Normalize] Z-score normalization...")
        train_df, test_df = self.cleaner.normalize_features(
            train_df, test_df, self.feature_columns, dataset_name, save_dir
        )

        return train_df, test_df

    def load_artifacts(self, dataset_name: str) -> Tuple[StandardScaler, LabelEncoder]:
        """Load saved scaler and encoder for predictions."""
        scaler_path = self.output_dir / f"{dataset_name}_scaler.pkl"
        encoder_path = self.output_dir / f"{dataset_name}_label_encoder.pkl"

        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)

        print(f"Loaded scaler from: {scaler_path}")
        print(f"Loaded encoder from: {encoder_path}")

        return scaler, encoder


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_xes_files(data_dir: str = "data") -> Dict[str, str]:
    """
    Scan a directory for XES files and return a dictionary mapping
    dataset names to file paths.

    Args:
        data_dir: Directory to scan for XES files

    Returns:
        Dictionary mapping dataset names to file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Data directory '{data_dir}' does not exist!")
        return {}

    xes_files = list(data_path.glob("**/*.xes"))

    datasets = {}
    for xes_file in xes_files:
        dataset_name = xes_file.stem.replace(' ', '_').lower()
        datasets[dataset_name] = str(xes_file)

    return datasets


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "processed_data"       # Cleaned datasets saved here
    SYNTHETIC_DIR = "synthetic_data"    # Synthetic datasets saved separately
    RAW_CSV_DIR = "raw_csv"


    print("EVENT LOG PREPROCESSING PIPELINE")

    print(f"\nConfiguration:")
    print(f"  Data directory:      {DATA_DIR}")
    print(f"  Output directory:    {OUTPUT_DIR}")
    print(f"  Synthetic directory: {SYNTHETIC_DIR}")
    print(f"  Raw CSV directory:   {RAW_CSV_DIR}")


    # Initialize preprocessor
    preprocessor = EventLogPreprocessor(
        output_dir=OUTPUT_DIR,
        raw_csv_dir=RAW_CSV_DIR,
        synthetic_dir=SYNTHETIC_DIR,
    )

    # Find all XES files
    print(f"\nScanning for XES files in '{DATA_DIR}'...")
    datasets = find_xes_files(DATA_DIR)

    if not datasets:
        print(f"\nNo XES files found in '{DATA_DIR}'!")

    else:
        print(f"\nFound {len(datasets)} XES file(s):")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")

        # Process each dataset

        # print("STARTING PREPROCESSING")

        for dataset_name, filepath in datasets.items():
            try:
                print(f"\n{'=' * 80}")
                print(f"Processing: {dataset_name}")
                print(f"{'=' * 80}")
                train_df, test_df = preprocessor.prepare_data(
                    filepath,
                    dataset_name,
                    test_size=0.2,
                    min_case_length=2,
                    time_unit='days',
                    generate_synthetic=False,
                )
                print(f"\nSuccessfully processed {dataset_name}")
            except Exception as e:
                print(f"\nError processing {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()


        print("PREPROCESSING COMPLETE")

        print(f"\nOutput files:")
        print(f"  - Raw CSVs:            {RAW_CSV_DIR}/")
        print(f"  - Cleaned data:        {OUTPUT_DIR}/")
        print(f"  - Synthetic data:      {SYNTHETIC_DIR}/")
        print(f"  - Encoders & scalers:  {OUTPUT_DIR}/ and {SYNTHETIC_DIR}/")