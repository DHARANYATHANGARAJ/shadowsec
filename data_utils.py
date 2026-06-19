"""
data_utils.py
---------------
Synthetic network-flow data generator.

Real CICIDS2017 / NSL-KDD CSVs can be uploaded directly through the
Dataset Upload page. This module exists so the platform is fully
demonstrable WITHOUT requiring the user to source a multi-gigabyte
public dataset first:

  * `generate_synthetic_dataset()` builds a CICIDS2017-style flow table
    with realistic per-attack feature distributions.
  * `train_model.py` uses it to train the bundled LSTM (or the
    scikit-learn fallback) when no pre-trained model artifacts exist.
  * The "Load Sample Dataset" button on the Upload page uses it so a
    brand-new user can explore the whole pipeline in one click.

  # FUTURE SCOPE: replace with a connector that pulls live NetFlow/IPFIX
  # records from a real sensor/exporter for true real-time ingestion.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "src_port",
    "dst_port",
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "syn_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "active_mean",
    "idle_mean",
    "window_size",
]

LABEL_COLUMN = "label"

# Per-attack (mean, std) profile for each numeric feature, hand-tuned to be
# loosely representative of public IDS dataset literature. This is what
# makes the synthetic classes separable enough for a demo model to learn.
_PROFILES = {
    "Benign": dict(
        flow_packets_per_sec=(50, 20), flow_bytes_per_sec=(4000, 1500),
        syn_flag_count=(1, 1), rst_flag_count=(0, 1), flow_duration=(500000, 200000),
        total_fwd_packets=(10, 5), total_bwd_packets=(10, 5),
        fwd_packet_length_mean=(300, 100), bwd_packet_length_mean=(300, 100),
        active_mean=(2000, 800), idle_mean=(5000, 2000), window_size=(8192, 2048),
    ),
    "DDoS": dict(
        flow_packets_per_sec=(5000, 1500), flow_bytes_per_sec=(800000, 200000),
        syn_flag_count=(40, 15), rst_flag_count=(2, 2), flow_duration=(20000, 8000),
        total_fwd_packets=(400, 100), total_bwd_packets=(5, 5),
        fwd_packet_length_mean=(60, 20), bwd_packet_length_mean=(40, 15),
        active_mean=(500, 200), idle_mean=(100, 50), window_size=(512, 256),
    ),
    "DoS": dict(
        flow_packets_per_sec=(1200, 400), flow_bytes_per_sec=(150000, 50000),
        syn_flag_count=(20, 8), rst_flag_count=(1, 1), flow_duration=(40000, 15000),
        total_fwd_packets=(150, 50), total_bwd_packets=(8, 6),
        fwd_packet_length_mean=(80, 30), bwd_packet_length_mean=(50, 20),
        active_mean=(800, 300), idle_mean=(300, 150), window_size=(1024, 512),
    ),
    "Port Scan": dict(
        flow_packets_per_sec=(15, 8), flow_bytes_per_sec=(600, 300),
        syn_flag_count=(8, 3), rst_flag_count=(6, 3), flow_duration=(3000, 1500),
        total_fwd_packets=(2, 1), total_bwd_packets=(1, 1),
        fwd_packet_length_mean=(40, 10), bwd_packet_length_mean=(40, 10),
        active_mean=(100, 50), idle_mean=(50, 30), window_size=(2048, 512),
    ),
    "Botnet": dict(
        flow_packets_per_sec=(80, 40), flow_bytes_per_sec=(9000, 4000),
        syn_flag_count=(3, 2), rst_flag_count=(1, 1), flow_duration=(800000, 300000),
        total_fwd_packets=(25, 10), total_bwd_packets=(25, 10),
        fwd_packet_length_mean=(200, 80), bwd_packet_length_mean=(220, 90),
        active_mean=(50000, 20000), idle_mean=(80000, 30000), window_size=(4096, 1024),
    ),
    "Brute Force": dict(
        flow_packets_per_sec=(300, 100), flow_bytes_per_sec=(20000, 8000),
        syn_flag_count=(15, 5), rst_flag_count=(10, 4), flow_duration=(10000, 4000),
        total_fwd_packets=(60, 20), total_bwd_packets=(55, 20),
        fwd_packet_length_mean=(120, 40), bwd_packet_length_mean=(100, 30),
        active_mean=(1500, 600), idle_mean=(400, 200), window_size=(4096, 1024),
    ),
    "Web Attack": dict(
        flow_packets_per_sec=(120, 50), flow_bytes_per_sec=(30000, 12000),
        syn_flag_count=(4, 2), rst_flag_count=(2, 1), flow_duration=(15000, 6000),
        total_fwd_packets=(35, 15), total_bwd_packets=(40, 18),
        fwd_packet_length_mean=(500, 200), bwd_packet_length_mean=(700, 300),
        active_mean=(3000, 1200), idle_mean=(1000, 500), window_size=(8192, 2048),
    ),
}

ATTACK_LIST = list(_PROFILES.keys())


def _sample_profile(profile, n, rng):
    data = {}
    for feature, (mean, std) in profile.items():
        values = rng.normal(mean, std, n)
        data[feature] = np.clip(values, 0, None)
    return data


def generate_synthetic_dataset(n_per_class=300, seed=42) -> pd.DataFrame:
    """Build a labeled, CICIDS2017-style synthetic flow dataset."""
    rng = np.random.default_rng(seed)
    frames = []
    for attack, profile in _PROFILES.items():
        n = n_per_class if attack != "Benign" else n_per_class * 2
        block = _sample_profile(profile, n, rng)
        df = pd.DataFrame(block)
        df["duration"] = df["flow_duration"] / 1000.0
        df["protocol_type"] = rng.choice(["TCP", "UDP", "ICMP"], size=n, p=[0.7, 0.25, 0.05])
        df["src_port"] = rng.integers(1024, 65535, size=n)
        df["dst_port"] = rng.choice([80, 443, 22, 21, 3389, 8080, 53], size=n)
        df["psh_flag_count"] = np.clip(rng.normal(2, 1, n), 0, None)
        df["ack_flag_count"] = np.clip(rng.normal(10, 4, n), 0, None)
        df["urg_flag_count"] = np.clip(rng.normal(0.2, 0.3, n), 0, None)
        df[LABEL_COLUMN] = attack
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Re-order columns nicely
    ordered = [c for c in FEATURE_COLUMNS if c in full.columns] + [LABEL_COLUMN]
    return full[ordered]


if __name__ == "__main__":
    df = generate_synthetic_dataset()
    print(df.shape)
    print(df[LABEL_COLUMN].value_counts())
    print(df.head())
