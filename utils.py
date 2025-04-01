import torch
import numpy as np
from torch.utils.data import Dataset

class MIXDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        s1 , s2 = sequence
        s1 = torch.tensor(s1[np.newaxis,:], dtype=torch.float32)
        s2 = torch.tensor(s2[np.newaxis,:], dtype=torch.float32)
        return (s1,s2),label

def padding(sequence):
    length = len(sequence)
    if length > 5000:
        result = np.array(sequence[:5000])
    else:
        result = np.pad(sequence, (0, 5000 - length), 'constant')
    return np.array(result)


def length_align(X, seq_len):
    if seq_len < X.shape[-1]:
        X = X[...,:seq_len]  # Truncate the sequence if seq_len is shorter than the sequence length
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]  # Calculate padding length
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)  # Pad the sequence with zeros
    return X

def load_data(data_path, feature_type, seq_len):
    """
    Load and process data from a specified path.

    Parameters:
    data_path (str): Path to the data file.
    feature_type (str): Type of feature to extract.
    seq_len (int): Desired sequence length.

    Returns:
    tuple: Processed feature tensor and label tensor.
    """
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if feature_type == "DIR":
        X = np.sign(X)  # Directional feature
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "DT":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "DT2":
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0  # Ensure no negative values
        X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
        X_time = length_align(X_time, seq_len)[:, np.newaxis]
        X = np.concatenate([X_dir, X_time], axis=1)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "TAM":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "TAF":
        X = length_align(X, seq_len)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "BUR":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "Origin":
        X = length_align(X, seq_len)
    elif feature_type == "MIX":
        BUR_X = data["X_BUR"]
        X = zip(X,BUR_X)
        X = list(X)
    elif feature_type == "MIXT":#interval + direction
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0  # Ensure no negative values
        X_dir = length_align(X_dir, seq_len)
        X_time = length_align(X_time, seq_len)
        X = zip(X_dir,X_time)
        X = list(X)
    elif feature_type == "TS":#interval
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0
        X_time = length_align(X_time, seq_len)
        X = torch.tensor(X_time[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "TID":#timestamp + direction
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time[X_time < 0] = 0  # Ensure no negative values
        X_dir = length_align(X_dir, seq_len)
        X_time = length_align(X_time, seq_len)
        X = zip(X_dir,X_time)
        X = list(X)
    else:
        raise ValueError(f"Feature type {feature_type} is not matched.")
    return X, y

def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths

def agg_interval(packets):
    features = []
    features.append([np.sum(packets>0), np.sum(packets<0)])

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts>0), np.sum(bursts<0)])

    pos_bursts = bursts[bursts>0]
    neg_bursts = np.abs(bursts[bursts<0])
    vals = []
    if len(pos_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(neg_bursts))
    features.append(vals)

    return np.array(features, dtype=np.float32)

def process_TAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((3, 2, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[:, :, interval_idx] = agg_interval(cur_packets)
        st_pos = ed_pos
    
    return index, TAF

def load_iter(X, y, batch_size, feature_type ,is_train=True):
    if feature_type == "MIX" or  feature_type == "MIXT" or feature_type == 'TID':
        dataset = MIXDataset(X,y)
    else:
        dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train)


def multi_eval(y_pred_score,y_true):
    max_tab = 2
    tp = {}
    for tab in range(1, max_tab+1):
        tp[tab] = 0

    for idx in range(y_pred_score.shape[0]):
        cur_pred = y_pred_score[idx]
        for tab in range(1, max_tab+1):
            target_webs = cur_pred.argsort()[-tab:]
            for target_web in target_webs:
                if y_true[idx,target_web] > 0:
                    tp[tab] += 1
    mapk=.0
    for tab in range(1, max_tab+1):
        p_tab = tp[tab] / (y_true.shape[0] * tab)
        mapk += p_tab
        print(f"p@{tab}: {round(p_tab,3)}, map@{tab}: {round(mapk/tab,3)}")
    return round(p_tab,3),round(mapk/tab,3)