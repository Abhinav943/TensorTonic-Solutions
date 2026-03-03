import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    padded = []

    for seq in seqs:
        padded_seq = seq[:max_len]  # truncate if longer
        padded_seq = padded_seq + [pad_value] * (max_len - len(padded_seq))
        padded.append(padded_seq)

    return np.array(padded)
    pass