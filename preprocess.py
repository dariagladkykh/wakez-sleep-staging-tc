import os
import numpy as np
import pandas as pd

DATA_DIR = "physionet.org/files/sleep-accel/1.0.0/"

def safe_read_file(path, names):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=' ', header=None, names=names, dtype=str)
        #convert time to num, coercing errors to NaN, then drop them
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df.dropna(subset=['time'], inplace=True)
        #convert time to int (seconds since PSG start)
        df['time'] = df['time'].astype(np.int64)
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def load_subject_data(subject_id):
    acc_path = os.path.join(DATA_DIR, "motion", f"{subject_id}_acceleration.txt")
    hr_path = os.path.join(DATA_DIR, "heart_rate", f"{subject_id}_heartrate.txt")
    steps_path = os.path.join(DATA_DIR, "steps", f"{subject_id}_steps.txt")
    label_path = os.path.join(DATA_DIR, "labels", f"{subject_id}_labeled_sleep.txt")

    acc = safe_read_file(acc_path, ['time', 'x', 'y', 'z'])
    hr = safe_read_file(hr_path, ['time', 'hr'])
    steps = safe_read_file(steps_path, ['time', 'steps'])
    labels = safe_read_file(label_path, ['time', 'stage'])

    if any(x is None for x in [acc, hr, steps, labels]):
        return None

    # check all values r right format
    for df, cols in [(acc, ['x','y','z']), (hr, ['hr']), (steps, ['steps']), (labels, ['stage'])]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols, inplace=True)

    # merge sbs
    df = labels
    df = df.merge(acc, on='time', how='left')
    df = df.merge(hr, on='time', how='left')
    df = df.merge(steps, on='time', how='left')

    # fill missing sensor values, then drop any remaining NaN in key columns
    df[['x','y','z','hr','steps']] = df[['x','y','z','hr','steps']].fillna(method='ffill').fillna(0)

    # Map stages
    def map_stage(s):
        if s == 0: return 0      # Wake
        elif s in [1,2,3]: return 1  # NREM
        elif s == 5: return 2    # REM
        else: return -1
    df['stage'] = df['stage'].apply(map_stage)
    df = df[df['stage'] != -1]

    return df[['time', 'x', 'y', 'z', 'hr', 'steps', 'stage']]

def create_epochs(df, window_sec=30):
    epochs = []
    labels = []
    times = sorted(df['time'].unique())
    if len(times) == 0:
        return np.array([]), np.array([])
    max_time = int(times[-1])
    for start in range(0, max_time, window_sec):
        end = start + window_sec
        seg = df[(df['time'] >= start) & (df['time'] < end)]
        if len(seg) == 0:
            continue
        if seg['stage'].nunique() == 1:
            label = seg['stage'].iloc[0]
            seg = seg.sort_values('time').reset_index(drop=True)
            if len(seg) < window_sec:
                last = seg.iloc[[-1]]
                pad = pd.concat([last] * (window_sec - len(seg)), ignore_index=True)
                seg = pd.concat([seg, pad], ignore_index=True)
            else:
                seg = seg.iloc[:window_sec]
            feats = seg[['x','y','z','hr','steps']].values.astype(np.float32)
            epochs.append(feats)
            labels.append(label)
    return np.array(epochs), np.array(labels)

def load_all_data():
    label_files = [f for f in os.listdir(os.path.join(DATA_DIR, "labels")) if f.endswith('.txt')]
    subject_ids = [f.split('_')[0] for f in label_files]
    Xs, ys = [], []
    for sid in subject_ids:
        df = load_subject_data(sid)
        if df is None or df.empty:
            continue
        X, y = create_epochs(df)
        if len(X) > 0:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise ValueError("No valid data loaded. Check file paths and content.")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)