
import argparse
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# scapy for pcap generation and reading
from scapy.all import IP, TCP, UDP, ICMP, Raw, wrpcap, rdpcap, Ether

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import os

# synthetic traffic generation 
# PCAP-only, NEVER sent on network
def synth_benign_flow(src='10.0.0.1', dst='10.0.0.2', proto='tcp',
                      sport=50000, dport=80,
                      pkt_count=8, avg_size=500, start_ts=None):
    if start_ts is None:
        start_ts = time.time()
    pkts = []
    ts = start_ts
    for i in range(pkt_count):
        payload = ("GET /index.html HTTP/1.1\r\nHost:example\r\n\r\n").encode('utf-8') if proto=='tcp' else b''
        if proto == 'tcp':
            pkt = Ether()/IP(src=src,dst=dst)/TCP(sport=sport,dport=dport,flags="PA")/Raw(load=payload)
        else:
            pkt = Ether()/IP(src=src,dst=dst)/UDP(sport=sport,dport=dport)/Raw(load=b'x'*avg_size)
        pkt.time = ts
        pkts.append(pkt)
        ts += abs(np.random.normal(0.05, 0.02))  
    return pkts

def synth_port_scan(scanner='10.0.0.99', target='10.0.0.2', start_port=20, end_port=200, pkt_gap=0.002):
    pkts = []
    ts = time.time()
    for p in range(start_port, end_port, 2):
        pkt = Ether()/IP(src=scanner, dst=target)/TCP(sport=random.randint(40000,60000), dport=p, flags="S")
        pkt.time = ts
        pkts.append(pkt)
        ts += pkt_gap
    return pkts

def synth_syn_flood(attacker='10.0.0.77', target='10.0.0.2', dport=22, rate=0.001, count=500):
    pkts=[]
    ts = time.time()
    for i in range(count):
        pkt = Ether()/IP(src=attacker, dst=target)/TCP(sport=random.randint(40000,65000), dport=dport, flags="S")
        pkt.time = ts
        pkts.append(pkt)
        ts += rate
    return pkts

def synth_udp_flood(attacker='10.0.0.88', target='10.0.0.2', dport=53, count=300):
    #ddoss-like many UDP packets
    pkts=[]
    ts = time.time()
    for i in range(count):
        pkt = Ether()/IP(src=attacker, dst=target)/UDP(sport=random.randint(40000,65000), dport=dport)/Raw(load=b'X'*200)
        pkt.time = ts
        pkts.append(pkt)
        ts += 0.002
    return pkts

def synth_bruteforce_like(attacker='10.0.0.55', target='10.0.0.2', http_port=80, attempts=80):
    # simulate many short TCP connections or POSTs
    pkts=[]
    ts = time.time()
    for i in range(attempts):
        pkt = Ether()/IP(src=attacker, dst=target)/TCP(sport=random.randint(40000,65000), dport=http_port, flags="PA")/Raw(load=b"user=admin&pass=guess")
        pkt.time = ts
        pkts.append(pkt)
        ts += abs(np.random.normal(0.03, 0.01))
    return pkts


def extract_flows_from_pcap(pcap_path, flow_timeout=60.0):
    # read pcap and extract flows based on 5-tuple
    pkts = rdpcap(pcap_path)
    flows = defaultdict(list)
    for p in pkts:
        if IP not in p:
            continue
        ip = p[IP]
        proto = ip.proto
        src = ip.src
        dst = ip.dst

        sport = None
        dport = None
        if pkt_has_tcp(p):
            sport = p[TCP].sport
            dport = p[TCP].dport
            proto_name = 'TCP'
        elif pkt_has_udp(p):
            sport = p[UDP].sport
            dport = p[UDP].dport
            proto_name = 'UDP'
        elif pkt_has_icmp(p):
            proto_name = 'ICMP'
            sport = 0; dport = 0
        else:
            proto_name = str(proto)
            
        # flow key: 5-tuple (src,dst,proto,sport,dport) - directional
        key = (src, dst, proto_name, sport, dport)
        flows[key].append(p)

    rows=[]
    for key, plist in flows.items():
        src, dst, proto_name, sport, dport = key
        times = [float(pkt.time) for pkt in plist]
        first = min(times); last = max(times)
        duration = max(0.0, last - first)
        pkt_count = len(plist)
        byte_count = sum(len(bytes(pkt)) for pkt in plist)
        avg_pkt_size = byte_count / pkt_count if pkt_count>0 else 0
        flags_score = 0.0
        syns = 0; acks=0; psh=0; rst=0
        for pkt in plist:
            if pkt_has_tcp(pkt):
                f = pkt[TCP].flags
                try:
                    fstr = str(f)
                except:
                    fstr = ''
                if 'S' in fstr and 'A' not in fstr:
                    syns += 1
                if 'A' in fstr:
                    acks += 1
                if 'R' in fstr:
                    rst += 1
                if 'P' in fstr:
                    psh += 1
        if pkt_count>0:
            flags_score = (syns*2 + rst*1.5 + psh*0.8) / pkt_count
        rows.append({
            'src': src, 'dst': dst, 'proto': proto_name,
            'sport': int(sport) if sport is not None else 0,
            'dport': int(dport) if dport is not None else 0,
            'pkt_count': pkt_count,
            'byte_count': byte_count,
            'duration': duration,
            'avg_pkt_size': avg_pkt_size,
            'flags_score': flags_score,
          
            'label': None
        })
    df = pd.DataFrame(rows)
    return df

def pkt_has_tcp(pkt):
    return TCP in pkt

def pkt_has_udp(pkt):
    return UDP in pkt

def pkt_has_icmp(pkt):
    return ICMP in pkt

# Main demo simulation
def annotate_labels(df, attacker_prefixes=None, attack_dst=None):
    # Heuristic labeling for demo: flows from known attacker IPs -> anomaly
    if attacker_prefixes is None:
        attacker_prefixes = ['10.0.0.77','10.0.0.88','10.0.0.99','10.0.0.55']
    def lab(row):
        if row['src'] in attacker_prefixes:
            return 1
        # other heuristics: extremely high pkt rate short duration
        if row['pkt_count']>100 and row['duration']<1.0:
            return 1
        return 0
    df['label'] = df.apply(lab, axis=1)
    return df

def prepare_features(df):
    # numeric features for model
    X = df[['proto','sport','dport','pkt_count','byte_count','duration','avg_pkt_size','flags_score']].copy()
    # encode proto
    X['proto'] = X['proto'].map({'TCP':0,'UDP':1}).fillna(2)
    # scale ports to 0-1
    X['sport'] = X['sport'] / 65535.0
    X['dport'] = X['dport'] / 65535.0
    return X

def train_placeholder_model(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300)
    ensemble = VotingClassifier([('lr',clf1),('rf',clf2),('mlp',clf3)], voting='soft')
    ensemble.fit(Xs, y_train)
    return ensemble, scaler

def evaluate_and_show(model, scaler, X_test, y_test, df_test):
    Xs = scaler.transform(X_test)
    try:
        scaler = joblib.load("scaler.pkl")
        train_columns = joblib.load("train_columns.pkl")
        print("✅ Loaded scaler and train columns.")
    except Exception as e:
        raise RuntimeError("Could not load scaler/train_columns. "
                        "Ensure they were saved during training.") from e

    # === Ensure test dataframe has same encoded columns ===
    df_test = pd.get_dummies(df_test)

    # Align with training columns
    df_test = df_test.reindex(columns=train_columns, fill_value=0)

    # === Scale data ===
    Xs = scaler.transform(df_test.values)
    print("After alignment, Xs shape:", Xs.shape)

    preds = model.predict(Xs)
    probs = model.predict_proba(Xs)[:,1] if hasattr(model, 'predict_proba') else None
    print("=== Classification Report ===")
    print(classification_report(y_test, preds, digits=4))
    print("Accuracy:", accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix (Simulated Demo)")
    plt.tight_layout()
    plt.show()

    if probs is not None:
        df_test = df_test.copy()
        df_test['score'] = probs
        df_test['pred'] = preds
        flagged = df_test[df_test['pred']==1].sort_values('score', ascending=False)
        print("\nTop flagged flows (predicted anomaly):")
        print(flagged[['src','dst','proto','sport','dport','pkt_count','byte_count','duration','flags_score','score']].head(10))

# CLI / run
def main(args):
    pkts = []
    # some benign flows from various hosts
    pkts += synth_benign_flow(src='10.0.0.10', dst='10.0.0.2', proto='tcp', sport=50432, dport=80, pkt_count=6)
    pkts += synth_benign_flow(src='10.0.0.11', dst='10.0.0.2', proto='udp', sport=58321, dport=53, pkt_count=5)
    pkts += synth_benign_flow(src='10.0.0.12', dst='10.0.0.3', proto='tcp', sport=50123, dport=443, pkt_count=10)
    # simulated attacks
    pkts += synth_port_scan(scanner='10.0.0.99', target='10.0.0.2', start_port=20, end_port=200)
    pkts += synth_syn_flood(attacker='10.0.0.77', target='10.0.0.2', dport=22, rate=0.0005, count=400)
    pkts += synth_udp_flood(attacker='10.0.0.88', target='10.0.0.2', dport=53, count=250)
    pkts += synth_bruteforce_like(attacker='10.0.0.55', target='10.0.0.2', http_port=80, attempts=80)
    # write to pcap
    pkts_sorted = sorted(pkts, key=lambda p: p.time)
    out_pcap = args.outpcap
    wrpcap(out_pcap, pkts_sorted)
    print(f"Wrote {len(pkts_sorted)} packets to {out_pcap} (offline file).")

    df = extract_flows_from_pcap(out_pcap)
    print("Extracted flows:", len(df))
    df = annotate_labels(df)
    if df['label'].sum()==0:
        df.loc[df.sample(int(0.03*len(df))).index,'label']=1
    X = prepare_features(df)
    y = df['label'].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.33, stratify=y, random_state=42)
    
    # walker add model that is good here
    if args.model:
        print("Loading model from", args.model)

        if os.path.isdir(args.model):
            try:
                # Define any custom loss or layer objects used in training
                def _dummy_loss(y_true, y_pred):
                    return tf.reduce_mean(y_pred - y_true) * 0.0  # placeholder no-op loss

                model = tf.keras.models.load_model(
                    args.model,
                    custom_objects={'_dummy_loss': _dummy_loss}
                )
                print("Loaded TensorFlow model from:", args.model)
            except Exception as e:
                print("Error loading TensorFlow model:", e)
                exit(1)

            # Try to load the scaler
            scaler_path = args.model + ".scaler"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print("✅ Loaded scaler:", scaler_path)
            else:
                print("⚠️ No scaler found — fitting a new one from training data.")
                scaler = StandardScaler()
                scaler.fit(X_train)

        else:
            # Fallback: joblib model
            try:
                model = joblib.load(args.model)
                print("✅ Loaded joblib model:", args.model)
            except Exception as e:
                print("❌ Error loading joblib model:", e)
                exit(1)

            try:
                scaler = joblib.load(args.model + ".scaler")
                print("✅ Loaded scaler:", args.model + ".scaler")
            except:
                scaler = StandardScaler()
                scaler.fit(X_train)

    else:
        print("⚠️ No model provided — training placeholder model.")
        model, scaler = train_placeholder_model(X_train, y_train)
        joblib.dump(model, "placeholder_ensemble.joblib")
        joblib.dump(scaler, "placeholder_ensemble.joblib.scaler")

    evaluate_and_show(model, scaler, X_test, y_test, df_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline intrusion demo (pcap generation + detection)")
    parser.add_argument("--outpcap", default="demo_simulated.pcap", help="Output pcap filename (written only)")
    parser.add_argument("--model", default=None, help="Optional: path to saved model (.joblib). If provided, will be used.")
    args = parser.parse_args()
    main(args)

    #C:\DRL-Anomaly-Based-IDS\qrdqn_modelHpcc