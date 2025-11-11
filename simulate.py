
import argparse
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# scapy for pcap generation and reading
from scapy.all import conf, IP, TCP, UDP, ICMP, Raw, wrpcap, rdpcap, Ether
conf.verb = 0
conf.route.resync() 
conf.use_pcap = True

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

# Maps (extend if needed)
PROTOCOL_MAP = {'tcp': 'protocol_type_tcp', 'udp': 'protocol_type_udp', 'icmp': 'protocol_type_icmp'}
SERVICE_MAP = {
    'http':'service_http', 'ftp':'service_ftp', 'ftp_data':'service_ftp_data', 
    'ssh':'service_ssh', 'domain':'service_domain', 'other':'service_other'
}
FLAG_MAP = {'OTH':'flag_OTH','REJ':'flag_REJ','RSTO':'flag_RSTO','RSTOS0':'flag_RSTOS0',
            'RSTR':'flag_RSTR','S0':'flag_S0','S1':'flag_S1','S2':'flag_S2','S3':'flag_S3',
            'SF':'flag_SF','SH':'flag_SH'}

def flows_to_kdd_csv(flow_csv_path, output_csv_path, train_columns_path, label_col='class'):
    """
    Convert simple flow CSV to full KDD-aligned CSV.
    flow_csv_path: input CSV with src,dst,proto,sport,dport,pkt_count,byte_count,duration,avg_pkt_size,flags_score
    output_csv_path: path to save KDD-formatted CSV
    train_columns_path: path to training column list (joblib .pkl)
    """
    df = pd.read_csv(flow_csv_path)
    
    # Load training columns
    train_columns = joblib.load(train_columns_path)
    
    # Copy df to avoid modifying original
    kdd_df = pd.DataFrame(index=df.index)
    
    # Numeric columns (fill missing ones with 0)
    numeric_cols = ['proto']
    
    numeric_cols = ['src_bytes','dst_bytes','wrong_fragment','urgent','hot','num_failed_logins',
                    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
                    'num_shells','num_access_files','num_outbound_cmds','count','srv_count',
                    'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
                    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                    'dst_host_rerror_rate','dst_host_srv_rerror_rate']
    
    
    for col in numeric_cols:
        if col in df.columns:
            kdd_df[col] = df[col]
        else:
            kdd_df[col] = 0.0
    
    # Add simple features from flow CSV
    kdd_df['src_bytes'] = df['byte_count'] / 2.0
    kdd_df['dst_bytes'] = df['byte_count'] / 2.0
    
    # Protocol one-hot
    for proto_col in ['protocol_type_tcp','protocol_type_udp','protocol_type_icmp']:
        kdd_df[proto_col] = 0
    for idx, row in df.iterrows():
        p = row['proto'].lower()
        proto_col = PROTOCOL_MAP.get(p, 'protocol_type_tcp')
        kdd_df.at[idx, proto_col] = 1
    
    # Service one-hot
    for svc_col in [c for c in train_columns if c.startswith('service_')]:
        kdd_df[svc_col] = 0
    for idx, row in df.iterrows():
        service = 'other'
        if row['dport'] in [80,443]:
            service='http'
        elif row['dport'] in [21,20]:
            service='ftp_data'
        elif row['dport']==22:
            service='ssh'
        elif row['dport']==53:
            service='domain'
        svc_col = SERVICE_MAP.get(service,'service_other')
        kdd_df.at[idx, svc_col] = 1
    
    # Flag one-hot
    for flag_col in [c for c in train_columns if c.startswith('flag_')]:
        kdd_df[flag_col] = 0
    for idx, row in df.iterrows():
        # simple heuristic from flags_score
        sc = row.get('flags_score',0.0)
        pkt_count = row.get('pkt_count',1)
        flag = 'OTH'
        if sc > 0.8 and pkt_count==1:
            flag='S0'
        elif sc>0.6 and pkt_count>1:
            flag='SF'
        elif sc>0.0 and sc<0.4:
            flag='REJ'
        flag_col = FLAG_MAP.get(flag,'flag_OTH')
        kdd_df.at[idx, flag_col] = 1
    
    # Land, logged_in, host flags
    for col in ['land_0','land_1','logged_in_0','logged_in_1','is_host_login_0','is_host_login_1','is_guest_login_0','is_guest_login_1']:
        kdd_df[col] = 0
    
    # Class/label
    if label_col in train_columns:
        kdd_df[label_col] = 'normal'  # default; you can change later
    
    # Align all columns to train_columns
    for c in train_columns:
        if c not in kdd_df.columns:
            kdd_df[c] = 0
    
    kdd_df = kdd_df[train_columns]
    
    kdd_df.to_csv(output_csv_path, index=False)
    print(f"KDD-aligned CSV saved to: {output_csv_path}")


# small service map: extend as needed
PORT_TO_SERVICE = {
    80: 'http', 443: 'http_443', 21: 'ftp', 20: 'ftp_data', 22: 'ssh', 23: 'telnet',
    25: 'smtp', 53: 'domain', 110: 'pop_3', 143: 'imap4', 79: 'finger', 111: 'sunrpc',
    137: 'netbios_ns', 138: 'netbios_dgm', 139: 'netbios_ssn', 161: 'ntp_u',
    # add more mappings if you need better coverage...
}

def infer_service_from_port(dport):
    try:
        p = int(dport)
    except:
        return 'other'
    return PORT_TO_SERVICE.get(p, 'other')

def infer_flag_from_flow_row(row):
    # heuristic using flags_score and pkt_count
    sc = row.get('flags_score', 0.0)
    pkts = row.get('pkt_count', 1)
    # heuristics (very approximate)
    if sc > 0.8 and pkts == 1:
        return 'S0'
    if sc > 0.6 and pkts > 1:
        return 'SF'
    if sc > 0.0 and sc < 0.4:
        return 'REJ'
    return 'OTH'

def flows_to_kdd(df_flows, train_columns, numeric_cols):
    df = df_flows.copy().reset_index(drop=True)

    # Protocol/flag/service mappings
    protocol_map = {'tcp':0,'udp':1,'icmp':2}
    flag_map = {'OTH':0,'REJ':1,'S0':5,'SF':9}
    service_map = {'http':0,'ftp':1,'ssh':2,'domain':3,'other':99}

    df['protocol_type'] = df['proto'].str.lower().map(protocol_map).fillna(2).astype(int)
    df['flag'] = df.apply(infer_flag_from_flow_row, axis=1).map(flag_map).fillna(0).astype(int)
    df['service'] = df['dport'].apply(infer_service_from_port).map(service_map).fillna(99).astype(int)

    # Basic numeric columns
    df['src_bytes'] = df['byte_count'] / 2
    df['dst_bytes'] = df['byte_count'] / 2
    df['land'] = (df['src']==df['dst']).astype(int)
    df['logged_in'] = 0
    df['is_host_login'] = 0
    df['is_guest_login'] = 0

    # Fill missing numeric columns
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Align to train_columns
    for c in train_columns:
        if c not in df.columns:
            df[c] = 0.0

    # Order columns
    kdd_final = df[train_columns].copy()
    return kdd_final


def _dummy_loss(y_true, y_pred):
    # placeholder loss for loading the model
    return tf.reduce_mean(y_pred - y_true) * 0.0



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
            pkt = IP(src=src,dst=dst)/TCP(sport=sport,dport=dport,flags="PA")/Raw(load=payload)
        else:
            pkt = IP(src=src,dst=dst)/UDP(sport=sport,dport=dport)/Raw(load=b'x'*avg_size)
        pkt.time = ts
        pkts.append(pkt)
        ts += abs(np.random.normal(0.05, 0.02))  
    return pkts

def synth_port_scan(scanner='10.0.0.99', target='10.0.0.2', start_port=20, end_port=200, pkt_gap=0.002):
    pkts = []
    ts = time.time()
    for p in range(start_port, end_port, 2):
        pkt = IP(src=scanner, dst=target)/TCP(sport=random.randint(40000,60000), dport=p, flags="S")
        pkt.time = ts
        pkts.append(pkt)
        ts += pkt_gap
    return pkts

def synth_syn_flood(attacker='10.0.0.77', target='10.0.0.2', dport=22, rate=0.001, count=500):
    pkts=[]
    ts = time.time()
    for i in range(count):
        pkt = IP(src=attacker, dst=target)/TCP(sport=random.randint(40000,65000), dport=dport, flags="S")
        pkt.time = ts
        pkts.append(pkt)
        ts += rate
    return pkts

def synth_udp_flood(attacker='10.0.0.88', target='10.0.0.2', dport=53, count=300):
    #ddoss-like many UDP packets
    pkts=[]
    ts = time.time()
    for i in range(count):
        pkt = IP(src=attacker, dst=target)/UDP(sport=random.randint(40000,65000), dport=dport)/Raw(load=b'X'*200)
        pkt.time = ts
        pkts.append(pkt)
        ts += 0.002
    return pkts

def synth_bruteforce_like(attacker='10.0.0.55', target='10.0.0.2', http_port=80, attempts=80):
    # simulate many short TCP connections or POSTs
    pkts=[]
    ts = time.time()
    for i in range(attempts):
        pkt = IP(src=attacker, dst=target)/TCP(sport=random.randint(40000,65000), dport=http_port, flags="PA")/Raw(load=b"user=admin&pass=guess")
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
        proto_name = 'OTHER'
        if pkt_has_tcp(p):
            sport = p[TCP].sport
            dport = p[TCP].dport
            proto_name = 'TCP'
        elif pkt_has_udp(p):
            sport = p[UDP].sport
            dport = p[UDP].dport
            proto_name = 'UDP'
        elif pkt_has_icmp(p):
            sport = 0
            dport = 0
            proto_name = 'ICMP'

        key = (src, dst, proto_name, sport, dport)
        flows[key].append(p)

    rows=[]
    for key, plist in flows.items():
        src, dst, proto_name, sport, dport = key
        times = [float(pkt.time) for pkt in plist]
        first = min(times)
        last = max(times)
        duration = max(0.0, last - first)
        pkt_count = len(plist)
        byte_count = sum(len(bytes(pkt)) for pkt in plist)
        avg_pkt_size = byte_count / pkt_count if pkt_count>0 else 0
        flags_score = 0.0
        syns=acks=psh=rst=0
        for pkt in plist:
            if pkt_has_tcp(pkt):
                f = str(pkt[TCP].flags)
                if 'S' in f and 'A' not in f: syns+=1
                if 'A' in f: acks+=1
                if 'R' in f: rst+=1
                if 'P' in f: psh+=1
        if pkt_count>0:
            flags_score = (syns*2 + rst*1.5 + psh*0.8)/pkt_count

        rows.append({
            'src': src, 'dst': dst, 'proto': proto_name,
            'sport': int(sport) if sport else 0,
            'dport': int(dport) if dport else 0,
            'pkt_count': pkt_count,
            'byte_count': byte_count,
            'duration': duration,
            'avg_pkt_size': avg_pkt_size,
            'flags_score': flags_score
        })

    df = pd.DataFrame(rows)

    # Load scaler & columns
    TRAIN_COLS_PATH = r"C:\DRL-Anomaly-Based-IDS\train_columns.pkl"
    SCALER_PATH = r"C:\DRL-Anomaly-Based-IDS\scaler.pkl"
    train_columns = joblib.load(TRAIN_COLS_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Numeric columns correspond to scaler (34)
    numeric_cols = train_columns[:len(scaler.mean_)]
    print("Number of training columns:", len(train_columns))
    print("Number of numeric scaler features:", len(numeric_cols))

    # Convert flows -> KDD format
    kdd_input = flows_to_kdd(df, train_columns=train_columns, numeric_cols=numeric_cols)

    # Scale only numeric features
    X_numeric_scaled = scaler.transform(kdd_input[numeric_cols])
    # Append remaining one-hot columns
    remaining_cols = [c for c in train_columns if c not in numeric_cols]
    X_final = np.hstack([X_numeric_scaled, kdd_input[remaining_cols].values])

    print("Final input shape to model:", X_final.shape)
    return df, X_final


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

def evaluate_and_show(model, scaler_path, train_columns_path, df_test, y_test):
    # Load scaler & train columns
    scaler = joblib.load(scaler_path)
    train_columns = joblib.load(train_columns_path)  # should be 34 features

    # Only keep training features (ignore others)
    df_test_aligned = df_test[train_columns].copy()

    # Scale
    Xs_scaled = scaler.transform(df_test_aligned.values)

    # Predict
    preds = (model.predict(Xs_scaled) > 0.5).astype(int).flatten()
    probs = model.predict(Xs_scaled).flatten()
    
    # Metrics & display
    print(classification_report(y_test, preds, digits=4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()


    # Show top anomalies
    df_results = df_test.copy()
    df_results['score'] = probs
    df_results['pred'] = preds
    flagged = df_results[df_results['pred']==1].sort_values('score', ascending=False)
    print("\nTop flagged flows (predicted anomaly):")
    print(flagged[['src','dst','proto','sport','dport','pkt_count','byte_count','duration','flags_score','score']].head(10))


# CLI / run
def main(args):
    print("Packet Creation")
    pkts = []
    # benign flows
    pkts += synth_benign_flow(src='10.0.0.10', dst='10.0.0.2', proto='tcp', sport=50432, dport=80, pkt_count=6)
    pkts += synth_benign_flow(src='10.0.0.11', dst='10.0.0.2', proto='udp', sport=58321, dport=53, pkt_count=5)
    pkts += synth_benign_flow(src='10.0.0.12', dst='10.0.0.3', proto='tcp', sport=50123, dport=443, pkt_count=10)
    # attacks
    pkts += synth_port_scan(scanner='10.0.0.99', target='10.0.0.2', start_port=20, end_port=200)
    pkts += synth_syn_flood(attacker='10.0.0.77', target='10.0.0.2', dport=22, rate=0.0005, count=400)
    pkts += synth_udp_flood(attacker='10.0.0.88', target='10.0.0.2', dport=53, count=250)
    pkts += synth_bruteforce_like(attacker='10.0.0.55', target='10.0.0.2', http_port=80, attempts=80)
    print("Packet Creation Successful")

    # Write to pcap
    out_pcap = args.outpcap
    #print("Pcap", pkts)

    wrpcap(out_pcap, sorted(pkts, key=lambda p: p.time))
    print(f"Wrote {len(pkts)} packets to {out_pcap} (offline file).")

    # ------------------------
    # Extract flows
    # ------------------------
    df, X_final = extract_flows_from_pcap(out_pcap)
    df.to_csv("demo_flows.csv", index=False)
    df_flows = pd.read_csv("demo_flows.csv")
    flows_to_kdd_csv(
        flow_csv_path="demo_flows.csv",
        output_csv_path="demo_flows_kdd.csv",
        train_columns_path=r"C:\DRL-Anomaly-Based-IDS\train_columns.pkl"
    )
    df_flows = pd.read_csv("demo_flows_kdd.csv")
    print(df_flows)
    df = annotate_labels(df)

    y = df['label'].astype(int)

    # Split for testing placeholder evaluation (optional)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_final, y, df, test_size=0.33, stratify=y, random_state=42
    )

    # ------------------------
    # Load TensorFlow model if provided
    # ------------------------
    if args.model:
        print("Loading model from", args.model)
        try:
            model = tf.keras.models.load_model(
                args.model,
                custom_objects={'_dummy_loss': _dummy_loss}
            )
            print("Loaded TensorFlow model successfully.")
        except Exception as e:
            print("Error loading model:", e)
            return

        # Load scaler & train columns
        scaler_path = r"C:\DRL-Anomaly-Based-IDS\scaler.pkl"
        train_columns_path = r"C:\DRL-Anomaly-Based-IDS\train_columns.pkl"
        evaluate_and_show(model, scaler_path, train_columns_path, df_test, y_test)

    else:
        print("No model provided — training placeholder model.")
        model, scaler = train_placeholder_model(X_train, y_train)
        joblib.dump(model, "placeholder_ensemble.joblib")
        joblib.dump(scaler, "placeholder_ensemble.joblib.scaler")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline intrusion demo (pcap generation + detection)")
    parser.add_argument("--outpcap", default="demo_simulated.pcap", help="Output pcap filename (written only)")
    parser.add_argument("--model", default=None, help="Optional: path to saved model (.joblib). If provided, will be used.")
    args = parser.parse_args()
    main(args)

    #python simulate.py "C:\DRL-Anomaly-Based-IDS\qrdqn_modelHpcc"