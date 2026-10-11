import zipfile, pandas as pd, numpy as np, os, argparse

def prepare(zip_path, out_dir, window=20, stride=10):
    os.makedirs(out_dir, exist_ok=True)
    train_users=set(range(1,30)); val_users=set(range(30,36))
    Xs, ys, meta = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        for n in names:
            p=n.split("/")
            label=1 if p[1].lower()=="fall" else 0
            user=int(p[2].replace("user",""))
            with z.open(n) as f: df=pd.read_csv(f)
            typ=df["Unnamed: 5"].astype(str).str.lower().str.strip()
            streams=[]
            for s in ("acc","gyro"):
                d=df[typ==s][["x","y","z"]].apply(pd.to_numeric,errors="coerce").dropna()
                streams.append(d.to_numpy(np.float32))
            common=min(len(streams[0]),len(streams[1]))
            if common < window: continue
            arr=np.concatenate([streams[0][:common],streams[1][:common]],axis=1)
            split="train" if user in train_users else ("val" if user in val_users else "test")
            for start in range(0,common-window+1,stride):
                Xs.append(arr[start:start+window]); ys.append(label)
                meta.append((user,label,n,start,split))
    X=np.stack(Xs).astype(np.float32); y=np.asarray(ys,np.int8)
    np.savez_compressed(os.path.join(out_dir,"bits2_imu_windows.npz"),X=X,y=y)
    pd.DataFrame(meta,columns=["user","label","source_file","start_sample","split"]).to_csv(
        os.path.join(out_dir,"window_metadata.csv"),index=False)
    print("X:",X.shape,"y:",y.shape,"class counts:",np.bincount(y))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("zip_path"); ap.add_argument("out_dir")
    args=ap.parse_args(); prepare(args.zip_path,args.out_dir)
