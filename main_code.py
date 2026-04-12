import os, json, time, logging, warnings, platform, tracemalloc, glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
import cv2

warnings.filterwarnings('ignore')

CONFIG={
    'seed':42, 'blocks':(4,4), 'cv_folds':5, 'use_pca':False, 'pca_var':0.95,
    'param_grid':[{'kernel':['linear'],'C':[1,10]},{'kernel':['rbf'],'C':[1,10],'gamma':['scale','auto']}],
    'outdir':'results'}
np.random.seed(CONFIG['seed'])
os.makedirs(CONFIG['outdir'],exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')


# read local datasets

"""Read volume CK+48 for the Emotion Recognition problem. and return size 48x48"""
def load_ck_plus_48(dataset_path):
    X, y = [], []
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    for label, emotion in enumerate(emotions):
        folder_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".DS_Store"): continue  # skip file DS_Store
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48)) # follow paper's size
                X.append(img / 255.0)
                y.append(label)
                
    return X, np.array(y)

"""Read CFD and Clutter for the Face Detection problem. Return images of size 64x64."""
def load_cfd_detection_nested(cfd_images_path, clutter_path, max_faces=1410):
    
    X, y = [], []
    face_count = 0
    
    # label 1 - img face
    sub_datasets = ['CFD', 'CFD-INDIA', 'CFD-MR']
    for sub_ds in sub_datasets:
        ds_path = os.path.join(cfd_images_path, sub_ds)
        if not os.path.exists(ds_path): continue
            
        for subject_folder in os.listdir(ds_path):
            subject_path = os.path.join(ds_path, subject_folder)
            if not os.path.isdir(subject_path): continue
                
            # -N.jpg
            img_files = glob.glob(os.path.join(subject_path, "*-N.jpg"))
            if not img_files: img_files = glob.glob(os.path.join(subject_path, "*.jpg"))
                
            if img_files:
                img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64)) # follow paper's size
                    X.append(img / 255.0)
                    y.append(1)
                    face_count += 1
                    if face_count >= max_faces: break
        if face_count >= max_faces: break

    # label 0 - img clutter
    clutter_count = 0
    if os.path.exists(clutter_path):
        for filename in os.listdir(clutter_path):
            if filename.endswith(".DS_Store"): continue  # skip file DS_Store
            img_path = os.path.join(clutter_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img / 255.0)
                y.append(0)
                clutter_count += 1
                if clutter_count >= face_count: break #
        
    return X, np.array(y)


# processing lbp and evaluation 
def preprocess(img):
    x=(img*255).astype(np.uint8) if img.max()<=1 else img.astype(np.uint8)
    x=cv2.GaussianBlur(x,(3,3),0)
    return cv2.createCLAHE(2.0,(8,8)).apply(x)

def neigh(p):
    return (p[:-2,:-2],p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2])

def hist(v,b):
    h,_=np.histogram(v.ravel(),bins=b,range=(0,b)); return h/(h.sum()+1e-9)

def stlbp(im):
    p=np.pad(im,1,mode='edge'); C=im; P1,P2,P3,P4,P5,P6,P7,P8=neigh(p)
    code=(P1>=C)+2*(P2>=C)+4*(P3>=C)+8*(P4>=C)+16*(P5>=C)+32*(P6>=C)+64*(P7>=C)+128*(P8>=C)
    return hist(code,256)

def sylbp8(im):
    p=np.pad(im,1,mode='edge'); P1,P2,P3,P4,P5,P6,P7,P8=neigh(p)
    code=(P1>=P5)+2*(P2>=P6)+4*(P3>=P7)+8*(P4>=P8)+16*(P5>=P1)+32*(P6>=P2)+64*(P7>=P3)+128*(P8>=P4)
    return hist(code,256)

def sylbp4(im):
    p=np.pad(im,1,mode='edge'); P1,P2,P3,P4,P5,P6,P7,P8=neigh(p)
    code=(P5>=P1)+2*(P6>=P2)+4*(P7>=P3)+8*(P8>=P4)
    return hist(code,16)

MAP={'stlbp':stlbp,'sylbp8':sylbp8,'sylbp4':sylbp4}

def check_redundancy(image):
    p=np.pad(image,1,mode='edge'); P1,P2,P3,P4,P5,P6,P7,P8=neigh(p)
    d1 = (P1 >= P5).astype(int).ravel()
    d5 = (P5 >= P1).astype(int).ravel()
    corr, _ = stats.pearsonr(d1, d5)
    logging.info(f"Pearson correlation coefficient between H(d1) and H(d5): {corr:.4f}")
    if corr < -0.9: logging.info("redundancy confirmation: d_i and d_{i+4} are strongly inversely linearly dependent.")

def features(images,method):
    br,bc=CONFIG['blocks']; bins=256 if method!='sylbp4' else 16
    X=np.zeros((len(images),br*bc*bins),dtype=np.float32)
    for n,img in enumerate(images):
        im=preprocess(img); h,w=im.shape; ys=np.array_split(np.arange(h),br); xs=np.array_split(np.arange(w),bc)
        ptr=0
        for ry in ys:
            for cx in xs:
                block=im[np.ix_(ry,cx)]
                X[n,ptr:ptr+bins]=MAP[method](block); ptr+=bins
    return X

def evaluate(name,images,y):
    logging.info(f"evaluate datasets: {name} ({len(images)} samples)")
    rows=[]; methods=['stlbp','sylbp8','sylbp4']
    cv_scores_dict = {} 
    skf=StratifiedKFold(CONFIG['cv_folds'],shuffle=True,random_state=CONFIG['seed'])
    
    for m in methods:
        tracemalloc.start()
        t=time.time()
        X=features(images,m)
        ext=time.time()-t
        current, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_ram / (1024 * 1024)

        if CONFIG['use_pca']: X=PCA(CONFIG['pca_var'],random_state=CONFIG['seed']).fit_transform(X)
        gs=GridSearchCV(SVC(random_state=CONFIG['seed']),CONFIG['param_grid'],cv=skf,scoring='accuracy',n_jobs=-1)
        gs.fit(X,y); best=gs.best_estimator_
        scores=cross_val_score(best,X,y,cv=skf,scoring='accuracy',n_jobs=-1)
        cv_scores_dict[m] = scores

        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
        best.fit(Xtr,ytr); pred=best.predict(Xte)
        
        logging.info(f"[{m.upper()}] Acc: {scores.mean():.4f} | Time: {ext:.2f}s | RAM: {peak_mb:.2f}MB | Dim: {X.shape[1]}")
        rows.append({'dataset':name,'method':m,'acc_mean':float(scores.mean()),'acc_std':float(scores.std()),'f1':float(f1_score(yte,pred,average='weighted')),'time':ext,'ram_mb':peak_mb,'dim':int(X.shape[1]),'params':str(gs.best_params_)})
        
        if m=='sylbp4':
            cm=confusion_matrix(yte,pred)
            plt.figure(figsize=(8,6)); sns_heatmap(cm, name) 
            with open(os.path.join(CONFIG['outdir'],f'report_{name}.txt'),'w') as f: f.write(classification_report(yte,pred))

    t_stat, p_val = stats.ttest_rel(cv_scores_dict['stlbp'], cv_scores_dict['sylbp4'])
    logging.info(f"StLBP vs SyLBP4 p-value: {p_val:.4f}")
    if p_val > 0.05: logging.info("CONCLUSION: There is no statistically significant difference in Accuracy. SyLBP4 is a viable replacement for StLBP.")
    
    return rows

def sns_heatmap(cm, name):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{name} SyLBP4 Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(CONFIG['outdir'],f'cm_{name}.png'))
    plt.close()

def dashboard(all_rows):
    for ds in sorted(set(r['dataset'] for r in all_rows)):
        sub=[r for r in all_rows if r['dataset']==ds]; methods=[r['method'] for r in sub]
        
        plt.figure(figsize=(8,4)); plt.bar(methods,[r['acc_mean']*100 for r in sub], color=['#3498db','#95a5a6','#e74c3c'])
        plt.title(f'{ds} Accuracy (%)'); plt.ylabel('%'); plt.ylim(0,105); plt.tight_layout(); plt.savefig(os.path.join(CONFIG['outdir'],f'acc_{ds}.png')); plt.close()
        
        plt.figure(figsize=(8,4)); plt.bar(methods,[r['time'] for r in sub], color=['#2ecc71','#95a5a6','#f1c40f'])
        plt.title(f'{ds} Extraction Time (s)'); plt.ylabel('Seconds'); plt.tight_layout(); plt.savefig(os.path.join(CONFIG['outdir'],f'time_{ds}.png')); plt.close()

        plt.figure(figsize=(8,4)); plt.bar(methods,[r['ram_mb'] for r in sub], color=['#9b59b6','#95a5a6','#e67e22'])
        plt.title(f'{ds} Memory Usage (MB)'); plt.ylabel('Peak RAM (MB)'); plt.tight_layout(); plt.savefig(os.path.join(CONFIG['outdir'],f'ram_{ds}.png')); plt.close()

def main():
    meta={'python':platform.python_version()}
    all_rows=[]
    redundancy_checked = False

    # define dataset paths
    base_dir = "./datasets"
    ck_path = os.path.join(base_dir, "CK_Plus_48")
    cfd_path = os.path.join(base_dir, "CFD_Faces")
    clutter_path = os.path.join(base_dir, "Clutter_Images")

    # 1. FACIAL EXPRESSION RECOGNITION (CK+48)
    if os.path.exists(ck_path):
        X_emo, y_emo = load_ck_plus_48(ck_path)
        if len(X_emo) > 0:
            if not redundancy_checked:
                check_redundancy(X_emo[0])
                redundancy_checked = True
            all_rows += evaluate('CKPlus48_Emotion', X_emo, y_emo)
    else:
        logging.error(f"Directory not found: {ck_path}")

    # 2. FACE DETECTION (CFD vs Clutter)
    if os.path.exists(cfd_path) and os.path.exists(clutter_path):
        X_det, y_det = load_cfd_detection_nested(cfd_path, clutter_path)
        
        if len(np.unique(y_det)) == 2:
            all_rows += evaluate('CFD_Detection', X_det, y_det)
        else:
            logging.error("Dataset for Face Detection is missing classes. Please check images in CFD_Faces or Clutter_Images.")
    else:
        logging.error("Directory not found: CFD_Faces or Clutter_Images.")

    dashboard(all_rows)
    with open(os.path.join(CONFIG['outdir'],'results.json'),'w') as f: json.dump({'meta':meta,'results':all_rows},f,indent=2)

if __name__=='__main__': 
    main()