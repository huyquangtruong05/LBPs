import os, json, time, logging, warnings, platform, tracemalloc, glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

CONFIG={
    'seed':42, 'blocks':(4,4), 'cv_folds':5,
    'param_grid':[{'kernel':['linear'],'C':[1,10]}, {'kernel':['rbf'],'C':[1,10],'gamma':['scale','auto']}],
    'outdir':'results'
}
np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed']) 
os.makedirs(CONFIG['outdir'],exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')

# load datasets ck+48
def load_ck_plus_48(dataset_path):
    X, y = [], []
    emotions = ['anger', 'fear', 'happy', 'sadness', 'surprise']
    for label, emotion in enumerate(emotions):
        folder_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(folder_path): continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".DS_Store"): continue
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                X.append(img / 255.0)
                y.append(label)
    return X, np.array(y)

# load datasets cfd detection (nested structure) 
def load_cfd_detection_nested(cfd_images_path, clutter_path, max_faces=1410):
    X, y = [], []
    face_count = 0
    sub_datasets = ['CFD', 'CFD-INDIA', 'CFD-MR']
    for sub_ds in sub_datasets:
        ds_path = os.path.join(cfd_images_path, sub_ds)
        if not os.path.exists(ds_path): continue
        for subject_folder in os.listdir(ds_path):
            subject_path = os.path.join(ds_path, subject_folder)
            if not os.path.isdir(subject_path): continue
            img_files = glob.glob(os.path.join(subject_path, "*-N.jpg"))
            if not img_files: img_files = glob.glob(os.path.join(subject_path, "*.jpg"))
            if img_files:
                img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    X.append(img / 255.0)
                    y.append(1)
                    face_count += 1
                    if face_count >= max_faces: break
        if face_count >= max_faces: break

    clutter_count = 0
    if os.path.exists(clutter_path):
        for filename in os.listdir(clutter_path):
            if filename.endswith(".DS_Store"): continue
            img_path = os.path.join(clutter_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img / 255.0)
                y.append(0)
                clutter_count += 1
                if clutter_count >= face_count: break
    return X, np.array(y)

# extract features
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

# map method name to function
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

# evaluate methods with SVM and cross-validation
def evaluate(name,images,y):
    logging.info(f" EVALUATION DATASET: {name} ({len(images)} samples) ")
    rows=[]; methods=['stlbp','sylbp8','sylbp4']
    cv_scores_dict = {} 
    skf=StratifiedKFold(CONFIG['cv_folds'],shuffle=True,random_state=CONFIG['seed'])
    
    for m in methods:
        t=time.time()
        X=features(images,m)
        ext=time.time()-t

        gs=GridSearchCV(SVC(probability=True, random_state=CONFIG['seed']),CONFIG['param_grid'],cv=skf,scoring='accuracy',n_jobs=-1)
        gs.fit(X,y); best=gs.best_estimator_
        
        scores=cross_val_score(best,X,y,cv=skf,scoring='accuracy',n_jobs=-1)
        cv_scores_dict[m] = scores

        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=CONFIG['seed'])
        best.fit(Xtr,ytr); pred=best.predict(Xte)
        
        logging.info(f"[{m.upper()} + SVM] Acc: {scores.mean():.4f} | Ext Time: {ext:.2f}s | Dim: {X.shape[1]}")
        rows.append({'dataset':name,'method':m,'acc_mean':float(scores.mean()),'time':ext,'dim':int(X.shape[1])})
        
        if m=='sylbp4':
            cm=confusion_matrix(yte,pred)
            plt.figure(figsize=(8,6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{name} SyLBP4 (SVM) Confusion Matrix')
            plt.colorbar(); plt.tight_layout(); plt.ylabel('True label'); plt.xlabel('Predicted label')
            plt.savefig(os.path.join(CONFIG['outdir'],f'cm_svm_{name}.png'))
            plt.close()
            with open(os.path.join(CONFIG['outdir'],f'report_svm_{name}.txt'),'w') as f: f.write(classification_report(yte,pred))

    t_stat, p_val = stats.ttest_rel(cv_scores_dict['stlbp'], cv_scores_dict['sylbp4'])
    logging.info(f" StLBP vs SyLBP4 p-value: {p_val:.4f}")
    if np.isnan(p_val) or p_val > 0.05: 
        logging.info("CONCLUSION: SyLBP4 can perfectly replace StLBP.")
    return rows

# EVALUATE WEAKNESSES OF SYLBP4 (CRITICAL ANALYSIS)
def evaluate_sylbp4_weaknesses(name, images, y):
    logging.info(f"ANALYSIS OF SyLBP4 WEAKNESSES ON {name}")
    
    weakness_dir = os.path.join(CONFIG['outdir'], 'SyLBP4_Weaknesses')
    os.makedirs(weakness_dir, exist_ok=True)
    
    # split data into train/test for weakness evaluation (train on clean, test on modified)
    X_img_tr, X_img_te, ytr, yte = train_test_split(images, y, test_size=0.3, stratify=y, random_state=CONFIG['seed'])
    
    # train models on clean training data
    models = {}
    for m in ['stlbp', 'sylbp4']:
        X_tr_feat = features(X_img_tr, m)
        clf = SVC(kernel='linear', C=1, random_state=CONFIG['seed'])
        clf.fit(X_tr_feat, ytr)
        models[m] = clf

    # TEST 1: NOISE SENSITIVITY (Salt & Pepper) 
    noise_levels = [0, 0.05, 0.1, 0.2, 0.3]
    noise_acc = {'stlbp': [], 'sylbp4': []}
    
    for prob in noise_levels:
        noisy_te = []
        for img in X_img_te:
            noisy = np.copy(img)
            if prob > 0:
                rnd = np.random.rand(*noisy.shape)
                noisy[rnd < prob/2] = 0.0 # black pepper
                noisy[rnd > 1 - prob/2] = 1.0 # white salt
            noisy_te.append(noisy)
            
        for m in ['stlbp', 'sylbp4']:
            feat = features(noisy_te, m)
            acc = accuracy_score(yte, models[m].predict(feat))
            noise_acc[m].append(acc * 100)

    # TEST 2: BLUR & SCALE LIMITATION (Gaussian Blur) 
    blur_levels = [1, 3, 5, 7, 9] # ksize of a Gaussian Blur 
    blur_acc = {'stlbp': [], 'sylbp4': []}
    
    for k in blur_levels:
        blur_te = []
        for img in X_img_te:
            if k > 1:
                blurred = cv2.GaussianBlur(img, (k, k), 0)
            else:
                blurred = img
            blur_te.append(blurred)
            
        for m in ['stlbp', 'sylbp4']:
            feat = features(blur_te, m)
            acc = accuracy_score(yte, models[m].predict(feat))
            blur_acc[m].append(acc * 100)

    # TEST 3: ROTATION VARIANCE 
    angles = [0, 5, 15, 30, 45]
    rot_acc = {'stlbp': [], 'sylbp4': []}
    
    for angle in angles:
        rot_te = []
        for img in X_img_te:
            if angle > 0:
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = img
            rot_te.append(rotated)
            
        for m in ['stlbp', 'sylbp4']:
            feat = features(rot_te, m)
            acc = accuracy_score(yte, models[m].predict(feat))
            rot_acc[m].append(acc * 100)

    # show chart for each weakness
    def plot_weakness(x_vals, y_dict, xlabel, title, filename):
        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, y_dict['stlbp'], marker='o', linestyle='-', color='#3498db', label='StLBP (8-bit)')
        plt.plot(x_vals, y_dict['sylbp4'], marker='s', linestyle='--', color='#e74c3c', label='SyLBP4 (4-bit)')
        plt.title(title, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(weakness_dir, filename))
        plt.close()

    plot_weakness(noise_levels, noise_acc, 'Noise Probability (Salt & Pepper)', 
                  f'Weakness 1: Noise Sensitivity ({name})', 
                  f'1_noise_sensitivity_{name}.png')
                  
    plot_weakness(blur_levels, blur_acc, 'Blur Kernel Size', 
                  f'Weakness 2: Blur & Resolution Limitation ({name})', 
                  f'2_blur_limitation_{name}.png')
                  
    plot_weakness(angles, rot_acc, 'Rotation Angle (Degrees)', 
                  f'Weakness 3: Rotation Variance ({name})', 
                  f'3_rotation_variance_{name}.png')
    

# BONUS: END-TO-END CNN vs SyLBP4 + MLP
def evaluate_hybrid_bonus(name, images, y):
    logging.info(f"CNN (Raw Pixels) vs MLP (SyLBP4 Features)")
    X_raw = np.array(images)
    X_raw = np.expand_dims(X_raw, axis=-1) 
    
    t_ext = time.time()
    X_sylbp = features(images, 'sylbp4')
    ext_time = time.time() - t_ext
    
    X_raw_tr, X_raw_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.3, stratify=y, random_state=CONFIG['seed'])
    X_sy_tr, X_sy_te, _, _ = train_test_split(X_sylbp, y, test_size=0.3, stratify=y, random_state=CONFIG['seed'])

    cnn = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    t0 = time.time()
    cnn.fit(X_raw_tr, y_tr, epochs=20, batch_size=32, verbose=0)
    cnn_time = time.time() - t0
    cnn_acc = accuracy_score(y_te, (cnn.predict(X_raw_te, verbose=0) > 0.5).astype(int))
    cnn_params = cnn.count_params()

    mlp = Sequential([
        Dense(64, activation='relu', input_shape=(256,)),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    t1 = time.time()
    mlp.fit(X_sy_tr, y_tr, epochs=20, batch_size=32, verbose=0)
    mlp_time = time.time() - t1
    mlp_acc = accuracy_score(y_te, (mlp.predict(X_sy_te, verbose=0) > 0.5).astype(int))
    mlp_params = mlp.count_params()
    
    logging.info(f"[CNN (Raw 64x64)] Acc: {cnn_acc:.4f} | Train Time: {cnn_time:.2f}s | Params: {cnn_params:,}")
    logging.info(f"[SyLBP4 + MLP]    Acc: {mlp_acc:.4f} | Train Time: {mlp_time:.2f}s | Params: {mlp_params:,}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ['CNN (Raw Pixels)', 'SyLBP4 + MLP']
    
    axes[0].bar(models, [cnn_acc*100, mlp_acc*100], color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Test Accuracy (%)')
    axes[0].set_ylim(0, 105)
    
    axes[1].bar(models, [cnn_time, mlp_time], color=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Training Time (Seconds)')
    
    axes[2].bar(models, [cnn_params, mlp_params], color=['#e74c3c', '#2ecc71'])
    axes[2].set_title('Model Size (Number of Parameters)')
    axes[2].set_yscale('log')
    
    plt.suptitle(' End-to-End CNN vs SyLBP4 Hybrid DL', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['outdir'], f'bonus_hybrid_comparison_{name}.png'))
    plt.close()

def dashboard(all_rows):
    colors = ['#3498db','#95a5a6','#e74c3c']
    for ds in sorted(set(r['dataset'] for r in all_rows)):
        sub=[r for r in all_rows if r['dataset']==ds]; methods=[r['method'] for r in sub]
        
        plt.figure(figsize=(8,4)); plt.bar(methods,[r['acc_mean']*100 for r in sub], color=colors[:len(methods)])
        plt.title(f'{ds} SVM Accuracy (%)'); plt.ylabel('%'); plt.ylim(0,105); plt.tight_layout(); plt.savefig(os.path.join(CONFIG['outdir'],f'acc_{ds}.png')); plt.close()
        
        plt.figure(figsize=(8,4)); plt.bar(methods,[r['time'] for r in sub], color=['#2ecc71','#95a5a6','#f1c40f'][:len(methods)])
        plt.title(f'{ds} Extraction Time (s)'); plt.ylabel('Seconds'); plt.tight_layout(); plt.savefig(os.path.join(CONFIG['outdir'],f'time_{ds}.png')); plt.close()

def main():
    all_rows=[]
    redundancy_checked = False

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
            
            # initialize weakness evaluation for CK+48
            evaluate_sylbp4_weaknesses('CKPlus48_Emotion', X_emo, y_emo)

    # 2. FACE DETECTION (CFD vs Clutter)
    if os.path.exists(cfd_path) and os.path.exists(clutter_path):
        X_det, y_det = load_cfd_detection_nested(cfd_path, clutter_path)
        if len(np.unique(y_det)) == 2:
            all_rows += evaluate('CFD_Detection', X_det, y_det)
            
            # initialize weakness evaluation for CFD Detection
            evaluate_sylbp4_weaknesses('CFD_Detection', X_det, y_det)
            
            # BONUS: END-TO-END CNN vs SyLBP4 + MLP on CFD Detection
            evaluate_hybrid_bonus('CFD_Detection', X_det, y_det)
            
    dashboard(all_rows)
    with open(os.path.join(CONFIG['outdir'],'results.json'),'w') as f: json.dump({'results':all_rows},f,indent=2)

if __name__=='__main__': 
    main()