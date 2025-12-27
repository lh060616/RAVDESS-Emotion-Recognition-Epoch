import os
import shutil
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_PATH = "ravdess_audio"
SPLIT_DATA_PATH = "split_ravdess"
DATASET_PLOTS_PATH = "dataset_plots"
EMOTION_CODE_MAP = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"surprised","08":"disgusted"}
EMOTION_LABEL_MAP = {v:k for k,v in enumerate(EMOTION_CODE_MAP.values())}
NUM_CLASSES = 8
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.002
MAX_TIME_STEPS = 100
RANDOM_SEED = 42
PATIENCE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

def split_ravdess_dataset():
    os.makedirs(SPLIT_DATA_PATH, exist_ok=True)
    train_root = os.path.join(SPLIT_DATA_PATH, "train")
    test_root = os.path.join(SPLIT_DATA_PATH, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    for emotion in EMOTION_CODE_MAP.values():
        os.makedirs(os.path.join(train_root, emotion), exist_ok=True)
        os.makedirs(os.path.join(test_root, emotion), exist_ok=True)
    audio_paths = []
    emotion_labels = []
    for root, _, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith(".wav") and len(file.split("-"))>=3:
                emotion_code = file.split("-")[2]
                if emotion_code in EMOTION_CODE_MAP:
                    audio_paths.append(os.path.join(root, file))
                    emotion_labels.append(EMOTION_CODE_MAP[emotion_code])
    if len(audio_paths)==0:
        raise ValueError("未找到RAVDESS音频文件！请检查RAW_DATA_PATH是否正确")
    train_paths, test_paths, train_labels, test_labels = train_test_split(audio_paths, emotion_labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=emotion_labels)
    print(f"复制训练集文件（{len(train_paths)}个）...")
    for path, label in tqdm(zip(train_paths, train_labels), total=len(train_paths)):
        shutil.copy2(path, os.path.join(train_root, label))
    print(f"复制测试集文件（{len(test_paths)}个）...")
    for path, label in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
        shutil.copy2(path, os.path.join(test_root, label))
    print(f"\n数据集划分完成：训练集{len(train_paths)}个，测试集{len(test_paths)}个")
    return train_root, test_root

def plot_mfcc_heatmap(mfcc_feat, emotion, data_type, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12,7), dpi=300)
    im = plt.imshow(mfcc_feat.T,cmap="plasma",aspect="auto",vmin=np.min(mfcc_feat),vmax=np.max(mfcc_feat))
    cbar = plt.colorbar(im, label="MFCC特征值（数值越大，特征越显著）")
    cbar.ax.tick_params(labelsize=10)
    plt.title(f"{data_type.upper()} - {emotion}情绪MFCC特征热力图",fontsize=16,fontweight="bold")
    plt.xlabel("时间步（0-100，对应语音时长3-5秒）",fontsize=12)
    plt.ylabel("MFCC特征维度（1-39，含基础+一阶/二阶差分）",fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✅ 已保存示意图：{save_path}")

def generate_multi_plots(dataset, data_type="train"):
    target_emotions = {"train":["angry","sad","happy","calm"],"test":["neutral","fearful","surprised","disgusted"]}[data_type]
    label2emotion = {v:k for k,v in dataset.emotion2label.items()}
    save_dir = os.path.join(DATASET_PLOTS_PATH, data_type)
    os.makedirs(save_dir, exist_ok=True)
    for emotion in target_emotions:
        emotion_label = dataset.emotion2label[emotion]
        sample_idx = next((idx for idx, lbl in enumerate(dataset.labels) if lbl == emotion_label),None)
        if sample_idx is None:
            print(f"❌ 未找到{emotion}情绪样本，跳过")
            continue
        mfcc_feat, _ = dataset[sample_idx]
        mfcc_feat = mfcc_feat.squeeze(0).numpy()
        save_path = os.path.join(save_dir, f"{data_type}_{emotion}_mfcc.png")
        plot_mfcc_heatmap(mfcc_feat, emotion, data_type, save_path)
    print(f"\n📊 {data_type}示意图生成完成！")
    print(f"   保存路径：{save_dir}")
    print(f"   包含情绪：{', '.join(target_emotions)}（共{len(target_emotions)}张）\n")

class RAVDESSAudioDataset(Dataset):
    def __init__(self, data_root, is_train=True, scaler=None):
        self.data_root = data_root
        self.is_train = is_train
        self.max_time_steps = MAX_TIME_STEPS
        self.emotion2label = EMOTION_LABEL_MAP
        self.audio_paths, self.labels = self._load_data()
        if self.is_train:
            self.scaler = StandardScaler()
            self._fit_scaler()
        else:
            if scaler is None:
                raise ValueError("测试集必须传入训练集的Scaler！")
            self.scaler = scaler

    def _load_data(self):
        audio_paths = []
        labels = []
        for emotion, label in self.emotion2label.items():
            emotion_dir = os.path.join(self.data_root, emotion)
            if not os.path.exists(emotion_dir):
                continue
            for file in os.listdir(emotion_dir):
                if file.endswith(".wav"):
                    audio_paths.append(os.path.join(emotion_dir, file))
                    labels.append(label)
        return audio_paths, labels

    def _fit_scaler(self):
        print("拟合特征标准化器...")
        all_feats = []
        for path in tqdm(self.audio_paths, total=len(self.audio_paths)):
            all_feats.append(self._extract_mfcc(path))
        all_feats = np.concatenate(all_feats, axis=0)
        self.scaler.fit(all_feats)

    def _extract_mfcc(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            if self.is_train:
                rate = random.uniform(0.8,1.2)
                y = librosa.effects.time_stretch(y, rate=rate)
                n_steps = random.randint(-3,3)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                y = y * random.uniform(0.9,1.1)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160, center=False)
            mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
            mfcc_feat = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
            if mfcc_feat.shape[1] < self.max_time_steps:
                mfcc_feat = np.pad(mfcc_feat, ((0,0),(0,self.max_time_steps - mfcc_feat.shape[1])), mode='constant')
            else:
                mfcc_feat = mfcc_feat[:, :self.max_time_steps]
            return mfcc_feat.T
        except Exception as e:
            print(f"提取MFCC失败 {audio_path}：{e}")
            return np.zeros((self.max_time_steps, 39))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        mfcc_feat = self._extract_mfcc(audio_path)
        mfcc_feat = self.scaler.transform(mfcc_feat)
        feat_tensor = torch.tensor(mfcc_feat, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feat_tensor, label_tensor

class AudioEmotionCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=8):
        super(AudioEmotionCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_blocks = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*25*9,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,num_classes)
        )
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_blocks(x)
        return x

def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.25, patience=1, verbose=True)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for feats, labels in pbar:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        _, pred = torch.max(outputs,1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        avg_loss = total_loss / total
        avg_acc = 100 * correct / total
        pbar.set_postfix({"loss":f"{avg_loss:.4f}","acc":f"{avg_acc:.2f}%"})
    return avg_loss, avg_acc

@torch.no_grad()
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for feats, labels in test_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * feats.size(0)
        _, pred = torch.max(outputs,1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    avg_acc = 100 * correct / total
    print(f"\n测试集 - 损失：{avg_loss:.4f}，准确率：{avg_acc:.2f}%")
    return avg_loss, avg_acc, all_preds, all_labels

def plot_training_curves(train_losses, train_accs, test_losses, test_accs, best_epoch):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(12,5),dpi=300)
    plt.subplot(1,2,1)
    plt.plot(range(1,len(train_losses)+1),train_losses,label="训练集损失",marker="o",markersize=4)
    plt.plot(range(1,len(test_losses)+1),test_losses,label="测试集损失",marker="s",markersize=4)
    plt.axvline(x=best_epoch,color="red",linestyle="--",linewidth=2,label=f"最优轮次：{best_epoch}")
    plt.xlabel("训练轮次（10轮上限）",fontsize=12)
    plt.ylabel("损失值",fontsize=12)
    plt.title("10轮训练-损失曲线",fontsize=14,fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True,alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(range(1,len(train_accs)+1),train_accs,label="训练集准确率",marker="o",markersize=4)
    plt.plot(range(1,len(test_accs)+1),test_accs,label="测试集准确率",marker="s",markersize=4)
    plt.axvline(x=best_epoch,color="red",linestyle="--",linewidth=2,label=f"最优轮次：{best_epoch}")
    plt.xlabel("训练轮次（10轮上限）",fontsize=12)
    plt.ylabel("准确率 (%)",fontsize=12)
    plt.title("10轮训练-准确率曲线",fontsize=14,fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png",bbox_inches="tight",pad_inches=0.1)
    plt.close()
    print("✅ 已保存训练曲线：training_curves.png")

def plot_confusion_matrix(true_labels, pred_labels, best_epoch):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    emotion_names = ["中性","平静","快乐","悲伤","愤怒","恐惧","惊讶","厌恶"]
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,8),dpi=300)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=emotion_names,yticklabels=emotion_names,annot_kws={"fontsize":10})
    plt.xlabel("预测标签",fontsize=12)
    plt.ylabel("真实标签",fontsize=12)
    plt.title(f"10轮训练-混淆矩阵（最优轮次：第{best_epoch}轮）",fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_epoch{best_epoch}.png",bbox_inches="tight",pad_inches=0.1)
    plt.close()
    print(f"✅ 已保存混淆矩阵：confusion_matrix_epoch{best_epoch}.png")

def main():
    try:
        print("===== 第一步：划分数据集 =====")
        train_root, test_root = split_ravdess_dataset()
        print("\n===== 第二步：加载数据并生成示意图 =====")
        train_dataset = RAVDESSAudioDataset(train_root, is_train=True)
        test_dataset = RAVDESSAudioDataset(test_root, is_train=False, scaler=train_dataset.scaler)
        generate_multi_plots(train_dataset, data_type="train")
        generate_multi_plots(test_dataset, data_type="test")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"\n数据加载完成：训练集{len(train_dataset)}个样本，测试集{len(test_dataset)}个样本")
        print("\n===== 第三步：初始化模型 =====")
        model = AudioEmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        lr_scheduler = get_lr_scheduler(optimizer)
        print("模型结构：")
        print(model)
        print("\n===== 第四步：开始10轮训练 =====")
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
        best_test_acc = 0.0
        no_improve_epoch = 0
        best_epoch = 0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            test_loss, test_acc, _, _ = test_model(model, test_loader, criterion)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            lr_scheduler.step(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                no_improve_epoch = 0
                model_path = f"best_model_epoch{best_epoch}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"💾 第{best_epoch}轮保存最优模型：{model_path}")
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= PATIENCE:
                    print(f"⏹️  连续{PATIENCE}轮无提升，提前停止训练")
                    break
        print("\n===== 第五步：绘制训练曲线 =====")
        plot_training_curves(train_losses, train_accs, test_losses, test_accs, best_epoch)
        print("\n===== 第六步：绘制混淆矩阵 =====")
        best_model = AudioEmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
        best_model.load_state_dict(torch.load(f"best_model_epoch{best_epoch}.pth"))
        _, _, all_preds, all_labels = test_model(best_model, test_loader, criterion)
        plot_confusion_matrix(all_labels, all_preds, best_epoch)
        print("\n===== 训练完成！所有文件汇总 =====")
        print(f"📈 最优测试准确率：{best_test_acc:.2f}%")
        print(f"🔧 最优轮次：第{best_epoch}轮")
        print(f"💾 核心文件：")
        print(f"  - 最优模型：best_model_epoch{best_epoch}.pth")
        print(f"  - 训练曲线：training_curves.png")
        print(f"  - 混淆矩阵：confusion_matrix_epoch{best_epoch}.png")
        print(f"  - 数据集示意图：{DATASET_PLOTS_PATH}（train/test各4张）")
    except Exception as e:
        print(f"❌ 执行出错：{e}")
        raise

if __name__ == "__main__":
    main()