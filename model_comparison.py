"""
MindTrack Advanced - Model Karsilastirma Calismasi
Programlama Dilleri Dersi Projesi

Bu dosya ayni dataset uzerinde 5 farkli model mimarisini
egitip performanslarini karsilastirir.

Modeller:
1. Baseline - Basit feedforward (mevcut)
2. Deep Network - Daha derin ag + BatchNorm
3. Residual Network - Skip connections
4. Attention Network - Self-attention mekanizmasi
5. Ensemble - Birden fazla modelin birlesimi

Kullanim:
    python model_comparison.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Font ayari
plt.rcParams['font.family'] = 'DejaVu Sans'

# Dosya yollari
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
DATASET_FILE = os.path.join(ASSETS_DIR, 'Sleep_health_and_lifestyle_dataset.csv')


# =============================================================================
# VERI HAZIRLAMA
# =============================================================================

def encode_gender(gender):
    return 0 if gender == "Male" else 1

def encode_bmi(bmi):
    mapping = {"Underweight": 0, "Normal": 1, "Normal Weight": 1, "Overweight": 2, "Obese": 3}
    return mapping.get(bmi, 1)

def generate_label(row):
    stress = row['Stress Level']
    sleep = row['Sleep Duration']
    quality = row['Quality of Sleep']
    
    if stress >= 7 and sleep <= 6 and quality <= 5:
        return 0  # riskli
    elif stress <= 4 and sleep >= 7 and quality >= 7:
        return 2  # ideal
    return 1  # dengeli

def load_and_prepare_data():
    """Dataset'i yukle ve hazirla"""
    print("=" * 60)
    print("VERI HAZIRLAMA")
    print("=" * 60)
    
    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset yuklendi: {len(df)} satir")
    
    # Gereksiz kolonlari at
    drop_cols = ['Person ID', 'Occupation', 'Sleep Disorder', 'Blood Pressure']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Encoding
    df['Gender'] = df['Gender'].apply(encode_gender)
    df['BMI Category'] = df['BMI Category'].apply(encode_bmi)
    
    # Etiket olustur
    df['Label'] = df.apply(generate_label, axis=1)
    
    feature_cols = [
        'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI Category',
        'Heart Rate', 'Daily Steps'
    ]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Label'].values.astype(np.int64)
    
    # Etiket dagilimi
    unique, counts = np.unique(y, return_counts=True)
    label_names = ['riskli', 'dengeli', 'ideal']
    print(f"Etiket dagilimi: {dict(zip(label_names, counts))}")
    print(f"Feature sayisi: {len(feature_cols)}")
    
    return X, y, feature_cols


# =============================================================================
# MODEL MIMARILERI
# =============================================================================

class BaselineModel(nn.Module):
    """
    Model 1: Baseline (Mevcut basit model)
    Mimari: 9 -> 32 -> 16 -> 3
    """
    def __init__(self, input_dim=9):
        super().__init__()
        self.name = "Baseline"
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepNetwork(nn.Module):
    """
    Model 2: Deep Network
    Daha derin mimari + BatchNorm + LeakyReLU
    Mimari: 9 -> 64 -> 128 -> 64 -> 32 -> 16 -> 3
    """
    def __init__(self, input_dim=9):
        super().__init__()
        self.name = "DeepNetwork"
        self.network = nn.Sequential(
            # Katman 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            # Katman 2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            # Katman 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            # Katman 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Katman 5
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            
            # Output
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block for ResidualNetwork"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class ResidualNetwork(nn.Module):
    """
    Model 3: Residual Network
    Skip connections ile daha iyi gradient flow
    """
    def __init__(self, input_dim=9):
        super().__init__()
        self.name = "ResidualNetwork"
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(64)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionNetwork(nn.Module):
    """
    Model 4: Attention Network
    Self-attention ile feature importance ogrenme
    """
    def __init__(self, input_dim=9):
        super().__init__()
        self.name = "AttentionNetwork"
        self.input_dim = input_dim
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1)
        )
        
        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Feature-wise attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        # Feature-wise attention (hangi feature onemli?)
        feat_attn = self.feature_attention(x)
        x_weighted = x * feat_attn
        
        # Embedding
        embedded = self.embedding(x_weighted)
        
        # Classification
        output = self.classifier(embedded)
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleNetwork(nn.Module):
    """
    Model 5: Ensemble Network
    3 farkli alt-model + voting
    """
    def __init__(self, input_dim=9):
        super().__init__()
        self.name = "EnsembleNetwork"
        
        # Alt-model 1: Shallow wide
        self.model1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
        
        # Alt-model 2: Deep narrow
        self.model2 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Alt-model 3: Balanced with BatchNorm
        self.model3 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Ensemble weights (ogrenilebilir)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        
        # Weighted average
        weights = torch.softmax(self.ensemble_weights, dim=0)
        output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# EGITIM VE DEGERLENDIRME
# =============================================================================

def train_and_evaluate(model, X, y, n_splits=5, epochs=100, lr=0.001):
    """
    K-Fold Cross Validation ile model egit ve degerlendir
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Veriyi ayir
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize (her fold icin ayri)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        
        # Tensor'a cevir
        X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Model (her fold icin yeni instance)
        fold_model = model.__class__(input_dim=X.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fold_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Egitim
        fold_model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step(epoch_loss)
        
        # Degerlendirme
        fold_model.eval()
        with torch.no_grad():
            val_outputs = fold_model(X_val_t)
            _, preds = torch.max(val_outputs, 1)
            preds = preds.numpy()
        
        # Metrikleri hesapla
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average='weighted', zero_division=0)
        rec = recall_score(y_val, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        all_preds.extend(preds)
        all_labels.extend(y_val)
    
    training_time = time.time() - start_time
    
    # Ortalama sonuclar
    avg_results = {
        'model_name': model.name,
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'f1_std': np.std([r['f1'] for r in fold_results]),
        'training_time': training_time,
        'parameters': model.count_parameters(),
        'all_preds': all_preds,
        'all_labels': all_labels
    }
    
    return avg_results, fold_results


# =============================================================================
# GORSELLESTIRME
# =============================================================================

def plot_comparison(results):
    """Model karsilastirma grafikleri"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    model_names = [r['model_name'] for r in results]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # 1. Accuracy karsilastirma
    ax1 = axes[0, 0]
    accuracies = [r['accuracy'] * 100 for r in results]
    acc_stds = [r['accuracy_std'] * 100 for r in results]
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, yerr=acc_stds, capsize=5)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Karsilastirmasi (5-Fold CV)')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. F1-Score karsilastirma
    ax2 = axes[0, 1]
    f1_scores = [r['f1'] * 100 for r in results]
    f1_stds = [r['f1_std'] * 100 for r in results]
    bars2 = ax2.bar(model_names, f1_scores, color=colors, alpha=0.8, yerr=f1_stds, capsize=5)
    ax2.set_ylabel('F1-Score (%)')
    ax2.set_title('Model F1-Score Karsilastirmasi (5-Fold CV)')
    ax2.set_ylim(0, 100)
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{f1:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Egitim suresi
    ax3 = axes[1, 0]
    times = [r['training_time'] for r in results]
    bars3 = ax3.bar(model_names, times, color=colors, alpha=0.8)
    ax3.set_ylabel('Egitim Suresi (saniye)')
    ax3.set_title('Model Egitim Suresi Karsilastirmasi')
    for bar, t in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
    ax3.tick_params(axis='x', rotation=15)
    
    # 4. Parametre sayisi
    ax4 = axes[1, 1]
    params = [r['parameters'] for r in results]
    bars4 = ax4.bar(model_names, params, color=colors, alpha=0.8)
    ax4.set_ylabel('Parametre Sayisi')
    ax4.set_title('Model Karmasikligi (Parametre Sayisi)')
    for bar, p in zip(bars4, params):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{p:,}', ha='center', va='bottom', fontsize=9)
    ax4.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(ASSETS_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGrafik kaydedildi: {save_path}")
    
    plt.show()


def plot_confusion_matrices(results):
    """Her model icin confusion matrix"""
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
    
    class_names = ['Riskli', 'Dengeli', 'Ideal']
    
    for idx, (r, ax) in enumerate(zip(results, axes)):
        cm = confusion_matrix(r['all_labels'], r['all_preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{r['model_name']}\nAcc: {r['accuracy']*100:.1f}%")
        ax.set_ylabel('Gercek')
        ax.set_xlabel('Tahmin')
    
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(ASSETS_DIR, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.show()


def plot_radar_chart(results):
    """Radar chart ile cok boyutlu karsilastirma"""
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Hiz*', 'Basitlik*']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Acilari hesapla
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Kapatmak icin
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Her model icin
    max_time = max(r['training_time'] for r in results)
    max_params = max(r['parameters'] for r in results)
    
    for idx, r in enumerate(results):
        values = [
            r['accuracy'],
            r['precision'],
            r['recall'],
            r['f1'],
            1 - (r['training_time'] / max_time),  # Ters cevir (dusuk = iyi)
            1 - (r['parameters'] / max_params)     # Ters cevir (dusuk = iyi)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=r['model_name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performans Radar Grafigi\n(*ters metrik: yuksek=iyi)', pad=20)
    
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(ASSETS_DIR, 'radar_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Radar chart kaydedildi: {save_path}")
    
    plt.show()


def print_results_table(results):
    """Sonuclari tablo olarak yazdir"""
    
    print("\n" + "=" * 90)
    print("MODEL KARSILASTIRMA SONUCLARI (5-Fold Cross Validation)")
    print("=" * 90)
    
    print(f"\n{'Model':<20} {'Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<15} {'Sure':<10} {'Param':<10}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['model_name']:<20} "
              f"{r['accuracy']*100:.2f}% +-{r['accuracy_std']*100:.1f}  "
              f"{r['precision']*100:.2f}%      "
              f"{r['recall']*100:.2f}%      "
              f"{r['f1']*100:.2f}% +-{r['f1_std']*100:.1f}   "
              f"{r['training_time']:.1f}s      "
              f"{r['parameters']:,}")
    
    print("-" * 90)
    
    # En iyi modeli bul
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    fastest = min(results, key=lambda x: x['training_time'])
    simplest = min(results, key=lambda x: x['parameters'])
    
    print(f"\n[+] EN YUKSEK ACCURACY: {best_acc['model_name']} ({best_acc['accuracy']*100:.2f}%)")
    print(f"[+] EN YUKSEK F1-SCORE: {best_f1['model_name']} ({best_f1['f1']*100:.2f}%)")
    print(f"[+] EN HIZLI: {fastest['model_name']} ({fastest['training_time']:.1f}s)")
    print(f"[+] EN BASIT: {simplest['model_name']} ({simplest['parameters']:,} parametre)")
    
    print("\n" + "=" * 90)


# =============================================================================
# ANA FONKSIYON
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("   MINDTRACK ADVANCED - MODEL KARSILASTIRMA")
    print("   5 Farkli Neural Network Mimarisi")
    print("=" * 60 + "\n")
    
    # Veri yukle
    X, y, feature_cols = load_and_prepare_data()
    input_dim = len(feature_cols)
    
    # Modelleri tanimla
    models = [
        BaselineModel(input_dim),
        DeepNetwork(input_dim),
        ResidualNetwork(input_dim),
        AttentionNetwork(input_dim),
        EnsembleNetwork(input_dim)
    ]
    
    # Her modeli egit ve degerlendir
    all_results = []
    
    for i, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(models)}] {model.name} egitiliyor...")
        print(f"Parametre sayisi: {model.count_parameters():,}")
        print("=" * 60)
        
        results, fold_results = train_and_evaluate(
            model, X, y, 
            n_splits=5, 
            epochs=100, 
            lr=0.001
        )
        
        all_results.append(results)
        
        print(f"[OK] {model.name} tamamlandi!")
        print(f"  Accuracy: {results['accuracy']*100:.2f}% (+-{results['accuracy_std']*100:.1f})")
        print(f"  F1-Score: {results['f1']*100:.2f}% (+-{results['f1_std']*100:.1f})")
        print(f"  Sure: {results['training_time']:.1f}s")
    
    # Sonuclari yazdir
    print_results_table(all_results)
    
    # Grafikleri ciz
    print("\nGrafikler olusturuluyor...")
    plot_comparison(all_results)
    plot_confusion_matrices(all_results)
    plot_radar_chart(all_results)
    
    print("\n[OK] Karsilastirma tamamlandi!")
    print(f"[INFO] Grafikler kaydedildi: {ASSETS_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()
