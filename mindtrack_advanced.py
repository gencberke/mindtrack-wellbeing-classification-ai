"""
MindTrack Advanced - Hibrit Zihinsel Sağlık Takip Sistemi
Programlama Dilleri Dersi Projesi

Bu uygulama iki aşamalı bir AI sistemi kullanıyor:
1. PyTorch modeli (dataset'ten eğitilmiş) -> riskli/dengeli/ideal sınıflandırma
2. OpenAI GPT -> kişiselleştirilmiş analiz ve tavsiyeler

Dataset: Kaggle Sleep Health and Lifestyle Dataset
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI

# .env'den API key yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

DATA_FILE = os.path.join(DATA_DIR, 'data.json')
MODEL_FILE = os.path.join(DATA_DIR, 'model_checkpoint.pth')
DATASET_FILE = os.path.join(ASSETS_DIR, 'Sleep_health_and_lifestyle_dataset.csv')


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def encode_gender(gender):
    """Cinsiyet -> sayı"""
    return 0 if gender == "Male" else 1


def encode_bmi(bmi):
    """BMI kategorisi -> sayı"""
    mapping = {
        "Underweight": 0,
        "Normal": 1,
        "Normal Weight": 1,
        "Overweight": 2,
        "Obese": 3
    }
    return mapping.get(bmi, 1)


def generate_label(row):
    """
    Dataset satırından etiket oluştur:
    0 = riskli, 1 = dengeli, 2 = ideal
    """
    stress = row['Stress Level']
    sleep = row['Sleep Duration']
    quality = row['Quality of Sleep']
    
    if stress >= 7 and sleep <= 6 and quality <= 5:
        return 0  # riskli
    elif stress <= 4 and sleep >= 7 and quality >= 7:
        return 2  # ideal
    return 1  # dengeli


def get_valid_input(prompt, min_val, max_val, is_float=False):
    """Kullanıcıdan geçerli sayı al"""
    while True:
        try:
            val = input(prompt)
            val = float(val) if is_float else int(val)
            if min_val <= val <= max_val:
                return val
            print(f"{min_val}-{max_val} arası bir değer gir.")
        except ValueError:
            print("Geçersiz, sayı girmelisin.")


# =============================================================================
# JSON DEPOLAMA
# =============================================================================

def load_entries():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_entries(entries):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)


def upsert_entry(entry):
    entries = load_entries()
    date = entry.get('date')
    
    found = False
    for i, e in enumerate(entries):
        if e.get('date') == date:
            entries[i] = entry
            found = True
            break
    
    if not found:
        entries.append(entry)
    
    entries.sort(key=lambda x: x['date'])
    save_entries(entries)


# =============================================================================
# PYTORCH MODELİ
# =============================================================================

class WellbeingClassifier(nn.Module):
    """
    3 katmanlı feedforward neural network
    9 input -> 32 -> 16 -> 3 output (riskli/dengeli/ideal)
    """
    def __init__(self, input_dim=9):
        super(WellbeingClassifier, self).__init__()
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


def preprocess_data(csv_path):
    """Kaggle datasetini model için hazırla"""
    df = pd.read_csv(csv_path)
    
    # gereksiz kolonları at
    drop_cols = ['Person ID', 'Occupation', 'Sleep Disorder', 'Blood Pressure']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # encoding
    df['Gender'] = df['Gender'].apply(encode_gender)
    df['BMI Category'] = df['BMI Category'].apply(encode_bmi)
    
    # etiket
    df['Label'] = df.apply(generate_label, axis=1)
    
    feature_cols = [
        'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI Category',
        'Heart Rate', 'Daily Steps'
    ]
    
    X = df[feature_cols].copy()
    y = df['Label'].copy()
    
    return X, y, feature_cols


def train_model():
    """Dataset'ten PyTorch modeli eğit"""
    print("Dataset yükleniyor...")
    
    if not os.path.exists(DATASET_FILE):
        print(f"HATA: Dataset bulunamadı!")
        print(f"Beklenen konum: {DATASET_FILE}")
        return False
    
    X, y, feature_cols = preprocess_data(DATASET_FILE)
    print(f"Dataset: {len(X)} satır, {len(feature_cols)} özellik")
    
    # etiket dağılımı
    print(f"Etiket dağılımı: {dict(y.value_counts())}")
    
    # tensorlara çevir
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    # normalizasyon
    mean = X_tensor.mean(dim=0)
    std = X_tensor.std(dim=0)
    std[std == 0] = 1.0
    X_normalized = (X_tensor - mean) / std
    
    # train/test split (basit)
    split = int(len(X) * 0.8)
    X_train, X_test = X_normalized[:split], X_normalized[split:]
    y_train, y_test = y_tensor[:split], y_tensor[split:]
    
    # dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # model
    model = WellbeingClassifier(len(feature_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # eğitim
    print("\nEğitim başlıyor...")
    model.train()
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # test accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).float().mean()
        print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # kaydet
    os.makedirs(DATA_DIR, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'mean': mean,
        'std': std,
        'classes': ['riskli', 'dengeli', 'ideal']
    }
    torch.save(checkpoint, MODEL_FILE)
    print(f"Model kaydedildi: {MODEL_FILE}")
    return True


def load_model():
    """Eğitilmiş modeli yükle"""
    if not os.path.exists(MODEL_FILE):
        return None, None
    
    checkpoint = torch.load(MODEL_FILE, weights_only=False)
    model = WellbeingClassifier(len(checkpoint['feature_cols']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def pytorch_predict(features_dict):
    """
    PyTorch modeli ile sınıflandırma yap
    Returns: (label, confidence)
    """
    model, checkpoint = load_model()
    
    if model is None:
        return "dengeli", 0.0
    
    # özellikleri sırala
    feature_order = checkpoint['feature_cols']
    input_vals = []
    
    for col in feature_order:
        val = features_dict.get(col)
        if col == 'Gender':
            val = encode_gender(val)
        elif col == 'BMI Category':
            val = encode_bmi(val)
        input_vals.append(float(val))
    
    # tensor + normalize
    input_tensor = torch.tensor(input_vals, dtype=torch.float32)
    normalized = (input_tensor - checkpoint['mean']) / checkpoint['std']
    normalized = normalized.unsqueeze(0)
    
    # tahmin
    with torch.no_grad():
        outputs = model(normalized)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        label = checkpoint['classes'][pred_idx.item()]
    
    return label, confidence.item()


# =============================================================================
# GPT ANALİZ (2. AŞAMA)
# =============================================================================

def gpt_analyze(metrics, diary_note, pytorch_prediction, confidence, recent_entries=None):
    """
    GPT'ye PyTorch sonucunu ve verileri gönder, detaylı analiz al
    """
    
    # geçmiş özeti
    history = ""
    if recent_entries and len(recent_entries) > 0:
        history = "\n\nSon günlerin özeti:\n"
        for e in recent_entries[-5:]:
            m = e['metrics']
            history += f"- {e['date']}: Uyku {m['sleep_hours']}s, Stres {m['stress_score']}/10, Ruh hali {m['mood_score']}/5\n"
    
    prompt = f"""Sen bir zihinsel sağlık asistanısın. Yapay zeka modelimiz kullanıcıyı analiz etti, şimdi sen detaylı değerlendirme yap.

MODEL TAHMİNİ: {pytorch_prediction.upper()} (güven: %{confidence*100:.0f})

BUGÜNKÜ VERİLER:
- Uyku: {metrics['sleep_hours']} saat (kalite: {metrics['quality_of_sleep']}/10)
- Ruh hali: {metrics['mood_score']}/5
- Stres: {metrics['stress_score']}/10
- Sosyal etkileşim: {metrics['social_score']}/5
- Fiziksel aktivite: {metrics['physical_activity']} dk
- Adım: {metrics['daily_steps']}
- Kalp atışı: {metrics['heart_rate']} bpm

GÜNLÜK NOTU:
{diary_note if diary_note else "(Boş)"}
{history}

Model "{pytorch_prediction}" dedi. Bu sonucu değerlendir ve Türkçe yanıt ver:

1. MODEL YORUMU: Model neden bu sonuca varmış olabilir? (1-2 cümle)

2. DETAYLI ANALİZ: Verilerde dikkat çeken noktalar (2-3 cümle)

3. KİŞİSEL TAVSİYELER: 3 somut öneri (madde madde)

Samimi ama profesyonel ol. Abartma."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen deneyimli bir zihinsel sağlık danışmanısın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT hatası: {e}"


def gpt_chat(user_message, entries):
    """Serbest sohbet modu"""
    
    data_summary = "Kayıtlı veri yok."
    if entries:
        recent = entries[-7:]
        data_summary = "Son kayıtlar:\n"
        for e in recent:
            m = e['metrics']
            pred = e.get('pytorch_prediction', '?')
            data_summary += f"- {e['date']}: {pred.upper()} | Uyku {m['sleep_hours']}s, Stres {m['stress_score']}/10\n"
    
    prompt = f"""Kullanıcıyla Türkçe sohbet et. Verileri kullanabilirsin.

VERİLER:
{data_summary}

KULLANICI:
{user_message}

Kısa ve samimi yanıt ver."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen empatik bir sağlık asistanısın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata: {e}"


def gpt_weekly_report(entries):
    """Haftalık GPT raporu"""
    weekly = get_weekly_entries(entries)
    
    if not weekly:
        return "Son 7 günde veri yok."
    
    data_text = ""
    for e in weekly:
        m = e['metrics']
        pred = e.get('pytorch_prediction', '?')
        data_text += f"{e['date']}: {pred.upper()}\n"
        data_text += f"  Uyku: {m['sleep_hours']}s, Stres: {m['stress_score']}/10, Ruh hali: {m['mood_score']}/5\n"
        if e.get('diary_entry'):
            data_text += f"  Not: {e['diary_entry'][:60]}...\n"
        data_text += "\n"
    
    prompt = f"""Bu haftalık zihinsel sağlık verilerini analiz et (Türkçe):

{data_text}

Rapor formatı:
1. HAFTA ÖZETİ (2-3 cümle)
2. TRENDLER (iyiye/kötüye giden)
3. ÖNE ÇIKAN GÜNLER
4. GELECEK HAFTA İÇİN ÖNERİLER (3 madde)"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen bir sağlık danışmanısın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata: {e}"


# =============================================================================
# ANALİTİK
# =============================================================================

def get_weekly_entries(entries):
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    
    weekly = []
    for e in entries:
        try:
            entry_date = datetime.strptime(e['date'], "%Y-%m-%d")
            if entry_date >= week_ago:
                weekly.append(e)
        except:
            pass
    return weekly


def calculate_weekly_summary(entries):
    weekly = get_weekly_entries(entries)
    if not weekly:
        return None
    
    sleep_vals = [e['metrics']['sleep_hours'] for e in weekly]
    mood_vals = [e['metrics']['mood_score'] for e in weekly]
    stress_vals = [e['metrics']['stress_score'] for e in weekly]
    
    # tahmin dağılımı
    predictions = [e.get('pytorch_prediction', 'dengeli') for e in weekly]
    pred_counts = {p: predictions.count(p) for p in set(predictions)}
    
    return {
        "count": len(weekly),
        "avg_sleep": sum(sleep_vals) / len(sleep_vals),
        "avg_mood": sum(mood_vals) / len(mood_vals),
        "avg_stress": sum(stress_vals) / len(stress_vals),
        "predictions": pred_counts
    }


def plot_weekly_data(entries):
    weekly = get_weekly_entries(entries)
    if not weekly:
        print("Grafik için veri yok.")
        return
    
    dates = [e['date'][-5:] for e in weekly]  # sadece ay-gün
    sleep = [e['metrics']['sleep_hours'] for e in weekly]
    mood = [e['metrics']['mood_score'] for e in weekly]
    stress = [e['metrics']['stress_score'] for e in weekly]
    
    # tahmin renkleri
    colors = []
    for e in weekly:
        pred = e.get('pytorch_prediction', 'dengeli')
        if pred == 'riskli':
            colors.append('red')
        elif pred == 'ideal':
            colors.append('green')
        else:
            colors.append('orange')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # uyku
    axes[0].bar(dates, sleep, color=colors, alpha=0.7)
    axes[0].axhline(y=7, color='green', linestyle='--', alpha=0.5, label='İdeal (7s)')
    axes[0].set_ylabel('Uyku (saat)')
    axes[0].set_title('Haftalık Trend - Renkler: 🔴Riskli 🟠Dengeli 🟢İdeal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ruh hali
    axes[1].plot(dates, mood, 's-', color='purple', linewidth=2, markersize=8)
    axes[1].fill_between(dates, mood, alpha=0.3, color='purple')
    axes[1].set_ylabel('Ruh Hali (1-5)')
    axes[1].set_ylim(0.5, 5.5)
    axes[1].grid(True, alpha=0.3)
    
    # stres
    axes[2].plot(dates, stress, '^-', color='red', linewidth=2, markersize=8)
    axes[2].fill_between(dates, stress, alpha=0.2, color='red')
    axes[2].axhline(y=7, color='red', linestyle='--', alpha=0.5, label='Yüksek stres sınırı')
    axes[2].set_ylabel('Stres (1-10)')
    axes[2].set_xlabel('Tarih')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # kaydet
    os.makedirs(ASSETS_DIR, exist_ok=True)
    filename = f'weekly_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    filepath = os.path.join(ASSETS_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Grafik kaydedildi: {filepath}")
    
    try:
        plt.show()
    except:
        pass


# =============================================================================
# ANA MENÜ
# =============================================================================

def enter_data():
    clear_screen()
    print("=" * 50)
    print("   HİBRİT ANALİZ - PyTorch + GPT")
    print("=" * 50)
    
    # tarih
    date_str = input("\nTarih (YYYY-AA-GG, bugün için boş): ").strip()
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # metrikler
    print("\n--- Temel Bilgiler ---")
    sleep = get_valid_input("Uyku süresi (saat): ", 0, 24, is_float=True)
    quality = get_valid_input("Uyku kalitesi (1-10): ", 1, 10)
    mood = get_valid_input("Ruh hali (1-5): ", 1, 5)
    stress = get_valid_input("Stres (1-10): ", 1, 10)
    social = get_valid_input("Sosyal etkileşim (1-5): ", 1, 5)
    
    print("\n--- Fiziksel Metrikler ---")
    activity = get_valid_input("Fiziksel aktivite (dk): ", 0, 1440)
    steps = get_valid_input("Adım sayısı: ", 0, 100000)
    heart_rate = get_valid_input("Kalp atışı (bpm): ", 40, 200)
    
    print("\n--- Ek Bilgiler (Model için) ---")
    gender = input("Cinsiyet (Male/Female): ").strip().capitalize()
    if gender not in ['Male', 'Female']:
        gender = 'Male'
    
    age = get_valid_input("Yaş: ", 10, 100)
    
    bmi = input("BMI (Underweight/Normal/Overweight/Obese): ").strip().capitalize()
    if bmi not in ['Underweight', 'Normal', 'Overweight', 'Obese']:
        bmi = 'Normal'
    
    diary = input("\nGünlük notu: ")
    
    # metrikleri hazırla
    metrics = {
        "sleep_hours": sleep,
        "quality_of_sleep": quality,
        "mood_score": mood,
        "stress_score": stress,
        "social_score": social,
        "physical_activity": activity,
        "daily_steps": steps,
        "heart_rate": heart_rate,
        "age": age,
        "gender": gender,
        "bmi_category": bmi
    }
    
    # PyTorch için feature dict
    pytorch_features = {
        'Gender': gender,
        'Age': age,
        'Sleep Duration': sleep,
        'Quality of Sleep': quality,
        'Physical Activity Level': activity,
        'Stress Level': stress,
        'BMI Category': bmi,
        'Heart Rate': heart_rate,
        'Daily Steps': steps
    }
    
    # 1. AŞAMA: PyTorch tahmini
    print("\n PyTorch modeli analiz ediyor...")
    prediction, confidence = pytorch_predict(pytorch_features)
    print(f"   Sonuç: {prediction.upper()} (güven: %{confidence*100:.0f})")
    
    # 2. AŞAMA: GPT analizi
    print("\n GPT detaylı analiz yapıyor...")
    entries = load_entries()
    gpt_response = gpt_analyze(metrics, diary, prediction, confidence, entries)
    
    # kaydet
    entry = {
        "date": date_str,
        "metrics": metrics,
        "diary_entry": diary,
        "pytorch_prediction": prediction,
        "pytorch_confidence": confidence,
        "gpt_analysis": gpt_response
    }
    upsert_entry(entry)
    
    # sonuçları göster
    print("\n" + "=" * 50)
    print(f" MODEL TAHMİNİ: {prediction.upper()}")
    print("=" * 50)
    print(gpt_response)
    print("=" * 50)
    
    input("\nDevam etmek için Enter...")


def chat_mode():
    clear_screen()
    print("=== GPT Sohbet Modu ===")
    print("(Çıkmak için 'q')\n")
    
    entries = load_entries()
    
    while True:
        user_input = input("\nSen: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue
        
        print("\n Düşünüyor...")
        response = gpt_chat(user_input, entries)
        print(f"\nAsistan: {response}")


def main_menu():
    while True:
        clear_screen()
        print("=" * 55)
        print("   MINDTRACK ADVANCED")
        print("   Hibrit AI Sistemi (PyTorch + GPT)")
        print("=" * 55)
        
        # model durumu
        model, _ = load_model()
        model_status = "Hazır" if model else "Eğitilmedi"
        print(f"\n   Model durumu: {model_status}")
        
        print("\n   1) Günlük veri gir (Hibrit Analiz)")
        print("   2) GPT ile sohbet")
        print("   3) Haftalık özet")
        print("   4) Haftalık GPT raporu")
        print("   5) Grafik çiz")
        print("   6) Modeli eğit")
        print("   7) Çıkış")
        
        choice = input("\n   Seçimin: ")
        
        if choice == '1':
            if not model:
                print("\n Önce modeli eğitmelisin! (seçenek 6)")
                input("Enter'a bas...")
            else:
                enter_data()
                
        elif choice == '2':
            chat_mode()
            
        elif choice == '3':
            clear_screen()
            entries = load_entries()
            summary = calculate_weekly_summary(entries)
            
            if summary:
                print("=== Haftalık Özet ===\n")
                print(f"Giriş sayısı: {summary['count']}")
                print(f"Ort. uyku: {summary['avg_sleep']:.1f} saat")
                print(f"Ort. ruh hali: {summary['avg_mood']:.1f}/5")
                print(f"Ort. stres: {summary['avg_stress']:.1f}/10")
                print(f"\nTahmin dağılımı: {summary['predictions']}")
            else:
                print("Son 7 günde veri yok.")
            input("\nEnter'a bas...")
            
        elif choice == '4':
            clear_screen()
            print("GPT rapor hazırlıyor...\n")
            entries = load_entries()
            report = gpt_weekly_report(entries)
            print("=" * 50)
            print("HAFTALIK GPT RAPORU")
            print("=" * 50)
            print(report)
            print("=" * 50)
            input("\nEnter'a bas...")
            
        elif choice == '5':
            clear_screen()
            entries = load_entries()
            plot_weekly_data(entries)
            input("\nEnter'a bas...")
            
        elif choice == '6':
            clear_screen()
            train_model()
            input("\nEnter'a bas...")
            
        elif choice == '7':
            print("\nGörüşürüz! ")
            sys.exit()
        
        else:
            input("Geçersiz seçim. Enter...")


if __name__ == "__main__":
    # API key kontrolü
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 50)
        print("HATA: OPENAI_API_KEY bulunamadı!")
        print("=" * 50)
        print("\n.env dosyası oluştur:")
        print("OPENAI_API_KEY=sk-proj-xxxxx")
        sys.exit(1)
    
    main_menu()
