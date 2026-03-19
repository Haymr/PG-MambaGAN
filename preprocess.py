"""
DICOM → NPY Veri Ön İşleme
============================
Mayo Clinic LDCT veri setini model eğitimi için hazırlar.

Adımlar:
1. DICOM dosyalarını oku
2. HU (Hounsfield Unit) dönüşümü
3. [-1000, 1000] HU aralığına clip
4. 256×256 (veya 512×512) boyutuna resize
5. [-1, 1] aralığına normalize
6. NPY formatında kaydet

Kullanım:
    python preprocess.py --input /path/to/DICOM --output /path/to/npy_output
    python preprocess.py --input /path/to/DICOM --output /path/to/npy_output --size 512
"""

import os
import argparse
import numpy as np

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom yüklü değil: pip install pydicom")

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python yüklü değil: pip install opencv-python")


# =============================================================================
# Sabitler
# =============================================================================
HU_MIN = -1000
HU_MAX = 1000


# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================

def find_dose_folders(patient_path):
    """
    Hasta klasöründe Low Dose ve Full Dose klasörlerini bulur.
    Projeksiyon (proj/sino) klasörlerini atlar.

    Returns:
        (low_dose_path, high_dose_path) veya (None, None)
    """
    low_p = None
    high_p = None

    for root, dirs, files in os.walk(patient_path):
        for d in dirs:
            d_lower = d.lower()

            # Projeksiyon klasörlerini atla
            if "proj" in d_lower or "sino" in d_lower:
                continue

            if "low dose" in d_lower or "low_dose" in d_lower or "quarter" in d_lower:
                low_p = os.path.join(root, d)
            elif "full dose" in d_lower or "full_dose" in d_lower or "high dose" in d_lower or "normal" in d_lower:
                high_p = os.path.join(root, d)

    return low_p, high_p


def dicom_to_hu(dcm):
    """DICOM pixel array'i HU birimine dönüştür"""
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    slope = getattr(dcm, 'RescaleSlope', 1)
    image = dcm.pixel_array.astype(np.float32) * slope + intercept
    return image


def normalize_image(image, img_size=(256, 256)):
    """
    CT görüntüsünü normalize et.
    HU → clip → resize → [-1, 1]
    """
    # 1. HU windowing
    image = np.clip(image, HU_MIN, HU_MAX)

    # 2. Resize
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)

    # 3. [HU_MIN, HU_MAX] → [0, 1] → [-1, 1]
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    image = (image * 2) - 1

    return image.astype(np.float32)


# =============================================================================
# Ana İşleme
# =============================================================================

def process_patient(patient_path, output_dir, img_size, patient_name):
    """Tek bir hastanın verilerini işle"""
    low_path, high_path = find_dose_folders(patient_path)

    if not low_path or not high_path:
        print(f"  ⚠️  {patient_name}: Low/High dose klasörleri bulunamadı, atlanıyor.")
        return 0

    low_files = sorted([f for f in os.listdir(low_path) if f.endswith('.dcm') or not f.startswith('.')])
    high_files = sorted([f for f in os.listdir(high_path) if f.endswith('.dcm') or not f.startswith('.')])

    min_len = min(len(low_files), len(high_files))
    count = 0

    for i in range(min_len):
        try:
            low_dcm = pydicom.dcmread(os.path.join(low_path, low_files[i]))
            high_dcm = pydicom.dcmread(os.path.join(high_path, high_files[i]))

            low_img = normalize_image(dicom_to_hu(low_dcm), img_size)
            high_img = normalize_image(dicom_to_hu(high_dcm), img_size)

            save_name = f"{patient_name}_{i:04d}.npy"
            np.save(os.path.join(output_dir, "trainA", save_name), low_img)
            np.save(os.path.join(output_dir, "trainB", save_name), high_img)
            count += 1

        except Exception as e:
            print(f"  ❌ Hata ({patient_name} - slice {i}): {e}")
            continue

    return count


def main():
    parser = argparse.ArgumentParser(description='DICOM → NPY Veri Ön İşleme')
    parser.add_argument('--input', type=str, required=True,
                        help='Ham DICOM veri seti klasörü (hasta alt klasörleri içeren)')
    parser.add_argument('--output', type=str, required=True,
                        help='İşlenmiş NPY çıktı klasörü')
    parser.add_argument('--size', type=int, default=256,
                        help='Çıktı görüntü boyutu (default: 256)')
    args = parser.parse_args()

    img_size = (args.size, args.size)

    # Çıktı klasörlerini oluştur
    os.makedirs(os.path.join(args.output, "trainA"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "trainB"), exist_ok=True)

    # Hasta klasörlerini bul
    try:
        patients = sorted([
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
            and not d.startswith('.')
        ])
    except FileNotFoundError:
        print(f"❌ HATA: '{args.input}' yolu bulunamıyor!")
        return

    print(f"📁 Kaynak: {args.input}")
    print(f"📂 Çıktı: {args.output}")
    print(f"📐 Boyut: {args.size}×{args.size}")
    print(f"👥 Bulunan hasta sayısı: {len(patients)}")
    print(f"\n{'='*50}")
    print("İşlem başlıyor...")
    print(f"{'='*50}\n")

    total = 0
    for patient in patients:
        patient_path = os.path.join(args.input, patient)
        count = process_patient(patient_path, args.output, img_size, patient)
        if count > 0:
            print(f"  ✅ {patient}: {count} slice işlendi")
        total += count

    print(f"\n{'='*50}")
    print(f"✅ TAMAMLANDI!")
    print(f"   Toplam slice: {total}")
    print(f"   trainA (LDCT): {args.output}/trainA/")
    print(f"   trainB (NDCT): {args.output}/trainB/")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
