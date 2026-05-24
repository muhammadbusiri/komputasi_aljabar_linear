import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Mendefinisikan Ukuran Gambar & Path Lokal
IMG_SIZE = 100
dataset_path = "faces"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Folder '{dataset_path}' tidak ditemukan! Letakkan folder 'faces' di workspace Anda.")

# 2. Load Dataset Wajah Riil dari Folder 'faces'
def load_dataset(path):
    data = []
    labels = []
    for file in os.listdir(path):
        # Abaikan file uji yang diawali kata 'test'
        if file.lower().startswith('test'):
            continue
            
        # Ekstrak nama (identitas) dari nama file (misal: 'arnold01.jpg' -> 'arnold')
        name = ''.join([c for c in file.split('.')[0] if not c.isdigit()]).lower()
        
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img.flatten())
            labels.append(name)
            
    return np.array(data, dtype=np.float32), np.array(labels)

# Memuat dataset
X, y = load_dataset(dataset_path)

print("===== DETIL DATASET LATIH =====")
print("Jumlah gambar training :", len(X))
print("Ukuran matrix data     :", X.shape)
print("Identitas unik         :", list(np.unique(y)))

# 3. Tampilkan Beberapa Sampel Wajah Training Riil
unique_names = list(np.unique(y))
plt.figure(figsize=(10, 4))
for idx, name in enumerate(unique_names):
    first_idx = np.where(y == name)[0][0]
    face = X[first_idx].reshape(IMG_SIZE, IMG_SIZE)
    plt.subplot(1, len(unique_names), idx+1)
    plt.imshow(face, cmap='gray')
    plt.title(name.upper(), fontsize=11, fontweight='bold')
    plt.axis('off')
plt.suptitle("CONTOH GAMBAR WAJAH TRAINING RIIL", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. Normalisasi & Komputasi SVD
mean_face = np.mean(X, axis=0)
A = X - mean_face
U, S, VT = np.linalg.svd(A, full_matrices=False)

print("\n===== HASIL KOMPUTASI DEKOMPOSISI SVD =====")
print("Ukuran U (Bobot Training)  :", U.shape)
print("Ukuran Sigma (Singular)    :", np.diag(S).shape)
print("Ukuran VT (Eigenfaces)     :", VT.shape)

print("\n===== 5 SINGULAR VALUES TERBESAR =====")
for i in range(min(5, len(S))):
    print(f"Singular Value {i+1} = {S[i]:.4f}")

# 5. Visualisasi Wajah Eigenface (Ghostly Faces)
k = 10
if k > len(VT):
    k = len(VT)

eigenfaces = VT[:k]

plt.figure(figsize=(15, 6))
for i in range(k):
    plt.subplot(2, 5, i+1)
    face = eigenfaces[i].reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(face, cmap='coolwarm')
    plt.title(f"Eigenface {i+1}")
    plt.axis('off')
plt.suptitle("VISUALISASI EIGENFACES HASIL DEKOMPOSISI SVD", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Proyeksi Training Set & Pengujian Wajah Baru
projected_train = np.dot(A, eigenfaces.T)

# Cari gambar uji berawalan 'test' di folder 'faces'
# Secara default, kita cari 'testterminator1.jpg' (uji Arnold)
test_file_path = os.path.join(dataset_path, "testterminator1.jpg")
if not os.path.exists(test_file_path):
    test_files = [f for f in os.listdir(dataset_path) if f.lower().startswith('test')]
    if len(test_files) > 0:
        test_file_path = os.path.join(dataset_path, test_files[0])
        
print(f"\nMenggunakan Wajah Uji Lokal: {test_file_path}")

test_img = cv2.imread(test_file_path, cv2.IMREAD_GRAYSCALE)
if test_img is None:
    raise FileNotFoundError(f"Gagal memuat gambar uji dari: {test_file_path}")
test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))

# Normalisasi & Proyeksi Test Image
test_vector = test_img.flatten().astype(np.float32)
test_normalized = test_vector - mean_face
projected_test = np.dot(test_normalized, eigenfaces.T)

# Hitung Jarak Euclidean murni dengan NumPy
distances = np.linalg.norm(projected_train - projected_test, axis=1)

print("\n===== DETIL JARAK EUCLIDEAN TERDEKAT =====")
for name in unique_names:
    idxs = np.where(y == name)[0]
    min_dist_name = np.min(distances[idxs])
    print(f"Jarak Terdekat ke {name.upper()} = {min_dist_name:.4f}")

# Klasifikasi
best_match_idx = np.argmin(distances)
hasil = y[best_match_idx]

print("\n====================================")
print("     HASIL PENGENALAN WAJAH SVD")
print("====================================")
print("Gambar Uji        :", os.path.basename(test_file_path))
print("Prediksi Identitas:", hasil.upper())
print("Jarak Terdekat    :", distances[best_match_idx])

# Load Wajah Latih Terdekat untuk Visualisasi
matched_img_idx = np.where(y == hasil)[0][0]
matched_img = X[matched_img_idx].reshape(IMG_SIZE, IMG_SIZE)

# Tampilkan Perbandingan
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(test_img, cmap='gray')
plt.title(f"Wajah Uji: {os.path.basename(test_file_path)}", fontsize=11, fontweight='bold', color='orange')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(matched_img, cmap='gray')
plt.title(f"Cocok Terdekat: {hasil.upper()}", fontsize=11, fontweight='bold', color='green')
plt.axis('off')

plt.suptitle("HASIL AKHIR PENGENALAN WAJAH SISTEM SVD", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
