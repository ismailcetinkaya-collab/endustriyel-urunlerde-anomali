# endustriyel-urunlerde-anomali

- Colab'da çalıştırma
- Google Colab⁠ adresini aç.
- İndirdiğin YOLO11_Anomaly_Colab_Test.ipynb dosyasını yükle.
- Notebook içindeki yükleme hücresinde yolo_anomaly_colab_bundle.zip paketini seç.
- Çalışma zamanı ayarlarından GPU'yu etkinleştir.
- Hücreleri sırayla çalıştır.
- Test ayarları
- Parametre Hızlı test Tam test
- SKU sayısı 3 15
- Test görüntüsü 120 6.000
- YOLO epoch 3 30
- Autoencoder epoch 3 40
- Önce hızlı testi çalıştırmanı öneririm. Sorunsuz tamamlanırsa ayar hücresinde FAST_TEST = False yaparak tam teste geçebilirsin.
- Kontrol notu: Python dosyasının sözdizimi derleme kontrolü başarılı. Ancak Colab üzerinde gerçek eğitim ve GPU çalıştırma testi henüz yapılmadı. Ayrıca sentetik veriden alınan metrikler gerçek üretim performansını kanıtlamaz; gerçek kamera görüntüleriyle ayrıca doğrulama gerekir.
