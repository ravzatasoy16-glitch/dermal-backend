import os
import io
import time 
import base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GEMINI İSTEMCİSİ (Çevresel Değişkenden Alınır) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- SUPABASE BAĞLANTISI (Çevresel Değişkenlerden Alınır) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- SINIF SIRASI (7 Sınıf - Alfabetik Sıraya Göre) ---
HASTALIK_ISIMLERI = [
    "acne", "benign_nv", "diger", "eczema", "melanoma", "psoriasis", "tinea"
]

# --- DOSYA YOLU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cilt_veriseti_7_sinif_model.pth")

# --- MODELİ YÜKLEME (MobileNetV2 Olarak Güncellendi) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"KRİTİK HATA: Model dosyası bulunamadı -> {MODEL_PATH}")

    # 1. MobileNetV2 iskeletini çağırıyoruz
    model = models.mobilenet_v2(weights=None)
    
    # 2. Sınıflandırıcı (classifier) katmanını 7 sınıfa göre uyarlıyoruz
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(HASTALIK_ISIMLERI))
    
    # 3. Eğittiğimiz ağırlıkları (.pth) modele yüklüyoruz
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("MobileNetV2 Modeli başarıyla yüklendi ve cihazda aktif:", DEVICE)
    return model

model = load_model()

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- TELEFONDAN GELECEK JSON VERİSİ İÇİN YAPI ---
class DegerlendirmeRequest(BaseModel):
    tc_no: str
    sikayet_detayi: str
    bolge: str
    sure: str
    aile_oykusu: str
    tarih: Optional[str] = None
    foto1_base64: str
    foto2_base64: Optional[str] = None
    foto3_base64: Optional[str] = None

# --- ENDPOINT ---
@app.post("/on-degerlendirme-json")
async def on_degerlendirme_json(req: DegerlendirmeRequest):
    try:
        def decode_b64(b64_str):
            if not b64_str: return None
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            return base64.b64decode(b64_str)

        image_bytes = decode_b64(req.foto1_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 3)
            
        turkce_isimler = {
            "acne": "Akne (Sivilce)",
            "benign_nv": "İyi Huylu Ben",
            "diger": "Bilinmeyen/Diğer",
            "eczema": "Egzama",
            "melanoma": "Melanom",
            "psoriasis": "Sedef Hastalığı",
            "tinea": "Mantar Enfeksiyonu"
        }

        top_3_listesi = []
        for i in range(3):
            idx = top_catid[i].item()
            conf = round(top_prob[i].item() * 100, 2)
            name = HASTALIK_ISIMLERI[idx]
            gercek_isim = turkce_isimler.get(name, name)
            top_3_listesi.append(f"{gercek_isim} (%{conf})")

        yan_yana_tahminler = ", ".join(top_3_listesi)

        en_yuksek_idx = top_catid[0].item()
        en_yuksek_sinif_adi = HASTALIK_ISIMLERI[en_yuksek_idx]
        en_yuksek_gercek_isim = turkce_isimler.get(en_yuksek_sinif_adi, en_yuksek_sinif_adi)

        if en_yuksek_sinif_adi == "diger":
            ai_raporu = (
                "Bu lezyon sistemin tanımlı ana hastalık sınıfları ile eşleşmemiştir; diğer sınıf kategorisinde yer almaktadır. "
                "Yapay zeka bu aşamada klinik bir ön değerlendirme veya tedavi önerisi oluşturamamaktadır. "
                "Uzman hekimin detaylı fiziki muayenesi ve değerlendirmesi önerilir."
            )
        else:
            prompt = f"""
            Sen profesyonel bir dermatoloji asistanısın. Yazacağın rapor kurumsal bir hastane raporu ciddiyetinde ve formatında olmalıdır.

            HASTA BİLGİLERİ:
            - Şikayet: "{req.sikayet_detayi}"
            - Bölge: "{req.bolge}"
            - Süre: "{req.sure}"
            - Aile Öyküsü: "{req.aile_oykusu}"
            
            YAPAY ZEKA MODEL TAHMİNLERİ: {yan_yana_tahminler}
            
            KATI KURALLAR:
            1. KESİNLİKLE 1., 2., 3. gibi numaralı başlıklar veya madde imleri KULLANMA. Rapor alt alta inen 3 düz paragraftan oluşmalıdır.
            2. KESİNLİKLE tıbbi jargon (intravenöz, eritematöz vb.) ve "İşbu rapor" gibi eski/resmi kelimeler kullanma.
            3. Metin içinde hiçbir kelimeyi kalınlaştırma (bold yapma) ve yıldız (*) işareti kullanma.
            4. Yüzdelik oranları KESİNLİKLE yazıyla (örn: yüzde seksen beş) YAZMA! Kesinlikle matematiksel formatta (% sembolü ve rakam ile, örn: %85.3) yaz.
            
            RAPOR İSKELETİ (Sadece aşağıdaki 3 paragrafı yaz):
            [Paragraf 1 - Klinik Sentez]: Hastanın şikayet bölgesi, süresi, aile öyküsü ile görsel model analiz sonuçlarını (% sembolü kullanarak) harmanlayarak mantıklı bir bütünlüğe kavuştur.
            [Paragraf 2 - Yaşam Tarzı ve Ürün Tavsiyeleri]: Çıkan en yüksek ihtimalli rahatsızlığa (%100 özgü) tavsiyeler ver. Gerçekten o hastalığa iyi gelecek veya uzak durulması gereken spesifik öneriler yaz. 
            [Paragraf 3 - Yasal Uyarı]: ! Bu raporun bir yapay zeka klinik karar destek sistemi ön değerlendirmesi olduğu ve kesin tanı niteliği taşımadığı, nihai kararın fiziki muayene ile uzman hekime ait olduğunu belirten modern bir uyarı cümlesi yaz ve cümlenin sonuna da ünlem koy !
            """

            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                ai_raporu = response.text.strip()
            except Exception as gemini_hata:
                print("Gemini API Hatası (Yedek Rapor Devrede):", gemini_hata)
                
                ai_raporu = f"""Hastanın "{req.bolge}" bölgesinde "{req.sure}" süredir devam eden "{req.sikayet_detayi}" şikayeti ve iletilen görsel veriler yapay zeka altyapımız ile değerlendirilmiştir. Görüntü analizi sonucunda yapay zeka modelinin tespit ettiği bulgular sırasıyla şöyledir: {yan_yana_tahminler}. Aile öyküsü ({req.aile_oykusu}) bu doğrultuda klinik tabloya eklenmiştir.

Çıkan en yüksek ihtimalli ({en_yuksek_gercek_isim}) ön bulgusuna yönelik olarak; şikayet bölgesinin hijyenine dikkat edilmesi, irritan maddelerden kaçınılması ve cilt bariyerini destekleyici dermokozmetik yaklaşımlar tercih edilmesi önerilmektedir. Spesifik lezyon yönetimi hastanın güncel durumuna göre planlanmalıdır.

! Bu raporun bir yapay zeka klinik karar destek sistemi ön değerlendirmesi olduğu ve kesin tanı niteliği taşımadığı, nihai kararın fiziki muayene ile uzman hekime ait olduğu unutulmamalıdır !"""

        timestamp = int(time.time())
        foto1_url, foto2_url, foto3_url = None, None, None

        try:
            path1 = f"{req.tc_no}_{timestamp}_f1.jpg"
            supabase.storage.from_("fotograflar").upload(path1, image_bytes, {"content-type": "image/jpeg"})
            foto1_url = supabase.storage.from_("fotograflar").get_public_url(path1)
        except Exception as e:
            print("Foto1 Storage Yükleme Hatası:", e)

        if req.foto2_base64:
            try:
                f2_bytes = decode_b64(req.foto2_base64)
                path2 = f"{req.tc_no}_{timestamp}_f2.jpg"
                supabase.storage.from_("fotograflar").upload(path2, f2_bytes, {"content-type": "image/jpeg"})
                foto2_url = supabase.storage.from_("fotograflar").get_public_url(path2)
            except Exception as e:
                print("Foto2 Storage Yükleme Hatası:", e)

        if req.foto3_base64:
            try:
                f3_bytes = decode_b64(req.foto3_base64)
                path3 = f"{req.tc_no}_{timestamp}_f3.jpg"
                supabase.storage.from_("fotograflar").upload(path3, f3_bytes, {"content-type": "image/jpeg"})
                foto3_url = supabase.storage.from_("fotograflar").get_public_url(path3)
            except Exception as e:
                print("Foto3 Storage Yükleme Hatası:", e)

        guncel_tarih_saat = datetime.now().strftime("%m-%d-%Y %H:%M:%S")

        kayit_verisi = {
            "tc_no": req.tc_no,
            "detay": req.sikayet_detayi,          
            "bolge": req.bolge,
            "sure": req.sure,
            "aile_oykusu": req.aile_oykusu,
            "ai_on_tani": ai_raporu,
            "foto1": foto1_url,
            "foto2": foto2_url,
            "foto3": foto3_url,
            "durum": "inceliyor",
            "tarih": guncel_tarih_saat  
        }

        supabase.table("analizler").insert(kayit_verisi).execute()

        return {
            "status": "success",
            "supabase_kayit_durumu": "Başarıyla kaydedildi"
        }

    except Exception as e:            
        raise HTTPException(status_code=500, detail=str(e))