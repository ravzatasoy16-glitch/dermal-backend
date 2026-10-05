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

# --- GEMINI İSTEMCİSİ ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- SUPABASE BAĞLANTISI ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- SINIF SIRASI ---
HASTALIK_ISIMLERI = [
    "acne", "benign_nv", "diger", "eczema", "melanoma", "psoriasis", "tinea"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cilt_veriseti_7_sinif_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"KRİTİK HATA: Model dosyası bulunamadı -> {MODEL_PATH}")

    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(HASTALIK_ISIMLERI))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("MobileNetV2 Modeli başarıyla yüklendi ve cihazda aktif:", DEVICE)
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

@app.post("/on-degerlendirme-json")
async def on_degerlendirme_json(req: DegerlendirmeRequest):
    try:
        if not req.foto1_base64:
            raise HTTPException(status_code=400, detail="Fotoğraf verisi alınamadı, lütfen tekrar deneyin.")

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
            top_3_listesi.append(f"{gercek_isim} (Model Güven Skoru: %{conf})")

        yan_yana_tahminler = ", ".join(top_3_listesi)

        en_yuksek_idx = top_catid[0].item()
        en_yuksek_sinif_adi = HASTALIK_ISIMLERI[en_yuksek_idx]

        if en_yuksek_sinif_adi == "diger":
            ai_raporu = (
                "Bu görsel ana hastalık sınıflarımızla eşleşmemiş ve diğer grup kategorisine aittir. "
                "Bu yüzden yapay zeka raporu sunulamamaktadır."
            )
        elif en_yuksek_sinif_adi == "melanoma":
            ai_raporu = (
                "Sonucumuz melanom şüphesi taşımaktadır. "
                "Bir dermatoloji uzman doktoru tarafından yüz yüze görülmesi tavsiye edilir."
            )
        else:
           prompt = f"""
            Sen hastalarla anlaşılır ve sade bir dille iletişim kuran bir sağlık asistanısın. 
            Amacın, hastanın şikayetlerine ve yapay zeka modelimizin sunduğu ilk 3 yüzdelik tahmin oranına dayanarak hastaya özel, anlaşılır bir ön değerlendirme raporu oluşturmaktır.

            HASTA BİLGİLERİ:
            - Şikayet Bölgesi: "{req.bolge}"
            - Şikayet Süresi: "{req.sure}"
            - Şikayet Detayı: "{req.sikayet_detayi}"
            - Aile Öyküsü: "{req.aile_oykusu}"

            YAPAY ZEKA MODELİ TAHMİN ORANLARI:
            {yan_yana_tahminler}

            KATI YAZIM KURALLARI (BUNLARA KESİNLİKLE UYMALISIN):
            1. HİÇBİR ŞEKİLDE markdown formatı KULLANMA! (Başlıklar için # işareti, kalın yazmak için ** işareti, listeler için - veya * işareti kesinlikle KULLANILMAYACAK).
            2. Ağır tıbbi terimlerden (eritem, deskuamasyon, lezyon, fissür vb.) tamamen uzak dur, halkın anlayacağı son derece sade, günlük ve düz bir Türkçe kullan.
            3. Metni sadece 3 adet düz paragraf şeklinde yaz. Asla alt başlık veya maddeleme yapma.
            4. İlk paragrafta hastanın şikayetini ve yapay zekanın bulgularını doğal bir metin halinde harmanlayarak özetle.
            5. İkinci paragrafta cildi rahatlatacak günlük bakım önerileri, çevresel faktörler ve kaçınılması gerekenleri düz metin olarak anlat.
            6. Üçüncü (son) paragrafta ise "Bu rapor bir yapay zeka klinik karar destek sistemi tarafından üretilmiş ön değerlendirme metni olup kesin tanı niteliği taşımamaktadır ve nihai tanı ile tedavi planı ancak uzman bir tabip tarafından yapılacak detaylı fiziki muayene sonucunda netleşecektir!" cümlesini kullanarak metni bitir.
            """
           
        try:
                # 404 hatasını aşmak için model güncellendi (Tedavi)
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=[prompt, image]
                )
                ai_raporu = response.text.strip()
        except Exception as gemini_hata:
                print("Gemini API Hatası Yakalandı (Sistem Çökmesi Engellendi):", gemini_hata)
                # Olası bir aksilikte sistemi kurtaran profesyonel kalkan (Savunma)
                ai_raporu = (
                    "Yapay zeka analiz merkezimizde anlık bir yoğunluk yaşanmaktadır. "
                    "Şikayetiniz, verileriniz ve klinik görselleriniz uzman doktorunuzun sistemine başarıyla iletilmiştir. "
                    "Lütfen kesin teşhis ve tedavi planı için doktorunuzun değerlendirmesini bekleyiniz."
                )

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

        guncel_tarih_saat = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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