from google import genai

# Kendi Gemini API anahtarını buraya yazıyorsun
client = genai.Client(api_key="BURAYA_API_KEY_YAZILACAK")

try:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Merhaba Gemini, sistem çalışıyor mu? Kısa bir selamlama ver.",
    )
    print("BAŞARILI! Gemini'dan gelen yanıt:")
    print(response.text)
except Exception as e:
    print("HATA ALDIK:", str(e))