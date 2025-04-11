import discord  # Discord kütüphanesi
from discord.ext import commands  # Komut sistemi
import os  # Dosya işlemleri
import numpy as np  # NumPy
from PIL import Image, ImageOps  # Görüntü işlemleri
from tensorflow.keras.models import load_model  # Model yükleme 


# Bot izinlerini ayarla
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

# Botu oluştur
bot = commands.Bot(command_prefix="!", intents=intents)

# Görsellerin kaydedileceği klasör
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def get_class(image_path, model_path, labels_path):
    # Model ve etiket dosyalarının varlığını kontrol et
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return "Hata: Model veya etiket dosyası bulunamadı.", 0.0
    
    # Modeli yükle
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()
    
    # Görüntü ön işleme
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Tahmin yap
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@bot.command()
async def check(ctx):
    if not ctx.message.attachments:
        await ctx.send("⚠️ Görsel yüklemeyi unuttun!")
        return
    
    for attachment in ctx.message.attachments:
        file_name = attachment.filename
        file_path = os.path.join(IMAGE_DIR, file_name)
        
        try:
            await attachment.save(file_path)
            await ctx.send(f"✅ Görsel başarıyla kaydedildi: `{file_name}`")
            
            # Modeli çalıştır
            model_path = "keras_model.h5"
            labels_path = "labels.txt"
            class_name, confidence = get_class(file_path, model_path, labels_path)
            
            """
            # Hata durumunda mesaj
            if "Hata" in class_name:
                await ctx.send(f"⚠️ {class_name}")
                return
            """
            # Sınıf adı ve güven skorunu logla
            print(f"Tahmin edilen sınıf: {class_name[2:]} - Güven: %{confidence*100:.2f}")

            messages = {
        "0 ekran_karti": "Ekran kartı (GPU), bilgisayarın en önemli donanım bileşenlerinden biridir ve bilgisayarın ekranına görüntü sağlamakla görevlidir. Bilgisayar kullanıcılarının oyun oynama, video izleme, fotoğraf düzenleme ve grafik tasarım gibi işlemleri yaparken yüksek kaliteli ve akıcı bir grafik deneyimi yaşamasını sağlar.",
        "1 fan": "Kasada fanlar, kasada yer alan bileşenlerin ve donanımların sıcaklıklarını düşürmek için kullanılan bir tür soğutma mekanizmasıdır. Kasada bulunan işlemci, ekran kartı, sabit disk, RAM ve diğer bileşenler yük altında çalıştıklarında ısınarak performanslarını düşürebilirler veya hatta hasar görebilirler. Bu nedenle, kasada fanlarının varlığı, sıcak havanın dışarı atılmasını sağlayarak kasadaki sıcaklığı düşürerek bileşenlerin daha stabil ve güvenilir bir şekilde çalışmalarını sağlar.",
        "2 monitor": "Monitör, başta televizyon ve bilgisayar olmak üzere birçok elektronik cihazın en önemli çıktı aygıtıdır. Monitör, plastik bir muhafaza içerisinde gerekli elektronik devreleri, güç transformatörünü ve resmi oluşturan birimleri içerir. Monitörle bilgisayar arasındaki iletişimi ekran kartı sağlar.",
        "3 RAM_karti": "Rastgele erişimli hafıza veya rastgele erişimli bellek (İngilizce: Random Access Memory, RAM) mikroişlemcili sistemlerde kullanılan, genellikle çalışma verileriyle birlikte makine kodunu depolamak için kullanılan herhangi bir sırada okunabilen ve değiştirilebilen bir tür geçici veri deposudur."
         }

            special_message = messages.get(class_name, "Bu sınıf için özel bir mesaj yok.")
             
            await ctx.send(f"🔍 Tahmin: `{class_name[2:]}` (%{confidence*100:.2f} güven)  {special_message}")
        except Exception as e:
            await ctx.send(f"⚠️ Hata oluştu: {str(e)}")
# Token ile botu çalıştır (Kendi token'ını buraya eklemelisin)
bot.run("MTMwMjY4ODI3NjMyMjMyMDQ4NQ.Gh2jCB.X5QqziAvh-d5C6iKtNF_Q9wQKzOhsnNGy2YNDs")