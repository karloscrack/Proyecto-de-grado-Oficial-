import os
import boto3
import time

# --- CREDENCIALES ---
# Intenta leerlas del sistema (Railway), si no existen, usa las que pongas aquí
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", "PON_AQUI_TU_ACCESS_KEY_SI_ES_LOCAL")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY", "PON_AQUI_TU_SECRET_KEY_SI_ES_LOCAL")
AWS_REGION = "us-east-1"
COLLECTION_ID = "estudiantes_db"

# Cliente Rekognition
rekog = boto3.client('rekognition', 
                     region_name=AWS_REGION, 
                     aws_access_key_id=AWS_ACCESS_KEY, 
                     aws_secret_access_key=AWS_SECRET_KEY)

def migrar_todo():
    carpeta_base = "perfiles_db"
    
    # Verificación de seguridad
    if AWS_ACCESS_KEY.startswith("PON_AQUI"):
        print("❌ ERROR: No has configurado tus credenciales AWS en el script.")
        return

    if not os.path.exists(carpeta_base):
        print(f"❌ No encuentro la carpeta '{carpeta_base}'.")
        return

    print("🚀 INICIANDO MIGRACIÓN A LA NUBE (Modo Alta Precisión)...")
    
    # Recorremos cada carpeta de usuario
    usuarios = [f for f in os.listdir(carpeta_base) if os.path.isdir(os.path.join(carpeta_base, f))]
    
    if not usuarios:
        print("⚠️ No hay usuarios en perfiles_db")
        return

    for cedula in usuarios:
        ruta_usuario = os.path.join(carpeta_base, cedula)
        # Buscamos imágenes dentro
        fotos = [f for f in os.listdir(ruta_usuario) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not fotos:
            print(f"⏩ Usuario {cedula} no tiene fotos válidas. Saltando...")
            continue
            
        print(f"\n👤 Procesando Usuario: {cedula}")
        
        for foto in fotos:
            ruta_Img = os.path.join(ruta_usuario, foto)
            try:
                with open(ruta_Img, 'rb') as image:
                    bytes_Img = image.read()
                
                print(f"   ☁️ Subiendo '{foto}' a AWS...", end="")
                
                # --- CORRECCIÓN 1: Guardamos la respuesta en 'response' ---
                response = rekog.index_faces(
                    CollectionId=COLLECTION_ID,
                    Image={'Bytes': bytes_Img},
                    ExternalImageId=cedula,
                    DetectionAttributes=['ALL'],
                    # --- CORRECCIÓN 2: Calidad ALTA para evitar confusión de identidad ---
                    QualityFilter='HIGH' 
                )

                # --- CORRECCIÓN 3: Indentación correcta del IF ---
                if response['FaceRecords']:
                    print(" ✅ OK (Cara indexada)")
                else:
                    print(" ⚠️ OJO: AWS rechazó la foto (mala calidad o sin rostro).")

                time.sleep(0.2) # Pausa para no saturar
                
            except Exception as e:
                print(f" ❌ Error: {e}")

    print("\n✨ MIGRACIÓN COMPLETADA. Amazon ahora conoce a tus usuarios.")

if __name__ == "__main__":
    migrar_todo()