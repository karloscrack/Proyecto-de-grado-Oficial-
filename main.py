import shutil
import os
import sqlite3
import logging
import datetime
import zipfile
import hashlib
import boto3
import cv2 
import numpy as np
import tempfile 
import smtplib
import pytz
import json
import io
import difflib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks, Request, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from botocore.config import Config
from pydantic import BaseModel

# --- 0. CONFIGURACIÓN DE ZONA HORARIA ECUADOR ---
ECUADOR_TZ = pytz.timezone('America/Guayaquil')  # UTC-5

def ahora_ecuador():
    """Devuelve la fecha/hora actual en zona horaria de Ecuador"""
    return datetime.datetime.now(ECUADOR_TZ)

# --- CONFIGURACIÓN DE CORREO ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # Usamos 465 para SSL directo
SMTP_EMAIL = "karlos.ayala.lopez.1234@gmail.com"
SMTP_PASSWORD = "mzjg jvxj mruk qgeb"

# --- 1. CONFIGURACIÓN Y CREDENCIALES AWS/B2 ---
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"
COLLECTION_ID = "estudiantes_db"

# Inicialización condicional de AWS Rekognition
try:
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        rekog = boto3.client('rekognition', region_name=AWS_REGION, 
                           aws_access_key_id=AWS_ACCESS_KEY, 
                           aws_secret_access_key=AWS_SECRET_KEY)
        print("✅ AWS Rekognition inicializado")
    else:
        rekog = None
        print("⚠️ AWS Rekognition no disponible (credenciales faltantes)")
except Exception as e:
    rekog = None
    print(f"⚠️ Error inicializando AWS Rekognition: {e}")

# Configuración Backblaze B2
ENDPOINT_B2 = "https://s3.us-east-005.backblazeb2.com"
KEY_ID_B2 = "00508884373dab40000000001"
APP_KEY_B2 = "K005jvkLLmLdUKhhVis1qLcnU4flx0g"
BUCKET_NAME = "Proyecto-Grado-Karlos-2025"

try:
    my_config = Config(signature_version='s3v4', region_name='us-east-005')
    s3_client = boto3.client('s3', 
                            endpoint_url=ENDPOINT_B2,
                            aws_access_key_id=KEY_ID_B2,
                            aws_secret_access_key=APP_KEY_B2,
                            config=my_config)
    print("✅ Cliente S3 (Backblaze) inicializado")
except Exception as e:
    s3_client = None
    print(f"⚠️ Cliente S3 no disponible: {e}")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- LÓGICA DE VOLUMEN PERSISTENTE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = "Bases_de_datos.db"
VOLUMEN_PATH = "/app/datos_persistentes"

# Determinar ruta final de la base de datos
if os.path.exists(VOLUMEN_PATH):
    db_en_volumen = os.path.join(VOLUMEN_PATH, DB_FILENAME)
    if not os.path.exists(db_en_volumen):
        db_original = os.path.join(BASE_DIR, DB_FILENAME)
        if os.path.exists(db_original):
            shutil.copy(db_original, db_en_volumen)
            print(f"✅ Base de datos copiada al volumen persistente: {db_en_volumen}")
    DB_NAME = db_en_volumen
else:
    DB_NAME = os.path.join(BASE_DIR, DB_FILENAME)

print(f"📁 Ruta base de datos: {DB_NAME}")

class EstadoUsuarioRequest(BaseModel):
    cedula: str
    activo: int

class BackupRequest(BaseModel):
    tipo: str = "completo"

# --- INICIO DEL CÓDIGO A PEGAR ---
def optimizar_sistema_db():
    """Ejecuta mantenimiento VACUUM en la base de datos"""
    try:
        # CORRECCIÓN: Usamos DB_NAME (que es la variable global segura)
        conn = sqlite3.connect(DB_NAME)
        conn.execute("VACUUM")
        conn.close()
        print("✅ Sistema optimizado (VACUUM ejecutado)")
        return True
    except Exception as e:
        print(f"⚠️ Alerta menor: No se pudo optimizar DB: {e}")
        return False

# --- 2. INICIALIZACIÓN DE BASE DE DATOS - MEJORADA ---
def init_db_completa():
    """Inicialización robusta de la base de datos con compatibilidad hacia atrás"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Tabla Usuarios
        c.execute('''CREATE TABLE IF NOT EXISTS Usuarios (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Nombre TEXT NOT NULL,
            Apellido TEXT NOT NULL,
            CI TEXT UNIQUE NOT NULL,
            Password TEXT NOT NULL,
            Tipo INTEGER DEFAULT 1,
            Foto TEXT,
            Activo INTEGER DEFAULT 1,
            Fecha_Desactivacion TIMESTAMP NULL,
            Ultimo_Acceso TIMESTAMP NULL,
            TutorialVisto INTEGER DEFAULT 0,
            Face_Encoding TEXT,
            Fecha_Registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Email TEXT,
            Telefono TEXT
        )''')
        
        # Tabla Evidencias
        c.execute('''CREATE TABLE IF NOT EXISTS Evidencias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            CI_Estudiante TEXT NOT NULL,
            Url_Archivo TEXT NOT NULL,
            Hash TEXT NOT NULL,
            Estado INTEGER DEFAULT 1,
            Tipo_Archivo TEXT DEFAULT 'documento',
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Tamanio_KB REAL DEFAULT 0,
            Asignado_Automaticamente INTEGER DEFAULT 0,
            FOREIGN KEY(CI_Estudiante) REFERENCES Usuarios(CI) ON DELETE CASCADE
        )''')

        # Tabla Solicitudes
        c.execute('''CREATE TABLE IF NOT EXISTS Solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Tipo TEXT NOT NULL,
            CI_Solicitante TEXT NOT NULL,
            Email TEXT,
            Detalle TEXT,
            Evidencia_Reportada_Url TEXT,
            Id_Evidencia INTEGER,
            Resuelto_Por TEXT,
            Respuesta TEXT,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Estado TEXT DEFAULT 'PENDIENTE',
            Fecha_Resolucion TIMESTAMP NULL
        )''')
        
        # Tabla Auditoria
        c.execute('''CREATE TABLE IF NOT EXISTS Auditoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Accion TEXT NOT NULL,
            Detalle TEXT,
            IP TEXT,
            Usuario TEXT,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Tabla para métricas y estadísticas
        c.execute('''CREATE TABLE IF NOT EXISTS Metricas_Sistema (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Fecha DATE UNIQUE,
            Total_Usuarios INTEGER DEFAULT 0,
            Total_Evidencias INTEGER DEFAULT 0,
            Solicitudes_Pendientes INTEGER DEFAULT 0,
            Almacenamiento_MB REAL DEFAULT 0
        )''')
        
        # Verificar y agregar columnas faltantes (para compatibilidad)
        columnas_compatibilidad = [
            ("Usuarios", "Email", "TEXT"),
            ("Usuarios", "Telefono", "TEXT"),
            ("Usuarios", "Ultimo_Acceso", "TIMESTAMP NULL"),
            ("Usuarios", "Fecha_Desactivacion", "TIMESTAMP NULL"),
            
            # 👇 CAMBIO CLAVE AQUÍ: Quitamos 'DEFAULT CURRENT_TIMESTAMP' y lo dejamos como 'TEXT' 👇
            ("Usuarios", "Fecha_Registro", "TEXT"), 
            # 👆 Esto permite que la columna se cree sin errores en bases de datos viejas
            
            ("Evidencias", "Tipo_Archivo", "TEXT DEFAULT 'documento'"),
            ("Evidencias", "Tamanio_KB", "REAL DEFAULT 0"),
            ("Evidencias", "Asignado_Automaticamente", "INTEGER DEFAULT 0"),
            ("Evidencias", "Hash", "TEXT DEFAULT 'PENDIENTE'"), 
            
            ("Solicitudes", "Fecha_Resolucion", "TIMESTAMP NULL"),
            ("Auditoria", "Usuario", "TEXT"),
            ("Auditoria", "IP", "TEXT")
        ]
        
        for tabla, columna, tipo in columnas_compatibilidad:
            try:
                c.execute(f"SELECT {columna} FROM {tabla} LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    c.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {tipo}")
                    print(f"✅ Columna {columna} agregada a tabla {tabla}")
                except Exception as e:
                    print(f"⚠️ No se pudo agregar columna {columna} a {tabla}: {e}")
        
        # Crear usuario admin si no existe
        c.execute("SELECT CI FROM Usuarios WHERE Tipo=0")
        if not c.fetchone():
            c.execute('''INSERT INTO Usuarios (Nombre, Apellido, CI, Password, Tipo, Activo) 
                         VALUES (?,?,?,?,?,?)''', 
                     ('Admin', 'Sistema', '9999999999', 'admin123', 0, 1))
            print("✅ Usuario admin creado")
        
        # Crear índices para mejor rendimiento
        c.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_ci ON Usuarios(CI)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_evidencias_ci ON Evidencias(CI_Estudiante)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_evidencias_fecha ON Evidencias(Fecha)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_solicitudes_estado ON Solicitudes(Estado)")
        c.execute("SELECT CI FROM Usuarios WHERE CI='9999999990'")
        if not c.fetchone():
            # Usamos CI 9999999990, Tipo 1 (Estudiante) para que aparezca en el panel
            c.execute("""INSERT INTO Usuarios (Nombre, Apellido, CI, Password, Tipo, Activo, Foto) 
                         VALUES ('Bandeja', 'Recuperados', '9999999990', '123456', 1, 1, '')""")
            print("✅ Usuario contenedor 'Bandeja Recuperados' creado")

        # 2. MOVER TODOS LOS HUÉRFANOS A ESTA BANDEJA
        # Todo lo que diga 'PENDIENTE' ahora será de este usuario
        c.execute("UPDATE Evidencias SET CI_Estudiante='9999999990' WHERE CI_Estudiante='PENDIENTE' OR CI_Estudiante IS NULL")
        if c.rowcount > 0:
            print(f"📦 {c.rowcount} evidencias recuperadas movidas a la Bandeja de Recuperados.")

        
        conn.commit()
        conn.close()
        print("✅ Base de datos verificada y actualizada correctamente")
        
        # Ejecutar optimización inicial
        optimizar_sistema_db()
        
    except Exception as e:
        print(f"❌ Error inicializando DB: {e}")
        raise

# Ejecutar inicialización al arrancar
init_db_completa()

# =========================================================================
# 3. FUNCIONES AUXILIARES
# =========================================================================
def get_db_connection():
    """Conexión a DB con compatibilidad de nombres de columna"""
    conn = sqlite3.connect(DB_NAME)
    
    # Hacer que las filas se comporten como diccionarios
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            column_name = col[0].replace('"', '')
            d[column_name] = row[idx]
        return d
    
    conn.row_factory = dict_factory
    return conn

def registrar_auditoria(accion: str, detalle: str, usuario: str = "Sistema", ip: str = ""):
    """Registra una acción en la tabla de auditoría con fecha de Ecuador"""
    try:
        fecha_ecuador = ahora_ecuador()
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO Auditoria (Accion, Detalle, Usuario, IP, Fecha) 
            VALUES (?, ?, ?, ?, ?)
        """, (accion, detalle, usuario, ip, fecha_ecuador))
        conn.commit()
        conn.close()
        logging.info(f"AUDITORIA: {accion} - {detalle}")
    except Exception as e:
        logging.error(f"Error en auditoria: {e}")

def enviar_correo_real(destinatario: str, asunto: str, mensaje: str, html: bool = False) -> bool:
    import requests # Usaremos la librería requests que es más sencilla para APIs
    
    # Tu clave gratuita de Resend
    API_KEY = "re_UgHvnVwc_GoohB6so8khU8mCBmLJB1bzJ"
    
    try:
        url = "https://api.resend.com/emails"
        payload = {
            "from": "onboarding@resend.dev",
            "to": [destinatario],
            "subject": asunto,
            "html": mensaje if html else f"<p>{mensaje}</p>"
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Enviamos el correo como una petición web normal (Puerto 443)
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"✅ Correo enviado vía API Resend a {destinatario}")
            return True
        else:
            print(f"❌ Error API Resend: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión con API Resend: {e}")
        return False
    
def calcular_hash(ruta: str) -> str:
    """Calcula hash SHA256 de un archivo"""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def obtener_tamanio_archivo_kb(ruta: str) -> float:
    """Obtiene el tamaño de un archivo en KB"""
    try:
        return os.path.getsize(ruta) / 1024
    except:
        return 0

def optimizar_sistema_db():
    """Ejecuta comandos de optimización en la base de datos"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Ejecutar VACUUM para optimizar espacio
        c.execute("VACUUM")
        
        # Ejecutar ANALYZE para optimizar consultas
        c.execute("ANALYZE")
        
        # Reconstruir índices
        c.execute("REINDEX")
        
        conn.commit()
        conn.close()
        print("✅ Sistema optimizado (VACUUM, ANALYZE, REINDEX)")
        return True
    except Exception as e:
        print(f"⚠️ Error optimizando sistema: {e}")
        return False

# --- REEMPLAZA TU FUNCIÓN 'identificar_rostro_aws' POR ESTA ---
def preparar_imagen_aws(ruta_imagen):
    """
    1. Lee la imagen (soporta AVIF si las librerías están instaladas).
    2. La convierte SIEMPRE a JPG (compatible con AWS).
    3. Comprime si pesa más de 5MB.
    """
    MAX_BYTES = 5242880 # 5 MB
    
    try:
        # Intentamos leer con OpenCV
        img = cv2.imread(ruta_imagen)
        
        # Si OpenCV falla (común con AVIF antiguos), intentamos con PIL (si está instalado)
        if img is None:
            try:
                from PIL import Image
                import numpy as np
                pil_img = Image.open(ruta_imagen).convert('RGB')
                img = np.array(pil_img) 
                # Convertir RGB (PIL) a BGR (OpenCV)
                img = img[:, :, ::-1].copy() 
            except ImportError:
                pass # Si no hay PIL, nos rendimos

        # Si después de todo no pudimos leer la imagen...
        if img is None:
            # Si es AVIF y no pudimos leerla, NO podemos mandarla cruda a AWS. Retornamos None o error.
            if ruta_imagen.lower().endswith('.avif'):
                print("⚠️ Error: No se pudo decodificar AVIF. Instala 'pillow-avif-plugin'.")
                return None
            # Si es JPG/PNG, mandamos crudo
            with open(ruta_imagen, 'rb') as f: return f.read()

        # COMPRESIÓN / CONVERSIÓN A JPG
        # Esto transforma el AVIF/WEBP/PNG a un JPG estándar que AWS sí entiende
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"⚠️ Error preparando imagen AWS: {e}")
        with open(ruta_imagen, 'rb') as f: return f.read()

def identificar_varios_rostros_aws(imagen_path: str, confidence_threshold: float = 80.0) -> List[str]:
    if not rekog: return []
    cedulas_encontradas = set()
    
    try:
        # 1. Cargar imagen con OpenCV (para recortes)
        img = cv2.imread(imagen_path)
        if img is None: return []
        height, width, _ = img.shape
        
        # 2. Obtener BYTES SEGUROS (Comprimidos si es necesario)
        image_bytes = preparar_imagen_aws(imagen_path)
            
        # 3. Detectar caras en la imagen (usando los bytes seguros)
        response_detect = rekog.detect_faces(Image={'Bytes': image_bytes})
        
        if not response_detect['FaceDetails']: return []

        # 4. Procesar cada cara
        for faceDetail in response_detect['FaceDetails']:
            bbox = faceDetail['BoundingBox']
            x = int(bbox['Left'] * width)
            y = int(bbox['Top'] * height)
            w = int(bbox['Width'] * width)
            h = int(bbox['Height'] * height)
            
            x, y = max(0, x), max(0, y)
            w, h = min(width - x, w), min(height - y, h)
            face_crop = img[y:y+h, x:x+w]
            if face_crop.size == 0: continue

            _, buffer = cv2.imencode('.jpg', face_crop)
            crop_bytes = buffer.tobytes()
            
            try:
                # AQUÍ ESTÁ EL CAMBIO CLAVE: MaxFaces=5 (Antes era 1)
                # Esto permite que si dos usuarios tienen la misma cara, AWS devuelva a ambos.
                search_res = rekog.search_faces_by_image(
                    CollectionId=COLLECTION_ID,
                    Image={'Bytes': crop_bytes},
                    MaxFaces=5, 
                    FaceMatchThreshold=confidence_threshold
                )
                
                # Ahora iteramos sobre TODAS las coincidencias encontradas para esa cara
                for match in search_res['FaceMatches']:
                    ced = match['Face'].get('ExternalImageId')
                    if ced: cedulas_encontradas.add(ced)
                    
            except: continue 

        return list(cedulas_encontradas)
    except Exception as e:
        print(f"Error IA Rostros: {e}")
        return []
    

# --- REEMPLAZA TU FUNCIÓN 'buscar_estudiantes_por_texto' POR ESTA VERSIÓN CON DEBUG VISUAL ---

def buscar_estudiantes_por_texto(imagen_path: str, conn):
    if not rekog: return [], []
    cedulas_encontradas = set()
    texto_leido_debug = []
    
    try:
        # 1. Obtener BYTES SEGUROS (Comprimidos < 5MB)
        image_bytes = preparar_imagen_aws(imagen_path)
            
        # 2. Detectar texto
        response = rekog.detect_text(Image={'Bytes': image_bytes})
        
        palabras_sueltas = []
        lineas_completas = []
        
        for t in response.get('TextDetections', []):
            txt = t['DetectedText'].lower()
            if t['Type'] == 'WORD':
                # Filtramos palabras basura muy cortas (de 1 o 2 letras) que causan ruido
                if len(txt) > 2: 
                    palabras_sueltas.append(txt)
                    texto_leido_debug.append(txt)
            elif t['Type'] == 'LINE':
                lineas_completas.append(txt)
        
        if not palabras_sueltas: return [], ["(Imagen ilegible)"]

        # 3. Lógica de Búsqueda
        estudiantes = conn.execute("SELECT Nombre, Apellido, CI FROM Usuarios WHERE Tipo=1").fetchall()
        
        for est in estudiantes:
            # Partes del nombre: "Taylor", "Swift"
            partes = est['Nombre'].lower().split() + est['Apellido'].lower().split()
            # Filtramos conectores como "de", "la"
            piezas_validas = {p for p in partes if len(p) > 2}
            
            # --- ESTRATEGIA A: LÍNEAS (Contexto exacto) ---
            # Si la línea dice literalmente "award to karlos ayala", es un match seguro
            match_linea = False
            nombre_corto = f"{est['Nombre'].split()[0]} {est['Apellido'].split()[0]}".lower()
            
            for linea in lineas_completas:
                # Buscamos la frase dentro de la línea con tolerancia media (65%)
                s = difflib.SequenceMatcher(None, nombre_corto, linea)
                match = s.find_longest_match(0, len(nombre_corto), 0, len(linea))
                if match.size > 4: # Si coinciden más de 4 letras seguidas en orden
                     if difflib.get_close_matches(nombre_corto, [linea], n=1, cutoff=0.65): 
                        cedulas_encontradas.add(est['CI'])
                        match_linea = True
                        break
            if match_linea: continue

            # --- ESTRATEGIA B: BOLSA DE PALABRAS (Filtro Dinámico) ---
            coincidencias = 0
            for pieza in piezas_validas:
                # AQUÍ ESTÁ EL TRUCO:
                # Palabras cortas (< 5 letras): Filtro DURO (0.85). Evita que 'Smith' se confunda con 'Swift'.
                # Palabras largas (>= 5 letras): Filtro SUAVE (0.60). Permite leer 'Karlos' en gótico ('Rarlos').
                umbral = 0.85 if len(pieza) < 5 else 0.60
                
                if difflib.get_close_matches(pieza, palabras_sueltas, n=1, cutoff=umbral):
                    coincidencias += 1
            
            # Reglas de Aprobación
            if coincidencias >= 2:
                cedulas_encontradas.add(est['CI'])
            elif len(piezas_validas) == 2 and coincidencias == 2:
                cedulas_encontradas.add(est['CI'])

    except Exception as e:
        print(f"⚠️ Error OCR: {e}")
        return [], [f"Error técnico: {str(e)}"]
        
    return list(cedulas_encontradas), texto_leido_debug

def coincidencia_difusa(partes_buscadas, palabras_en_imagen, umbral):
    """
    Verifica si TODAS las 'partes_buscadas' están presentes en 'palabras_en_imagen'
    con cierta tolerancia a errores (umbral).
    """
    aciertos = 0
    # Usamos una copia para no afectar búsquedas de otros estudiantes
    pool = palabras_en_imagen.copy()
    
    for parte in partes_buscadas:
        # Busca la palabra más parecida en la 'bolsa' de palabras de la imagen
        matches = difflib.get_close_matches(parte, pool, n=1, cutoff=umbral)
        if matches:
            aciertos += 1
            # Opcional: pool.remove(matches[0]) si quisieras evitar repetir palabras
            
    # Éxito si encontramos TODAS las partes (Ej: Encontró "Juan" Y encontró "Perez")
    return aciertos == len(partes_buscadas)

# Función auxiliar por si no la tienes
def calcular_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
    
def calcular_estadisticas_reales() -> dict:
    """Calcula estadísticas REALES filtrando SOLO ESTUDIANTES (Ignora Admins)"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Contar usuarios activos (Estudiantes Tipo 1 activos)
        c.execute("SELECT COUNT(*) as total FROM Usuarios WHERE Tipo = 1 AND Activo = 1")
        fila_usuarios = c.fetchone()
        usuarios_activos = fila_usuarios['total'] if fila_usuarios else 0
        
        # 2. Contar evidencias (SOLO DE ESTUDIANTES - JOIN CON USUARIOS)
        # Esto excluye automáticamente las fotos subidas al perfil del Admin
        c.execute("""
            SELECT COUNT(e.id) as total 
            FROM Evidencias e
            JOIN Usuarios u ON e.CI_Estudiante = u.CI
            WHERE u.Tipo = 1
        """)
        fila_evidencias = c.fetchone()
        total_evidencias = fila_evidencias['total'] if fila_evidencias else 0
        
        # 3. Sumar peso (SOLO DE ESTUDIANTES)
        c.execute("""
            SELECT SUM(e.Tamanio_KB) as peso_total 
            FROM Evidencias e
            JOIN Usuarios u ON e.CI_Estudiante = u.CI
            WHERE u.Tipo = 1
        """)
        fila_peso = c.fetchone()
        resultado_kb = fila_peso['peso_total'] if fila_peso and fila_peso['peso_total'] else 0
        
        # Lógica de estimación
        total_kb = resultado_kb
        nota_almacenamiento = "Calculado exacto (Solo Estudiantes)"
        
        # Corrección para cuando hay archivos pero pesan 0
        if total_kb == 0 and total_evidencias > 0:
            total_kb = total_evidencias * 2500 

        tamanio_total_mb = total_kb / 1024
        
        # Costos (Basados solo en consumo de estudiantes)
        costo_rekognition = (total_evidencias / 1000) * 1.0
        costo_almacenamiento = (tamanio_total_mb / 1024) * 0.023
        
        # 4. Solicitudes pendientes
        c.execute("SELECT COUNT(*) as total FROM Solicitudes WHERE Estado = 'PENDIENTE'")
        fila_solicitudes = c.fetchone()
        solicitudes_pendientes = fila_solicitudes['total'] if fila_solicitudes else 0
        
        conn.close()
        
        return {
            "usuarios_activos": usuarios_activos,
            "total_evidencias": total_evidencias,
            "almacenamiento_mb": round(tamanio_total_mb, 2),
            "almacenamiento_gb": round(tamanio_total_mb / 1024, 4),
            "costo_estimado_usd": {
                "rekognition": round(costo_rekognition, 2),
                "almacenamiento": round(costo_almacenamiento, 4),
                "total": round(costo_rekognition + costo_almacenamiento, 2)
            },
            "solicitudes_pendientes": solicitudes_pendientes,
            "nota": nota_almacenamiento
        }
    except Exception as e:
        print(f"Error estadisticas: {e}")
        return {
            "usuarios_activos": 0, "total_evidencias": 0, 
            "almacenamiento_mb": 0, "almacenamiento_gb": 0,
            "solicitudes_pendientes": 0
        }

# =========================================================================
# 4. CONFIGURACIÓN FASTAPI
# =========================================================================
app = FastAPI(title="Sistema Educativo Despertar", version="7.0")

@app.on_event("startup")
async def al_iniciar_sistema():
    """Este código se ejecuta AUTOMÁTICAMENTE cuando Railway enciende el servidor"""
    print("⚡ EVENTO DE INICIO DETECTADO: Ejecutando protocolos de mantenimiento...")
    
    # 1. Ejecutar limpieza de duplicados (Calcula hashes faltantes y borra copias)
    # Ejecutamos en un hilo aparte para no bloquear el arranque si son muchos archivos
    import threading
    hilo_limpieza = threading.Thread(target=limpieza_duplicados_startup)
    hilo_limpieza.start()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://proyecto-grado-karlos.vercel.app",  # Tu frontend
        "http://localhost:5500",
        "*" # Opcional: permite todos para pruebas
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 5. ENDPOINTS PRINCIPALES
# =========================================================================

@app.get("/")
def home():
    """Endpoint raíz del sistema"""
    return {
        "status": "online", 
        "backend": "Sistema Educativo Despertar V7.0",
        "cors_enabled": True,
        "zona_horaria": "America/Guayaquil (UTC-5)",
        "timestamp": ahora_ecuador().isoformat()
    }

@app.get("/health")
async def health_check():
    """Verifica salud del sistema"""
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        
        # Verificar tablas principales
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as count FROM Usuarios")
        usuarios = c.fetchone()['count']
        
        c.execute("SELECT COUNT(*) as count FROM Evidencias")
        evidencias = c.fetchone()['count']
        
        conn.close()
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": ahora_ecuador().isoformat(),
            "database": "connected",
            "estadisticas": {
                "usuarios": usuarios,
                "evidencias": evidencias
            },
            "aws_rekognition": "available" if rekog else "unavailable",
            "s3_storage": "available" if s3_client else "unavailable"
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "unhealthy",
            "error": str(e)
        })

# =========================================================================
# 6. ENDPOINTS DE AUTENTICACIÓN
# =========================================================================

@app.post("/iniciar_sesion")
async def iniciar_sesion(request: Request, cedula: str = Form(...), contrasena: str = Form(...)):
    """Versión corregida que envía ID para el perfil"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM Usuarios WHERE TRIM(CI) = ?", (cedula.strip(),))
        user = c.fetchone()
        
        if not user or user["Password"] != contrasena.strip():
            conn.close()
            return JSONResponse({"autenticado": False, "mensaje": "Credenciales inválidas"})

        # Datos seguros para el frontend
        datos_usuario = {
            "id": user["ID"],  # <--- CRÍTICO: ESTO ES LO QUE NECESITAS
            "nombre": user["Nombre"],
            "apellido": user["Apellido"],
            "cedula": user["CI"],
            "tipo": user["Tipo"],
            "url_foto": user.get("Foto", ""),
            "email": user.get("Email", ""),
            "tutorial_visto": bool(user.get("TutorialVisto", 0))
        }
        
        conn.close()
        return JSONResponse({"autenticado": True, "mensaje": "Bienvenido", "datos": datos_usuario})
        
    except Exception as e:
        print(f"Error login: {e}")
        return JSONResponse({"autenticado": False, "mensaje": str(e)})
# =========================================================================
# 7. ENDPOINTS DE GESTIÓN DE USUARIOS
# =========================================================================

@app.post("/registrar_usuario")
async def registrar_usuario(
    nombre: str = Form(...),
    apellido: str = Form(...),
    cedula: str = Form(...),
    contrasena: str = Form(...),
    tipo_usuario: int = Form(...),
    email: Optional[str] = Form(None),
    telefono: Optional[str] = Form(None),
    foto: UploadFile = File(...)
):
    """Registra un nuevo usuario con zona horaria Ecuador"""
    try:
        cedula = cedula.strip()
        contrasena = contrasena.strip()
        
        # Validaciones básicas
        if not cedula or not contrasena:
            return JSONResponse(content={
                "error": "La cédula y contraseña son requeridas"
            })
        
        conn = get_db_connection()
        c = conn.cursor()
        
        # Verificar si usuario ya existe
        c.execute("SELECT CI FROM Usuarios WHERE CI=?", (cedula,))
        if c.fetchone():
            conn.close()
            return JSONResponse(content={
                "error": "Usuario ya existe en el sistema"
            })
        
        # Manejar archivo de foto
        temp_dir = tempfile.mkdtemp()
        foto_path = os.path.join(temp_dir, foto.filename)
        
        with open(foto_path, "wb") as f:
            shutil.copyfileobj(foto.file, f)
        
        # Subir a almacenamiento
        nombre_nube = f"perfiles/{cedula}_{int(ahora_ecuador().timestamp())}_{foto.filename}"
        url_foto = ""
        
        if s3_client:
            try:
                s3_client.upload_file(
                    foto_path, 
                    BUCKET_NAME, 
                    nombre_nube,
                    ExtraArgs={'ACL': 'public-read'}
                )
                url_foto = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
                print(f"✅ Foto subida a S3: {url_foto}")
            except Exception as e:
                print(f"⚠️ Error subiendo a S3: {e}")
                url_foto = f"/local/perfiles/{foto.filename}"
        else:
            url_foto = f"/local/perfiles/{foto.filename}"
        
        # Insertar usuario con fecha de Ecuador
        fecha_registro = ahora_ecuador()
        
        # 👇 CAMBIO CLAVE: Convertimos la fecha a texto simple para evitar errores
        fecha_str = fecha_registro.strftime("%Y-%m-%d %H:%M:%S") 
        
        c.execute("""
            INSERT INTO Usuarios 
            (Nombre, Apellido, CI, Password, Tipo, Foto, Activo, Email, Telefono, Fecha_Registro)
            VALUES (?,?,?,?,?,?,1,?,?,?)
        """, (
            nombre.strip(),
            apellido.strip(),
            cedula,
            contrasena,
            tipo_usuario,
            url_foto,
            email,
            telefono,
            fecha_str  # <--- Usamos la versión texto, no el objeto fecha
        ))
        
        # Si es estudiante, agregar a colección de rostros AWS
        if tipo_usuario == 1 and rekog:
            try:
                with open(foto_path, 'rb') as image_file:
                    image_bytes = image_file.read()
                
                rekog.index_faces(
                    CollectionId=COLLECTION_ID,
                    Image={'Bytes': image_bytes},
                    ExternalImageId=cedula,
                    MaxFaces=1,
                    QualityFilter='AUTO'
                )
                print(f"✅ Rostro indexado en AWS para estudiante {cedula}")
            except Exception as e:
                print(f"⚠️ Error indexando rostro en AWS: {e}")
        
        conn.commit()
        conn.close()
        
        # Limpiar archivos temporales
        shutil.rmtree(temp_dir)
        
        # Registrar auditoría
        registrar_auditoria("REGISTRO_USUARIO", f"Usuario {cedula} registrado")
        
        return JSONResponse(content={
            "mensaje": "Usuario registrado exitosamente",
            "url_foto": url_foto,
            "cedula": cedula,
            "fecha_registro": fecha_registro.isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error en registrar_usuario: {e}")
        return JSONResponse(content={"error": str(e)})

@app.post("/buscar_estudiante")
async def buscar_estudiante(cedula: str = Form(...)):
    """Busca datos de un estudiante y sus evidencias para el perfil"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Buscar usuario
        c.execute("SELECT * FROM Usuarios WHERE CI = ?", (cedula,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            # Usamos 'encontrado' porque así lo espera perfil.html
            return JSONResponse({"encontrado": False, "mensaje": "Estudiante no encontrado"})
            
        # 2. Obtener galería de evidencias (CRÍTICO: FALTABA ESTO)
        try:
            c.execute("""
                SELECT id, Url_Archivo as url, Tipo_Archivo as tipo, Fecha, Estado 
                FROM Evidencias 
                WHERE CI_Estudiante = ? AND Estado = 1 
                ORDER BY Fecha DESC
            """, (cedula,))
            evs = [dict(r) for r in c.fetchall()]
        except Exception as e:
            print(f"Error obteniendo galería: {e}")
            evs = []

        conn.close()
        
        # 3. Preparar datos de respuesta
        datos_usuario = {
            "id": user["ID"],
            "nombre": user["Nombre"],
            "apellido": user["Apellido"],
            "cedula": user["CI"],
            "tipo": user["Tipo"],
            "url_foto": user.get("Foto", ""),
            "email": user.get("Email", ""),
            "tutorial_visto": bool(user.get("TutorialVisto", 0)),
            "galeria": evs  # <--- Aquí va la lista de fotos
        }
            
        # 4. Respuesta con la estructura EXACTA que espera tu HTML
        return JSONResponse({
            "encontrado": True,  # <--- CAMBIADO DE 'exito' A 'encontrado'
            "datos": datos_usuario
        })
        
    except Exception as e:
        print(f"Error en buscar_estudiante: {e}")
        return JSONResponse({"encontrado": False, "mensaje": str(e)})
    
@app.post("/cambiar_estado_usuario")
async def cambiar_estado_usuario(datos: EstadoUsuarioRequest):
    """Activa/desactiva un usuario"""
    try:
        conn = get_db_connection()
        
        fecha_desactivacion = ahora_ecuador() if datos.activo == 0 else None
        
        conn.execute("""
            UPDATE Usuarios 
            SET Activo = ?, Fecha_Desactivacion = ?
            WHERE CI = ?
        """, (datos.activo, fecha_desactivacion, datos.cedula))
        
        conn.commit()
        
        # Obtener datos del usuario para auditoría
        c = conn.cursor()
        c.execute("SELECT Nombre, Apellido FROM Usuarios WHERE CI = ?", (datos.cedula,))
        user = c.fetchone()
        
        conn.close()
        
        estado_texto = "desactivada" if datos.activo == 0 else "activada"
        registrar_auditoria(
            "CAMBIO_ESTADO_USUARIO",
            f"Usuario {datos.cedula} ({user['Nombre']} {user['Apellido']}) {estado_texto}"
        )
        
        return JSONResponse(content={
            "mensaje": f"Estado del usuario actualizado a {'activo' if datos.activo == 1 else 'inactivo'}",
            "fecha_cambio": ahora_ecuador().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.delete("/eliminar_usuario/{cedula}")
async def eliminar_usuario(cedula: str):
    """Elimina usuario y SUS ARCHIVOS FÍSICOS (si nadie más los usa)"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Obtener archivos del usuario antes de borrarlos
        evidencias = c.execute("SELECT id, Url_Archivo FROM Evidencias WHERE CI_Estudiante = ?", (cedula,)).fetchall()
        
        archivos_borrados = 0
        espacio_liberado = 0
        
        for ev in evidencias:
            url = ev['Url_Archivo']
            
            # 2. VERIFICACIÓN DE SEGURIDAD: ¿Alguien más usa este mismo archivo?
            # Contamos cuántas veces aparece esta URL en total en la base de datos
            uso_compartido = c.execute("SELECT COUNT(*) as n FROM Evidencias WHERE Url_Archivo = ?", (url,)).fetchone()['n']
            
            # Si 'n' es 1, significa que SOLO este usuario lo tiene. ¡Podemos borrarlo de la nube!
            # Si 'n' > 1, significa que otro estudiante comparte la foto. NO la borramos de S3, solo de la BD.
            if uso_compartido == 1:
                if s3_client and BUCKET_NAME in url:
                    try:
                        parsed = urlparse(url)
                        key = parsed.path.lstrip('/')
                        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                        print(f"   🗑️ Archivo de usuario eliminado físicamente: {key}")
                        archivos_borrados += 1
                        # Estimamos 1MB por archivo si no tenemos el dato a mano
                        espacio_liberado += 1
                    except Exception as e:
                        print(f"⚠️ No se pudo borrar archivo {key}: {e}")
            else:
                print(f"   🛡️ Archivo protegido (compartido por otros): {url}")

        # 3. Borrar registros de la base de datos
        c.execute("DELETE FROM Evidencias WHERE CI_Estudiante = ?", (cedula,))
        c.execute("DELETE FROM Usuarios WHERE CI = ?", (cedula,))
        
        conn.commit()
        conn.close()
        
        # Auditoría
        detalles = f"Usuario {cedula} eliminado. {archivos_borrados} archivos borrados de la nube."
        registrar_auditoria("ELIMINACION_USUARIO", detalles)
        
        return JSONResponse({"status": "ok", "mensaje": f"Usuario y {archivos_borrados} archivos eliminados correctamente."})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# =========================================================================
# 8. ENDPOINTS DE EVIDENCIAS
# =========================================================================

def garantizar_limite_storage(ruta_archivo, limite_mb=1000):
    """
    Si el archivo supera el limite_mb (ej: 1GB), lo comprime para ahorrar espacio.
    Si no lo supera, lo deja intacto (Calidad Original).
    """
    try:
        peso_actual_mb = os.path.getsize(ruta_archivo) / (1024 * 1024)
        
        # CASO 1: El archivo está dentro del límite (Ej: 500MB) -> NO TOCAR
        if peso_actual_mb <= limite_mb:
            return ruta_archivo # Devolvemos la ruta original
            
        # CASO 2: El archivo es gigante (Ej: 2GB) -> COMPRIMIR
        print(f"⚠️ Archivo gigante ({peso_actual_mb:.2f} MB). Comprimiendo para cumplir cuota de 1GB...")
        
        # Nombre para el archivo comprimido
        dir_name = os.path.dirname(ruta_archivo)
        base_name = os.path.basename(ruta_archivo)
        ruta_comprimida = os.path.join(dir_name, f"compressed_{base_name}")
        
        # Detectar si es VIDEO
        ext = os.path.splitext(ruta_archivo)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Usamos OpenCV para reducir resolución y bitrate
            cap = cv2.VideoCapture(ruta_archivo)
            
            # Calculamos nueva resolución (HD 720p suele ser suficiente y ligero)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Si es 4K, lo bajamos a 720p. Si es menor, lo bajamos un 30%
            scale = 0.5 if width > 1920 else 0.7
            new_w, new_h = int(width * scale), int(height * scale)
            
            # Configurar el escritor de video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ruta_comprimida, fourcc, 24, (new_w, new_h))
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                # Redimensionar frame
                frame_b = cv2.resize(frame, (new_w, new_h))
                out.write(frame_b)
                
            cap.release()
            out.release()
            
            # Reemplazar archivo original con el comprimido
            if os.path.exists(ruta_comprimida):
                shutil.move(ruta_comprimida, ruta_archivo) # Sobrescribimos el original
                nuevo_peso = os.path.getsize(ruta_archivo) / (1024 * 1024)
                print(f"✅ Compresión exitosa. Nuevo peso: {nuevo_peso:.2f} MB")
                return ruta_archivo
                
    except Exception as e:
        print(f"⚠️ Error intentando comprimir video storage: {e}")
    
    # Si algo falla o no es video, devolvemos el original
    return ruta_archivo

@app.post("/subir_evidencia_ia")
async def subir_evidencia_ia(archivo: UploadFile = File(...)):
    temp_dir = None
    try:
        # 1. Guardar archivo temporalmente para análisis
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f: shutil.copyfileobj(archivo.file, f)
        
        # 2. Calcular Huella Digital (Hash)
        file_hash = calcular_hash(path)
        
        # 3. Detectar Tipo de Archivo
        ext = os.path.splitext(archivo.filename)[1].lower()
        es_imagen = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif']
        es_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
        tipo_archivo = "video" if es_video else ("imagen" if es_imagen else "documento")
        
        # 4. EJECUTAR IA (Detectar a TODOS los que aparecen)
        cedulas_detectadas = set() 
        texto_debug = []
        
        if rekog:
            if es_imagen:
                # Nota: identificar_varios_rostros_aws ya está configurado con MaxFaces=5
                rostros = identificar_varios_rostros_aws(path)
                cedulas_detectadas.update(rostros)
                
                textos_ceds, debug_ocr = buscar_estudiantes_por_texto(path, get_db_connection())
                cedulas_detectadas.update(textos_ceds)
                if debug_ocr: texto_debug = debug_ocr
                
            elif es_video:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 24
                
                # --- CAMBIO 1: MENOS CARGA DE PROCESAMIENTO ---
                # Antes analizábamos cada 2 segundos. Ahora cada 4 segundos para ahorrar RAM.
                intervalo = int(fps * 4) 
                
                curr = 0
                max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # --- CAMBIO 2: LÍMITE DE SEGURIDAD ---
                # Si el video es muy largo, solo analizamos los primeros 10 análisis para no saturar
                contador_analisis = 0
                MAX_ANALISIS_POR_VIDEO = 8 
                
                while curr < max_frames and contador_analisis < MAX_ANALISIS_POR_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Redimensionar frame si es muy grande (4K -> HD) para ahorrar memoria
                    if frame.shape[1] > 1280:
                        scale = 1280 / frame.shape[1]
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)

                    frame_path = os.path.join(temp_dir, f"frame_{curr}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    rostros = identificar_varios_rostros_aws(frame_path)
                    cedulas_detectadas.update(rostros)
                    
                    # Limpiamos el frame del disco inmediatamente
                    try: os.remove(frame_path) 
                    except: pass
                    
                    curr += intervalo
                    contador_analisis += 1
                    
                    # Si ya encontramos gente, no necesitamos seguir machacando el servidor
                    if len(cedulas_detectadas) >= 5: break
                
                cap.release()

        # 5. VERIFICAR SI EL ARCHIVO YA EXISTE FÍSICAMENTE
        conn = get_db_connection()
        archivo_existente = conn.execute("SELECT Url_Archivo, Tamanio_KB FROM Evidencias WHERE Hash = ? LIMIT 1", (file_hash,)).fetchone()
        
        url_final = ""
        tamanio_kb = 0
        
        if archivo_existente:
            url_final = archivo_existente['Url_Archivo']
            tamanio_kb = archivo_existente['Tamanio_KB']
            print(f"♻️ Archivo existente detectado. Reusando URL: {url_final}")
        else:
            path = garantizar_limite_storage(path, limite_mb=1000)
            tamanio_kb = os.path.getsize(path) / 1024
            
            url_final = f"/local/{archivo.filename}"
            if s3_client:
                try:
                    nube = f"evidencias/{int(ahora_ecuador().timestamp())}_{archivo.filename}"
                    ct = 'video/mp4' if es_video else archivo.content_type
                    s3_client.upload_file(path, BUCKET_NAME, nube, ExtraArgs={'ACL':'public-read', 'ContentType': ct})
                    url_final = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nube}"
                except: pass

        # 6. ASIGNACIÓN INTELIGENTE (FILTRANDO ADMINS)
        asignados_nuevos = []
        ya_lo_tenian = []
        
        if cedulas_detectadas:
            for ced in cedulas_detectadas:
                # Paso A: Traemos también el 'Tipo' del usuario
                u = conn.execute("SELECT Nombre, Apellido, Tipo FROM Usuarios WHERE CI=?", (ced,)).fetchone()
                
                # --- FILTRO DE SEGURIDAD: SOLO ESTUDIANTES (Tipo 1) ---
                if u and u['Tipo'] == 1:
                    nombre_completo = f"{u['Nombre']} {u['Apellido']}"
                    
                    # Paso B: ¿Ya lo tiene?
                    ya_existe = conn.execute("""
                        SELECT id FROM Evidencias 
                        WHERE CI_Estudiante = ? AND Hash = ?
                    """, (ced, file_hash)).fetchone()
                    
                    if ya_existe:
                        ya_lo_tenian.append(nombre_completo)
                    else:
                        conn.execute("""
                            INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente) 
                            VALUES (?, ?, ?, 1, ?, ?, 1)
                        """, (ced, url_final, file_hash, tipo_archivo, tamanio_kb))
                        asignados_nuevos.append(nombre_completo)
                
                # Si u['Tipo'] == 0 (Admin), simplemente no hacemos nada y el bucle continúa.

            # Generar Mensaje
            if asignados_nuevos:
                msg = f"✅ Asignado a: {', '.join(asignados_nuevos)}."
                status = "exito"
                if ya_lo_tenian:
                    msg += f" (Omitidos: {', '.join(ya_lo_tenian)})"
            elif ya_lo_tenian:
                msg = f"⚠️ Evidencia ya existente para: {', '.join(ya_lo_tenian)}."
                status = "alerta"
            else:
                # Detectó rostros pero eran Admins o no registrados
                check_pendiente = conn.execute("SELECT id FROM Evidencias WHERE Hash = ? AND CI_Estudiante = 'PENDIENTE'", (file_hash,)).fetchone()
                if not check_pendiente:
                    conn.execute("""
                        INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente) 
                        VALUES ('PENDIENTE', ?, ?, 1, ?, ?, 0)
                    """, (url_final, file_hash, tipo_archivo, tamanio_kb))
                msg = "⚠️ Rostros detectados pero no son estudiantes activos."
                status = "alerta"

        else:
            # 7. NADIE DETECTADO
            check_pendiente = conn.execute("SELECT id FROM Evidencias WHERE Hash = ?", (file_hash,)).fetchone()
            
            if check_pendiente:
                status = "error"
                msg = "⚠️ DUPLICADO: Este archivo ya existe y la IA no encontró nuevos estudiantes."
            else:
                conn.execute("""
                    INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente) 
                    VALUES ('PENDIENTE', ?, ?, 1, ?, ?, 0)
                """, (url_final, file_hash, tipo_archivo, tamanio_kb))
                
                muestras_texto = ", ".join(texto_debug[:5]) if texto_debug else "Nada legible"
                msg = f"⚠️ No se identificó alumno. Guardado como pendiente. (Texto: {muestras_texto}...)"
                status = "alerta"

        conn.commit()
        conn.close()
        shutil.rmtree(temp_dir)
        return JSONResponse({"status": status, "mensaje": msg})

    except Exception as e:
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return JSONResponse({"status": "error", "mensaje": str(e)})
    
@app.post("/subir_manual")
async def subir_manual(
    cedulas: str = Form(...), 
    archivo: UploadFile = File(...), 
    comentario: Optional[str] = Form(None)
):
    try:
        lista_cedulas = [c.strip() for c in cedulas.split(",") if c.strip()]
        if not lista_cedulas:
            return JSONResponse(content={"error": "Debe especificar al menos una cédula"})
        
        # 1. Guardar temporalmente
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f: shutil.copyfileobj(archivo.file, f)

        # --- NUEVO: DETECCIÓN DE DUPLICADOS ---
        file_hash = calcular_hash(path)
        conn = get_db_connection()
        duplicado = conn.execute("SELECT id FROM Evidencias WHERE Hash = ?", (file_hash,)).fetchone()
        
        if duplicado:
            conn.close()
            shutil.rmtree(temp_dir)
            # Mostramos un error claro en el frontend
            return JSONResponse({"error": "⚠️ ARCHIVO DUPLICADO: Esta evidencia ya fue subida anteriormente."})
        # --------------------------------------
            
        # 2. Subir a la nube
        tamanio_kb = os.path.getsize(path) / 1024
        url_archivo = f"/local/{archivo.filename}"
        
        if s3_client:
            try:
                nombre_nube = f"evidencias/manual_{int(ahora_ecuador().timestamp())}_{archivo.filename}"
                s3_client.upload_file(path, BUCKET_NAME, nombre_nube, ExtraArgs={'ACL': 'public-read'})
                url_archivo = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
            except: pass
            
        c = conn.cursor()
        count = 0
        
        # 3. Guardar en BD (Incluyendo Hash)
        for ced in lista_cedulas:
            if c.execute("SELECT CI FROM Usuarios WHERE CI=?", (ced,)).fetchone():
                c.execute("""
                    INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                    VALUES (?, ?, ?, 1, 'documento', ?, 0)
                """, (ced, url_archivo, file_hash, tamanio_kb))
                count += 1
        
        conn.commit()
        conn.close()
        shutil.rmtree(temp_dir)
        
        return JSONResponse({"status": "ok", "mensaje": f"Asignado a {count} estudiantes"})
    except Exception as e:
        return JSONResponse({"error": str(e)})
    
# =========================================================================
# 9. ENDPOINTS DE BACKUP Y MANTENIMIENTO
# =========================================================================

@app.get("/crear_backup")
async def crear_backup():
    """Crea y descarga una copia de seguridad de la base de datos"""
    try:
        # Crear nombre de archivo con fecha de Ecuador
        fecha = ahora_ecuador().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_despertar_{fecha}.db"
        backup_path = os.path.join(tempfile.gettempdir(), backup_filename)
        
        # Copiar base de datos
        shutil.copy2(DB_NAME, backup_path)
        
        # Registrar auditoría
        registrar_auditoria("CREACION_BACKUP", f"Backup creado: {backup_filename}")
        
        # Preparar respuesta para descarga directa
        def iterfile():
            with open(backup_path, "rb") as f:
                yield from f
            
            # Eliminar archivo temporal después de enviar
            os.remove(backup_path)
        
        return StreamingResponse(
            iterfile(),
            media_type="application/x-sqlite3",
            headers={
                "Content-Disposition": f"attachment; filename={backup_filename}",
                "Content-Type": "application/x-sqlite3"
            }
        )
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.get("/descargar_multimedia_zip")
async def descargar_multimedia_zip():
    """Crea y descarga un ZIP con archivos multimedia"""
    try:
        # Crear archivo ZIP en memoria
        zip_buffer = io.BytesIO()
        fecha = ahora_ecuador().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"multimedia_despertar_{fecha}.zip"
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Agregar información del sistema
            info = {
                "fecha_backup": ahora_ecuador().isoformat(),
                "total_usuarios": 0,
                "total_evidencias": 0,
                "sistema": "Despertar Educativo"
            }
            
            zip_file.writestr("INFO_SISTEMA.json", json.dumps(info, indent=2))
            
            # Si hay acceso a S3, simular estructura
            if s3_client:
                # Nota: En producción, aquí se listarían y descargarían archivos reales
                zip_file.writestr("S3_INFO.txt", "Archivos almacenados en Backblaze B2")
            else:
                # Buscar archivos locales
                local_paths = []
                if os.path.exists("/app/datos_persistentes"):
                    for root, dirs, files in os.walk("/app/datos_persistentes"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, "/app/datos_persistentes")
                            try:
                                zip_file.write(file_path, arcname)
                                local_paths.append(arcname)
                            except:
                                pass
                
                if not local_paths:
                    zip_file.writestr("SIN_ARCHIVOS.txt", "No se encontraron archivos multimedia locales")
        
        zip_buffer.seek(0)
        
        # Registrar auditoría
        registrar_auditoria("DESCARGA_ZIP", f"ZIP multimedia descargado: {zip_filename}")
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "Content-Type": "application/zip"
            }
        )
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

from urllib.parse import urlparse
import re
import os

@app.post("/optimizar_sistema")
async def optimizar_sistema(background_tasks: BackgroundTasks):
    """
    V4.3 - MANTENIMIENTO COMPLETO + REPORTE SHERLOCK:
    1. 🧲 IMÁN: Recupera archivos perdidos.
    2. 🧟 ZOMBIES: Borra evidencias de usuarios eliminados.
    3. 👻 CAZAFANTASMAS: Borra archivos inexistentes (Local/Nube).
    4. 🤖 TERMINATOR: Borra duplicados físicos reales en S3.
    5. 🕵️ SHERLOCK: Imprime lista de dueños de archivos.
    """
    try:
        def tarea_mantenimiento_profundo():
            print("🔧 INICIANDO MANTENIMIENTO SHERLOCK V4.3 (MODO COMPLETO)...")
            try:
                conn = get_db_connection()
                c = conn.cursor()

                # =========================================================
                # PASO 1: EL IMÁN (Recuperar perdidos)
                # =========================================================
                print("🧲 Paso 1: Atrayendo archivos huérfanos a la Bandeja...")
                c.execute("""
                    UPDATE Evidencias 
                    SET CI_Estudiante = '9999999990' 
                    WHERE CI_Estudiante IS NULL OR CI_Estudiante = 'PENDIENTE' OR CI_Estudiante = ''
                """)

                # =========================================================
                # PASO 2: ELIMINAR ZOMBIES (Usuarios borrados)
                # =========================================================
                print("🧟 Paso 2: Eliminando Zombies...")
                c.execute("DELETE FROM Evidencias WHERE CI_Estudiante NOT IN (SELECT CI FROM Usuarios)")
                if c.rowcount > 0:
                    print(f"   💀 {c.rowcount} zombies eliminados.")

                # =========================================================
                # PASO 3: CAZAFANTASMAS (Verificación REAL en Nube/Disco)
                # =========================================================
                print("👻 Paso 3: Cazando fantasmas (Validación física)...")
                evidencias = c.execute("SELECT id, Url_Archivo FROM Evidencias").fetchall()
                fantasmas = 0
                
                for ev in evidencias:
                    url = ev['Url_Archivo']
                    existe = False
                    peso_kb = 0
                    
                    # A. Verificación NUBE
                    if s3_client and BUCKET_NAME in url:
                        try:
                            parsed = urlparse(url)
                            key = parsed.path.lstrip('/')
                            meta = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                            peso_kb = meta['ContentLength'] / 1024
                            existe = True
                        except Exception as e:
                            if "404" in str(e) or "Not Found" in str(e): existe = False
                            else: existe = True 
                    
                    # B. Verificación LOCAL 
                    elif "/local/" in url:
                        ruta_fisica = url.replace("/local/", "./").lstrip("/")
                        if not os.path.exists(ruta_fisica):
                            # Intento ruta absoluta para Docker
                            ruta_fisica = os.path.join(os.getcwd(), url.replace("/local/", "").lstrip("/"))
                        
                        if os.path.exists(ruta_fisica):
                            existe = True
                            peso_kb = os.path.getsize(ruta_fisica) / 1024
                        else:
                            existe = False

                    # ACCIÓN
                    if existe:
                        if peso_kb > 0:
                            c.execute("UPDATE Evidencias SET Tamanio_KB = ? WHERE id = ?", (peso_kb, ev['id']))
                    else:
                        c.execute("DELETE FROM Evidencias WHERE id = ?", (ev['id'],))
                        fantasmas += 1
                
                print(f"   ✨ {fantasmas} fantasmas eliminados.")

                # =========================================================
                # PASO 4: TERMINATOR (Borrado Físico de Duplicados S3)
                # =========================================================
                print("🤖 Paso 4: Eliminando duplicados físicos en S3...")
                
                def borrar_de_nube_real(url_archivo):
                    if s3_client and BUCKET_NAME in url_archivo:
                        try:
                            parsed = urlparse(url_archivo)
                            key = parsed.path.lstrip('/')
                            s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                            print(f"   🗑️ Borrado S3: {key}")
                        except: pass

                todas = c.execute("SELECT id, CI_Estudiante, Url_Archivo, Hash FROM Evidencias").fetchall()
                vistos = {}
                ids_a_borrar = []
                
                for ev in todas:
                    cedula = ev['CI_Estudiante']
                    url = ev['Url_Archivo']
                    nombre_archivo = url.split('/')[-1]
                    nombre_limpio = re.sub(r'^(manual_)?\d+_', '', nombre_archivo).lower()
                    
                    clave = f"{cedula}|{ev.get('Hash')}" if ev.get('Hash') and ev.get('Hash') != 'PENDIENTE' else f"{cedula}|{nombre_limpio}"
                    
                    if clave in vistos:
                        original = vistos[clave]
                        if url != original['Url_Archivo']: 
                            borrar_de_nube_real(url) # ¡ESTA ES LA LÍNEA CLAVE QUE BORRA EN LA NUBE!
                        ids_a_borrar.append(ev['id'])
                    else:
                        vistos[clave] = ev
                
                if ids_a_borrar:
                    placeholders = ','.join(['?'] * len(ids_a_borrar))
                    c.execute(f"DELETE FROM Evidencias WHERE id IN ({placeholders})", ids_a_borrar)
                    print(f"   ✨ {len(ids_a_borrar)} duplicados eliminados.")
                
                conn.commit() # Guardamos limpieza antes del reporte

                # =========================================================
                # 🕵️ PASO 5: REPORTE SHERLOCK HOLMES
                # =========================================================
                print("\n📋 === REPORTE DE EVIDENCIAS (¿Quién tiene los 87?) ===")
                
                usuarios_con_fotos = c.execute("""
                    SELECT u.Nombre, u.Apellido, u.CI, u.Tipo, u.Activo, COUNT(e.id) as Cantidad
                    FROM Usuarios u
                    JOIN Evidencias e ON u.CI = e.CI_Estudiante
                    GROUP BY u.CI
                    ORDER BY Cantidad DESC
                """).fetchall()

                total_revisado = 0
                for u in usuarios_con_fotos:
                    estado = "🟢 ACTIVO" if u['Activo'] == 1 else "🔴 INACTIVO"
                    rol = "👮 ADMIN" if u['Tipo'] == 0 else "🎓 ESTUDIANTE"
                    print(f"   👤 {u['Nombre']} {u['Apellido']} ({u['CI']})")
                    print(f"      Estado: {estado} | Rol: {rol} | 📂 Archivos: {u['Cantidad']}")
                    total_revisado += u['Cantidad']
                
                print(f"   🔢 TOTAL SUMADO: {total_revisado}")
                print("============================================\n")

                # =========================================================
                # FINALIZACIÓN
                # =========================================================
                conn.isolation_level = None 
                c.execute("VACUUM")
                conn.close()
                
                # Actualizar métricas
                stats = calcular_estadisticas_reales()
                conn2 = get_db_connection()
                conn2.execute("INSERT OR REPLACE INTO Metricas_Sistema (Fecha, Total_Evidencias, Almacenamiento_MB) VALUES (?, ?, ?)", 
                            (ahora_ecuador().date().isoformat(), stats.get('total_evidencias',0), stats.get('almacenamiento_mb',0)))
                conn2.commit()
                conn2.close()
                
                print("✅ MANTENIMIENTO COMPLETO FINALIZADO.")
                
            except Exception as e:
                print(f"❌ Error en mantenimiento: {e}")

        background_tasks.add_task(tarea_mantenimiento_profundo)
        return JSONResponse({"status": "ok", "mensaje": "🕵️ Investigando y limpiando a fondo..."})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
# =========================================================================
# 10. ENDPOINTS DE ESTADÍSTICAS Y REPORTES
# =========================================================================

@app.get("/estadisticas_almacenamiento")
async def estadisticas_almacenamiento():
    """Devuelve estadísticas reales de almacenamiento"""
    try:
        stats = calcular_estadisticas_reales()
        
        # Si no hay AWS configurado, mostrar datos simulados pero claros
        if not rekog:
            stats["aws_configurado"] = False
            stats["nota_aws"] = "AWS Rekognition no configurado - usando datos simulados"
        else:
            stats["aws_configurado"] = True
        
        if not s3_client:
            stats["s3_configurado"] = False
            stats["nota_s3"] = "Backblaze B2 no configurado - usando almacenamiento local"
        else:
            stats["s3_configurado"] = True
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.get("/datos_graficos_dashboard")
async def datos_graficos_dashboard():
    """Provee datos para gráficos del dashboard"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Evolución de registros por mes
        c.execute("""
            SELECT strftime('%Y-%m', Fecha_Registro) as mes,
                   COUNT(*) as cantidad
            FROM Usuarios 
            WHERE Fecha_Registro IS NOT NULL
            GROUP BY mes
            ORDER BY mes DESC
            LIMIT 12
        """)
        evolucion_usuarios = [dict(row) for row in c.fetchall()]
        
        # 2. Distribución de tipos de archivo
        c.execute("""
            SELECT Tipo_Archivo, COUNT(*) as cantidad
            FROM Evidencias
            GROUP BY Tipo_Archivo
        """)
        distribucion_archivos = [dict(row) for row in c.fetchall()]
        
        # 3. Solicitudes por estado
        c.execute("""
            SELECT Estado, COUNT(*) as cantidad
            FROM Solicitudes
            GROUP BY Estado
        """)
        solicitudes_estado = [dict(row) for row in c.fetchall()]
        
        # 4. Top 5 estudiantes con más evidencias
        c.execute("""
            SELECT u.Nombre, u.Apellido, u.CI, COUNT(e.id) as total
            FROM Usuarios u
            LEFT JOIN Evidencias e ON u.CI = e.CI_Estudiante
            WHERE u.Tipo = 1
            GROUP BY u.CI
            ORDER BY total DESC
            LIMIT 5
        """)
        top_estudiantes = [dict(row) for row in c.fetchall()]
        
        # 5. Actividad por hora del día (últimos 7 días)
        c.execute("""
            SELECT strftime('%H', Fecha) as hora,
                   COUNT(*) as actividades
            FROM Auditoria
            WHERE DATE(Fecha) >= DATE('now', '-7 days')
            GROUP BY hora
            ORDER BY hora
        """)
        actividad_horaria = [dict(row) for row in c.fetchall()]
        
        conn.close()
        
        return JSONResponse(content={
            "evolucion_usuarios": evolucion_usuarios,
            "distribucion_archivos": distribucion_archivos,
            "solicitudes_estado": solicitudes_estado,
            "top_estudiantes": top_estudiantes,
            "actividad_horaria": actividad_horaria,
            "fecha_consulta": ahora_ecuador().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# =========================================================================
# 11. ENDPOINTS DE SOLICITUDES Y GESTIÓN
# =========================================================================
@app.get("/obtener_solicitudes")
async def obtener_solicitudes(limit: int = 100):
    """Obtiene las solicitudes del sistema (pendientes e historial)"""
    try:
        conn = get_db_connection()
        # Unir con nombre de usuario para mostrar quién solicita
        rows = conn.execute("""
            SELECT s.*, u.Nombre, u.Apellido 
            FROM Solicitudes s 
            LEFT JOIN Usuarios u ON s.CI_Solicitante = u.CI 
            ORDER BY s.Fecha DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        conn.close()
        return JSONResponse([dict(r) for r in rows])
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
# =========================================================================
# 1. ENDPOINTS DE SOLICITUDES (LADO ESTUDIANTE) - ¡ESTO ES LO QUE TE FALTA!
# =========================================================================

@app.post("/solicitar_recuperacion")
async def solicitar_recuperacion(
    cedula: str = Form(...),
    email: str = Form(...),
    mensaje: Optional[str] = Form(None)
):
    """El estudiante pide recuperar contraseña desde login"""
    try:
        conn = get_db_connection()
        # Verificar si el usuario existe
        user = conn.execute("SELECT Nombre, Apellido FROM Usuarios WHERE CI=?", (cedula,)).fetchone()
        if not user:
            conn.close()
            return JSONResponse({"status": "error", "mensaje": "La cédula no está registrada."})
            
        detalle = f"Solicitud de recuperación. Correo contacto: {email}. "
        if mensaje: detalle += f"Mensaje: {mensaje}"
        
        conn.execute("""
            INSERT INTO Solicitudes (Tipo, CI_Solicitante, Email, Detalle, Estado, Fecha)
            VALUES ('RECUPERACION_CONTRASENA', ?, ?, ?, 'PENDIENTE', ?)
        """, (cedula, email, detalle, ahora_ecuador()))
        
        conn.commit()
        conn.close()
        return JSONResponse({"status": "ok", "mensaje": "Solicitud enviada al administrador."})
    except Exception as e:
        return JSONResponse({"status": "error", "mensaje": str(e)})

@app.post("/solicitar_subida")
async def solicitar_subida(cedula: str = Form(...), archivo: UploadFile = File(...)):
    """El estudiante sube un archivo para aprobación del admin"""
    try:
        # 1. Guardar temporalmente para subir a la nube
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f: shutil.copyfileobj(archivo.file, f)
        
        url_archivo = f"/local/{archivo.filename}"
        if s3_client:
            try:
                # Nombre temporal (propuesta)
                nombre_nube = f"evidencias/propuesta_{int(ahora_ecuador().timestamp())}_{archivo.filename}"
                s3_client.upload_file(path, BUCKET_NAME, nombre_nube, ExtraArgs={'ACL': 'public-read'})
                url_archivo = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
            except: pass
        
        shutil.rmtree(temp_dir)
        
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO Solicitudes (Tipo, CI_Solicitante, Detalle, Evidencia_Reportada_Url, Estado, Fecha)
            VALUES ('SUBIR_EVIDENCIA', ?, 'El estudiante desea agregar esta evidencia.', ?, 'PENDIENTE', ?)
        """, (cedula, url_archivo, ahora_ecuador()))
        
        conn.commit()
        conn.close()
        return JSONResponse({"status": "ok", "mensaje": "Archivo enviado a revisión."})
    except Exception as e:
        return JSONResponse({"status": "error", "mensaje": str(e)})

@app.post("/reportar_evidencia")
async def reportar_evidencia(cedula: str = Form(...), id_evidencia: int = Form(...), motivo: str = Form(...)):
    """El estudiante reporta 'No soy yo'"""
    try:
        conn = get_db_connection()
        
        # Obtener URL para mostrarla al admin
        ev = conn.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_evidencia,)).fetchone()
        url = ev['Url_Archivo'] if ev else ""
        
        conn.execute("""
            INSERT INTO Solicitudes (Tipo, CI_Solicitante, Detalle, Id_Evidencia, Evidencia_Reportada_Url, Estado, Fecha)
            VALUES ('REPORTE_EVIDENCIA', ?, ?, ?, ?, 'PENDIENTE', ?)
        """, (cedula, motivo, id_evidencia, url, ahora_ecuador()))
        
        conn.commit()
        conn.close()
        return JSONResponse({"status": "ok", "mensaje": "Reporte enviado."})
    except Exception as e:
        return JSONResponse({"status": "error", "mensaje": str(e)})

@app.get("/obtener_solicitudes_por_cedula")
async def obtener_solicitudes_por_cedula(cedula: str):
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM Solicitudes WHERE CI_Solicitante=? ORDER BY Fecha DESC", (cedula,)).fetchall()
        conn.close()
        return JSONResponse([dict(r) for r in rows])
    except: return JSONResponse([])

# =========================================================================
# AQUI DEBERÍA SEGUIR TU FUNCIÓN @app.post("/gestionar_solicitud") ...
# =========================================================================

@app.post("/gestionar_solicitud")
async def gestionar_solicitud(
    background_tasks: BackgroundTasks,  # <--- MAGIA AQUÍ: Permite tareas de fondo
    id_solicitud: int = Form(...),
    accion: str = Form(...), 
    mensaje: str = Form(...), 
    id_admin: str = Form("Admin")
):
    try:
        # 1. Normalizar acción
        accion_norm = "APROBADA" if accion.lower() in ['aprobar', 'aceptar', 'aprobada'] else "RECHAZADA"
        fecha_resolucion = ahora_ecuador()
        
        conn = get_db_connection()
        
        # 2. Obtener datos
        sol = conn.execute("""
            SELECT s.*, u.Nombre, u.Apellido, u.Email as UserEmail
            FROM Solicitudes s
            LEFT JOIN Usuarios u ON s.CI_Solicitante = u.CI
            WHERE s.id = ?
        """, (id_solicitud,)).fetchone()
        
        if not sol:
            return JSONResponse({"status": "error", "mensaje": "Solicitud no encontrada"})
        
        tipo = sol['Tipo']
        ci_solicitante = sol['CI_Solicitante']
        email_destino = sol['Email'] if sol['Email'] else sol['UserEmail']
        nombre_usuario = f"{sol['Nombre']} {sol['Apellido']}"
        
        # --- LÓGICA DE ACCIONES ---
        cuerpo_correo = "" # Inicializar variable
        
        if tipo == 'REPORTE_EVIDENCIA':
            id_evidencia = sol['Id_Evidencia']
            if accion_norm == 'APROBADA':
                conn.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
                cuerpo_correo = f"Hola {nombre_usuario},\n\nTu reporte ha sido ACEPTADO. La evidencia ha sido eliminada.\n\nAdmin: {mensaje}"
            else:
                cuerpo_correo = f"Hola {nombre_usuario},\n\nTu reporte fue revisado pero decidimos mantener la evidencia.\n\nMotivo: {mensaje}"

        elif tipo == 'SUBIR_EVIDENCIA':
            url_archivo = sol['Evidencia_Reportada_Url']
            if accion_norm == 'APROBADA':
                conn.execute("""
                    INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                    VALUES (?, ?, 'MANUAL_APROBADO', 1, 'documento', 0, 0)
                """, (ci_solicitante, url_archivo))
                cuerpo_correo = f"Hola {nombre_usuario},\n\nTu solicitud de subida fue APROBADA. Ya está en tu perfil.\n\nAdmin: {mensaje}"
            else:
                cuerpo_correo = f"Hola {nombre_usuario},\n\nTu solicitud de subida fue RECHAZADA.\n\nMotivo: {mensaje}"

        elif tipo in ['RECUPERACION_CONTRASENA', 'PROBLEMA_ERROR']:
            cuerpo_correo = f"Hola {nombre_usuario},\n\nRespuesta a tu solicitud ({tipo.replace('_',' ')}):\n\n{mensaje}\n\nAtentamente,\nSoporte U.E. Despertar"

        else:
            cuerpo_correo = f"Hola {nombre_usuario},\n\nTu solicitud ha sido procesada: {mensaje}"

        # 3. ACTUALIZAR BD
        conn.execute("""
            UPDATE Solicitudes 
            SET Estado=?, Resuelto_Por=?, Respuesta=?, Fecha_Resolucion=?
            WHERE id=?
        """, (accion_norm, id_admin, mensaje, fecha_resolucion, id_solicitud))
        
        conn.commit()
        conn.close()
        
        # 4. ENVIAR CORREO EN SEGUNDO PLANO (AQUÍ ESTÁ LA VELOCIDAD)
        if email_destino:
            asunto = f"Respuesta a Solicitud: {tipo.replace('_', ' ')}"
            # IMPORTANTE: Que los nombres coincidan con los de enviar_correo_real
            background_tasks.add_task(enviar_correo_real, email_destino, asunto, cuerpo_correo)
        
        return JSONResponse({
            "status": "ok", 
            "mensaje": "Acción procesada correctamente (Correo enviándose en segundo plano)."
        })

    except Exception as e:
        return JSONResponse({"status": "error", "mensaje": str(e)})
# =========================================================================
# 12. ENDPOINTS DE LOGS Y AUDITORÍA
# =========================================================================

@app.get("/obtener_logs")
async def obtener_logs(limit: int = 100):
    """Devuelve una LISTA SIMPLE de logs para evitar errores en el admin"""
    try:
        conn = get_db_connection()
        # Traemos todo de auditoria ordenado por fecha
        logs = conn.execute("SELECT * FROM Auditoria ORDER BY Fecha DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        
        # Convertimos a lista de diccionarios simple
        lista_logs = [dict(row) for row in logs]
        return JSONResponse(content=lista_logs) # <--- Enviamos LISTA directa, no objeto
    except Exception as e:
        print(f"Error logs: {e}")
        return JSONResponse(content=[]) # En caso de error, lista vacía para no romper la página
# =========================================================================
# 13. ENDPOINTS EXISTENTES MANTENIDOS
# =========================================================================

@app.get("/listar_usuarios")
async def listar_usuarios():
    """Lista todos los usuarios de forma segura (a prueba de fallos)"""
    try:
        conn = get_db_connection()
        # ✅ TRUCO: Usamos SELECT * para traer lo que haya, sin exigir columnas específicas
        rows = conn.execute("SELECT * FROM Usuarios ORDER BY Apellido, Nombre").fetchall()
        
        usuarios_seguros = []
        for row in rows:
            # Convertimos la fila a diccionario
            r = dict(row)
            
            # Construimos el usuario validando campo por campo
            # Si una columna (como Fecha_Registro) no existe, usa None y NO FALLA
            usuario = {
                "ID": r.get("ID"),
                "Nombre": r.get("Nombre"),
                "Apellido": r.get("Apellido"),
                "CI": r.get("CI"),
                "Tipo": r.get("Tipo"),
                "Activo": r.get("Activo"),
                "Foto": r.get("Foto"),
                "Contrasena": r.get("Password", ""), 
                "Email": r.get("Email", ""),
                "Telefono": r.get("Telefono", ""),
                
                # 👇 AQUÍ ESTÁ EL ARREGLO: Si no existe, pone None y el sistema NO SE ROMPE 👇
                "Fecha_Registro": r.get("Fecha_Registro", None),
                "Ultimo_Acceso": r.get("Ultimo_Acceso", None)
            }
            usuarios_seguros.append(usuario)
            
        conn.close()
        return JSONResponse(content=usuarios_seguros)
    except Exception as e:
        print(f"❌ Error listando usuarios: {e}")
        # En el peor caso devolvemos lista vacía para que el admin cargue igual
        return JSONResponse(content=[])
    
@app.get("/resumen_estudiantes_con_evidencias")
async def resumen_estudiantes_con_evidencias():
    """Resumen de estudiantes con sus evidencias"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT u.Nombre, u.Apellido, u.CI, u.Foto, 
                   COUNT(e.id) as total_evidencias,
                   SUM(e.Tamanio_KB) as total_kb
            FROM Usuarios u
            LEFT JOIN Evidencias e ON u.CI = e.CI_Estudiante AND e.Estado = 1
            WHERE u.Tipo = 1 AND u.Activo = 1
            GROUP BY u.CI
            ORDER BY total_evidencias DESC
        """)
        data = [dict(row) for row in c.fetchall()]
        conn.close()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.get("/todas_evidencias")
async def todas_evidencias(cedula: str):
    """Obtiene todas las evidencias de un estudiante específico"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT id, Url_Archivo, Tipo_Archivo, Fecha, Estado, Tamanio_KB
            FROM Evidencias
            WHERE CI_Estudiante = ?
            ORDER BY Fecha DESC
        """, (cedula,))
        rows = c.fetchall()
        conn.close()
        # Convertimos las filas a diccionarios
        return JSONResponse([dict(r) for r in rows])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/eliminar_evidencia/{id_evidencia}")
async def eliminar_evidencia(id_evidencia: int):
    """Elimina una evidencia del sistema"""
    try:
        conn = get_db_connection()
        
        # Obtener información de la evidencia
        ev = conn.execute("""
            SELECT Url_Archivo, CI_Estudiante 
            FROM Evidencias 
            WHERE id = ?
        """, (id_evidencia,)).fetchone()
        
        if ev:
            # Eliminar de S3 si está configurado
            if s3_client and ev['Url_Archivo'] and BUCKET_NAME in ev['Url_Archivo']:
                try:
                    key = ev['Url_Archivo'].split(f"{BUCKET_NAME}/")[-1]
                    s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                    print(f"✅ Archivo eliminado de S3: {key}")
                except Exception as e:
                    print(f"⚠️ Error eliminando de S3: {e}")
            
            # Eliminar de la base de datos
            conn.execute("DELETE FROM Evidencias WHERE id = ?", (id_evidencia,))
            conn.commit()
            
            # Registrar auditoría
            registrar_auditoria(
                "ELIMINACION_EVIDENCIA",
                f"Evidencia {id_evidencia} eliminada para estudiante {ev['CI_Estudiante']}"
            )
        
        conn.close()
        return JSONResponse(content={"status": "ok", "mensaje": "Evidencia eliminada"})
        
    except Exception as e:
        return JSONResponse(content={"status": "error", "mensaje": str(e)})

@app.get("/diagnostico_usuario/{cedula}")
async def diagnostico_usuario(cedula: str):
    """Diagnóstico completo de un usuario"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Información de la tabla
        c.execute("PRAGMA table_info(Usuarios)")
        columnas = c.fetchall()
        
        # Buscar usuario
        c.execute("SELECT * FROM Usuarios WHERE CI = ?", (cedula,))
        usuario = c.fetchone()
        
        # Evidencias del usuario
        c.execute("""
            SELECT COUNT(*) as total, 
                   SUM(Tamanio_KB) as total_kb,
                   Tipo_Archivo,
                   COUNT(*) as cantidad
            FROM Evidencias
            WHERE CI_Estudiante = ?
            GROUP BY Tipo_Archivo
        """, (cedula,))
        estadisticas_evidencias = c.fetchall()
        
        # Solicitudes del usuario
        c.execute("""
            SELECT Estado, COUNT(*) as cantidad
            FROM Solicitudes
            WHERE CI_Solicitante = ?
            GROUP BY Estado
        """, (cedula,))
        estadisticas_solicitudes = c.fetchall()
        
        conn.close()
        
        return JSONResponse(content={
            "cedula_buscada": cedula,
            "usuario_encontrado": bool(usuario),
            "usuario": usuario,
            "estructura_tabla": columnas,
            "estadisticas_evidencias": estadisticas_evidencias,
            "estadisticas_solicitudes": estadisticas_solicitudes,
            "fecha_diagnostico": ahora_ecuador().isoformat(),
            "zona_horaria": "America/Guayaquil (UTC-5)"
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.get("/reset-db")
async def reset_database():
    """Reinicia la base de datos (SOLO DESARROLLO)"""
    try:
        init_db_completa()
        return JSONResponse(content={
            "status": "ok",
            "mensaje": "Base de datos reinicializada",
            "fecha": ahora_ecuador().isoformat()
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# =========================================================================
# 14. ENDPOINTS CORS Y UTILIDADES
# =========================================================================

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    """Manejador de preflight CORS"""
    response = JSONResponse(content={"message": "Preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/cors-debug")
async def cors_debug():
    """Endpoint para debug de CORS"""
    return JSONResponse(content={
        "message": "CORS Debug Endpoint",
        "allow_origin": "*",
        "allow_methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
        "allow_headers": "*",
        "allow_credentials": "true",
        "timestamp": ahora_ecuador().isoformat(),
        "zona_horaria": "America/Guayaquil (UTC-5)"
    })

# =========================================================================
# 15. INICIO DE LA APLICACIÓN
# =========================================================================

class PasswordRequest(BaseModel):
    cedula: str
    nueva_contrasena: str

@app.post("/cambiar_contrasena")
async def cambiar_contrasena(datos: PasswordRequest):
    try:
        conn = get_db_connection()
        conn.execute("UPDATE Usuarios SET Password = ? WHERE CI = ?", (datos.nueva_contrasena, datos.cedula))
        conn.commit()
        conn.close()
        return JSONResponse({"mensaje": "Contraseña actualizada correctamente"})
    except Exception as e:
        return JSONResponse({"error": str(e)})
    
@app.post("/descargar_evidencias_zip")
async def descargar_evidencias_zip(ids: str = Form(...)):
    try:
        id_list = ids.split(',')
        zip_buffer = io.BytesIO()
        
        conn = get_db_connection()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for id_ev in id_list:
                row = conn.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_ev,)).fetchone()
                if row:
                    url = row['Url_Archivo']
                    filename = url.split('/')[-1]
                    # Aquí simulamos el archivo creando un txt con la URL
                    # (Para descarga real necesitarías descargar de S3 primero)
                    zip_file.writestr(filename + ".txt", f"Archivo ubicado en: {url}")
        
        conn.close()
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=seleccion_evidencias.zip"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)})


from urllib.parse import urlparse 

def limpieza_duplicados_startup():
    """
    V5.2 - VERSIÓN MAESTRA: Incluye TODAS las fases (0, 1, 2, 3) restauradas
    y la Fase 4 corregida para Backblaze.
    """
    print("🧹 INICIANDO PROTOCOLO DE LIMPIEZA Y MANTENIMIENTO...")
    
    try:
        conn = get_db_connection()
        
        # ---------------------------------------------------------
        # FASE 0: LIMPIEZA POR URL EXACTA
        # ---------------------------------------------------------
        print("🔍 FASE 0: Buscando URLs idénticas...")
        urls_repetidas = conn.execute("""
            SELECT Url_Archivo, COUNT(*) as cantidad FROM Evidencias 
            GROUP BY Url_Archivo HAVING cantidad > 1
        """).fetchall()
        
        eliminados_0 = 0
        for row in urls_repetidas:
            copias = conn.execute("SELECT id FROM Evidencias WHERE Url_Archivo = ? ORDER BY id ASC", (row['Url_Archivo'],)).fetchall()
            for copia in copias[1:]: # Borrar todos menos el primero (el original)
                conn.execute("DELETE FROM Evidencias WHERE id = ?", (copia['id'],))
                eliminados_0 += 1
        
        if eliminados_0 > 0: print(f"   ✨ Fase 0: {eliminados_0} registros eliminados.")

        # ---------------------------------------------------------
        # FASE 1: REPARAR ARCHIVOS ANTIGUOS (CALCULAR HASHES)
        # ---------------------------------------------------------
        print("⏳ FASE 1: Verificando huellas digitales...")
        pendientes = conn.execute("SELECT id, Url_Archivo FROM Evidencias WHERE Hash = 'PENDIENTE' OR Hash IS NULL").fetchall()
        
        count_hashed = 0
        for row in pendientes:
            try:
                url = row['Url_Archivo']
                temp_path = None
                file_hash = None
                
                # Intento de descarga para calcular hash
                if "http" in url and s3_client:
                    try:
                        # Extraer key usando urlparse para seguridad
                        parsed = urlparse(url)
                        key = parsed.path.lstrip('/')
                        
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            s3_client.download_fileobj(BUCKET_NAME, key, tmp)
                            temp_path = tmp.name
                        file_hash = calcular_hash(temp_path)
                    except: pass 
                
                # Si es local
                elif "/local/" in url:
                    # Ajuste de ruta si es necesario
                    ruta_local = url.replace("/local/", "./")
                    if not os.path.exists(ruta_local):
                         # Intento ruta absoluta para Docker
                         ruta_local = os.path.join(BASE_DIR, url.replace("/local/", "").lstrip("/"))
                    
                    if os.path.exists(ruta_local):
                        file_hash = calcular_hash(ruta_local)

                if file_hash:
                    conn.execute("UPDATE Evidencias SET Hash = ? WHERE id = ?", (file_hash, row['id']))
                    count_hashed += 1
                
                if temp_path and os.path.exists(temp_path): os.remove(temp_path)
            except: pass
            
        conn.commit()
        if count_hashed > 0: print(f"   ✨ Fase 1: {count_hashed} hashes calculados.")

        # ---------------------------------------------------------
        # FASE 2: ELIMINAR POR HASH (CONTENIDO IDÉNTICO)
        # ---------------------------------------------------------
        print("🔍 FASE 2: Buscando contenido idéntico (Hash)...")
        grupos_hash = conn.execute("""
            SELECT Hash, COUNT(*) as cantidad FROM Evidencias 
            WHERE Hash NOT IN ('PENDIENTE', '', 'RECUPERADO') GROUP BY Hash HAVING cantidad > 1
        """).fetchall()
        
        eliminados_2 = 0
        for grupo in grupos_hash:
            copias = conn.execute("SELECT id, Url_Archivo FROM Evidencias WHERE Hash = ? ORDER BY id ASC", (grupo['Hash'],)).fetchall()
            original = copias[0]
            for copia in copias[1:]:
                # Intentar borrar de S3 si la URL es diferente a la original
                if s3_client and copia['Url_Archivo'] != original['Url_Archivo'] and BUCKET_NAME in copia['Url_Archivo']:
                    try:
                        parsed = urlparse(copia['Url_Archivo'])
                        key = parsed.path.lstrip('/')
                        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                    except: pass
                
                conn.execute("DELETE FROM Evidencias WHERE id = ?", (copia['id'],))
                eliminados_2 += 1
        
        if eliminados_2 > 0: print(f"   ✨ Fase 2: {eliminados_2} duplicados exactos eliminados.")

        # ---------------------------------------------------------
        # FASE 3: ELIMINAR POR NOMBRE + ESTUDIANTE (HEURÍSTICA)
        # ---------------------------------------------------------
        print("🔍 FASE 3: Buscando duplicados por nombre de archivo...")
        todas = conn.execute("SELECT id, CI_Estudiante, Url_Archivo FROM Evidencias").fetchall()
        agrupados = {}
        import re
        eliminados_3 = 0
        
        for ev in todas:
            url = ev['Url_Archivo']
            cedula = ev['CI_Estudiante']
            filename = os.path.basename(url)
            # Limpiamos timestamp inicial (ej: 123456_foto.jpg -> foto.jpg)
            clean_name = re.sub(r'^\d+_', '', filename)
            clave = f"{cedula}|{clean_name}"
            
            if clave not in agrupados: agrupados[clave] = []
            agrupados[clave].append(ev)
            
        for clave, lista in agrupados.items():
            if len(lista) > 1:
                lista.sort(key=lambda x: x['id'])
                # original = lista[0] # El primero se queda
                duplicados = lista[1:] # El resto se va
                for dup in duplicados:
                    conn.execute("DELETE FROM Evidencias WHERE id = ?", (dup['id'],))
                    # Intento de borrado físico
                    if s3_client and BUCKET_NAME in dup['Url_Archivo']:
                        try:
                            parsed = urlparse(dup['Url_Archivo'])
                            key = parsed.path.lstrip('/')
                            s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                        except: pass
                    eliminados_3 += 1

        if eliminados_3 > 0: print(f"   ✨ Fase 3: {eliminados_3} archivos eliminados por nombre.")

        # ---------------------------------------------------------
        # FASE 4: SINCRONIZACIÓN SEGURA CON NUBE (Corrección 404)
        # ---------------------------------------------------------
        print("☁️ FASE 4: Auditando existencia real en la nube (Modo Seguro)...")
        
        # Recargamos la lista actualizada
        conn.commit()
        evidencias = conn.execute("SELECT id, Url_Archivo FROM Evidencias").fetchall()
        
        actualizados_peso = 0
        eliminados_nube = 0
        
        for ev in evidencias:
            url = ev['Url_Archivo']
            existe = False
            peso_kb = 0
            
            # A) Si es archivo en Nube
            if s3_client and BUCKET_NAME in url:
                try:
                    # CORRECCIÓN CRÍTICA: Usamos urlparse
                    parsed = urlparse(url)
                    key = parsed.path.lstrip('/')
                    
                    # Preguntamos a Backblaze
                    meta = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                    peso_kb = meta['ContentLength'] / 1024
                    existe = True
                except Exception as e:
                    # Solo marcamos como inexistente si es un error 404 REAL
                    error_msg = str(e)
                    if "404" in error_msg or "Not Found" in error_msg:
                        existe = False
                    else:
                        # Si es otro error (conexión, permisos), asumimos que EXISTE para no borrarlo por error
                        existe = True 
                        # print(f"   ⚠️ Error conexión S3 ID {ev['id']}: {e}") 
            
            # B) Si es archivo Local
            elif "/local/" in url:
                existe = True # En local asumimos que existe para evitar borrados accidentales en Docker
                # Lógica opcional para local...
            
            # ACCIONES FASE 4
            if existe:
                # Actualizamos el peso real
                conn.execute("UPDATE Evidencias SET Tamanio_KB = ? WHERE id = ?", (peso_kb, ev['id']))
                actualizados_peso += 1
            else:
                # Solo borramos si estamos SEGUROS de que es un 404
                print(f"   👻 Eliminando fantasma CONFIRMADO ID {ev['id']} (404 en nube)")
                conn.execute("DELETE FROM Evidencias WHERE id = ?", (ev['id'],))
                eliminados_nube += 1
        
        conn.commit()
        print(f"✅ FASE 4 COMPLETADA: {actualizados_peso} pesos actualizados, {eliminados_nube} fantasmas eliminados.")
        
        # --- FINALIZACIÓN ---
        # Recalcular estadísticas inmediatamente
        try:
            stats = calcular_estadisticas_reales()
            fecha_hoy = ahora_ecuador().date().isoformat()
            conn.execute("""
                INSERT OR REPLACE INTO Metricas_Sistema 
                (Fecha, Total_Usuarios, Total_Evidencias, Solicitudes_Pendientes, Almacenamiento_MB)
                VALUES (?, ?, ?, ?, ?)
            """, (fecha_hoy, stats.get("usuarios_activos",0), stats.get("total_evidencias",0), 
                  stats.get("solicitudes_pendientes",0), stats.get("almacenamiento_mb",0)))
            conn.commit()
        except: pass
        
        conn.close()
        
        total_limpieza = eliminados_0 + eliminados_2 + eliminados_3 + eliminados_nube
        print(f"✅ MANTENIMIENTO TOTAL FINALIZADO. Total registros limpiados: {total_limpieza}")

    except Exception as e:
        print(f"❌ Error en limpieza startup: {e}")

    # --- AGREGA ESTE NUEVO ENDPOINT PARA RECUPERAR TUS DATOS ---

@app.post("/recuperar_evidencias_nube")
async def recuperar_evidencias_nube(background_tasks: BackgroundTasks):
    """
    Escanea TODO el bucket de Backblaze, encuentra los archivos que faltan en la BD
    y los restaura automáticamente (intentando usar IA para re-asignarlos).
    """
    def tarea_rescate():
        print("🚑 INICIANDO OPERACIÓN RESCATE DE EVIDENCIAS...")
        if not s3_client:
            print("❌ No hay conexión S3 para el rescate.")
            return

        conn = get_db_connection()
        restaurados = 0
        
        try:
            # 1. Listar TODOS los objetos en la carpeta 'evidencias/' de la nube
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix='evidencias/')
            
            for page in pages:
                if 'Contents' not in page: continue
                
                for obj in page['Contents']:
                    key = obj['Key'] # Ejemplo: evidencias/123_foto.jpg
                    
                    # Ignoramos carpetas vacías
                    if key.endswith('/'): continue
                    
                    # Construimos la URL que debería tener
                    url_archivo = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{key}"
                    
                    # 2. Verificar si ya existe en la BD
                    existe = conn.execute("SELECT id FROM Evidencias WHERE Url_Archivo = ?", (url_archivo,)).fetchone()
                    
                    if not existe:
                        print(f"   📥 Recuperando: {key}...")
                        
                        # Descargar para análisis IA (si es imagen)
                        temp_path = None
                        ci_detectada = 'PENDIENTE' # Por defecto
                        asignado_auto = 0
                        
                        try:
                            # Descargar archivo temporal
                            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                                s3_client.download_fileobj(BUCKET_NAME, key, tmp)
                                temp_path = tmp.name
                            
                            # Intentar reconocer rostros para re-asignar
                            if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
                                rostros = identificar_varios_rostros_aws(temp_path)
                                if rostros:
                                    # Si encuentra varios, asignamos al primero que encontremos registrado (simplificado para rescate)
                                    for rostro in rostros:
                                        u = conn.execute("SELECT CI FROM Usuarios WHERE CI=?", (rostro,)).fetchone()
                                        if u:
                                            ci_detectada = u['CI']
                                            asignado_auto = 1
                                            break
                            
                            # Calcular Hash nuevo
                            nuevo_hash = calcular_hash(temp_path)
                            size_kb = obj['Size'] / 1024
                            
                            # Insertar en BD
                            tipo = 'video' if key.lower().endswith(('.mp4', '.avi')) else 'imagen'
                            conn.execute("""
                                INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                                VALUES (?, ?, ?, 1, ?, ?, ?)
                            """, (ci_detectada, url_archivo, nuevo_hash, tipo, size_kb, asignado_auto))
                            
                            restaurados += 1
                            
                            if temp_path and os.path.exists(temp_path): os.remove(temp_path)
                            
                        except Exception as e:
                            print(f"   ⚠️ Error parcial recuperando {key}: {e}")
                            # Intentamos insertar aunque sea sin IA
                            try:
                                conn.execute("""
                                    INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                                    VALUES ('9999999990', ?, 'RECUPERADO', 1, 'desconocido', 0, 0)
                                """, (url_archivo,))
                                restaurados += 1
                            except: pass

            conn.commit()
            print(f"✅ OPERACIÓN RESCATE FINALIZADA: {restaurados} evidencias restauradas.")
            
            # Actualizar estadísticas
            stats = calcular_estadisticas_reales()
            fecha = ahora_ecuador().date().isoformat()
            conn.execute("INSERT OR REPLACE INTO Metricas_Sistema (Fecha, Total_Evidencias, Almacenamiento_MB) VALUES (?, ?, ?)", 
                        (fecha, stats.get('total_evidencias',0), stats.get('almacenamiento_mb',0)))
            conn.commit()

        except Exception as e:
            print(f"❌ Error crítico en rescate: {e}")
        finally:
            conn.close()

    background_tasks.add_task(tarea_rescate)
    return JSONResponse({"mensaje": "🚑 Rescate iniciado. Revisa los logs en 2 minutos."})

class ReasignarRequest(BaseModel):
    ids: str
    cedula_destino: str

@app.post("/reasignar_evidencias")
async def reasignar_evidencias(datos: ReasignarRequest):
    """
    V2.0 - Reasignación Multi-Destino:
    - Permite enviar una lista de cédulas destino (separadas por comas).
    - Al primer destino le MUEVE el archivo.
    - A los siguientes destinos les crea una COPIA (Clon) en la base de datos.
    """
    try:
        if not datos.ids or not datos.cedula_destino:
            return JSONResponse({"error": "Faltan datos"})
            
        ids_evidencias = [id.strip() for id in datos.ids.split(',') if id.strip()]
        cedulas_destino = [ced.strip() for ced in datos.cedula_destino.split(',') if ced.strip()]
        
        if not ids_evidencias or not cedulas_destino:
             return JSONResponse({"error": "Selección inválida"})

        conn = get_db_connection()
        c = conn.cursor()
        
        movidos = 0
        clonados = 0
        
        # 1. Obtener datos originales de las evidencias antes de moverlas
        placeholders = ','.join(['?'] * len(ids_evidencias))
        evidencias_originales = c.execute(f"SELECT * FROM Evidencias WHERE id IN ({placeholders})", ids_evidencias).fetchall()

        # 2. PROCESAR CADA EVIDENCIA
        for ev in evidencias_originales:
            # A) Mover al PRIMER estudiante de la lista (UPDATE)
            primer_destino = cedulas_destino[0]
            c.execute("UPDATE Evidencias SET CI_Estudiante = ?, Asignado_Automaticamente = 0 WHERE id = ?", (primer_destino, ev['id']))
            movidos += 1
            
            # B) Clonar para el RESTO de estudiantes (INSERT)
            if len(cedulas_destino) > 1:
                for otro_destino in cedulas_destino[1:]:
                    c.execute("""
                        INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                    """, (otro_destino, ev['Url_Archivo'], ev['Hash'], ev['Estado'], ev['Tipo_Archivo'], ev['Tamanio_KB']))
                    clonados += 1

        conn.commit()
        conn.close()
        
        mensaje = f"✅ Archivos movidos a 1 estudiante."
        if clonados > 0:
            mensaje += f" Y se crearon copias para {len(cedulas_destino)-1} estudiantes más."
        
        return JSONResponse({"mensaje": mensaje})
        
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    
    # Configuración del puerto
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 60)
    print("🚀 SISTEMA EDUCATIVO DESPERTAR - BACKEND V7.0")
    print("=" * 60)
    print(f"📁 Base de datos: {DB_NAME}")
    print(f"🌍 Zona horaria: America/Guayaquil (UTC-5)")
    print(f"🤖 AWS Rekognition: {'✅ Disponible' if rekog else '❌ No disponible'}")
    print(f"💾 S3 Storage: {'✅ Disponible' if s3_client else '❌ No disponible'}")
    print(f"📧 Servidor SMTP: {'✅ Configurado' if SMTP_EMAIL and 'tu_correo' not in SMTP_EMAIL else '⚠️ Simulado'}")
    print(f"🔐 Usuario admin: 9999999999 / admin123")
    
    # 👇👇👇 ESTA LÍNEA ES LA CLAVE QUE TE FALTABA 👇👇👇
    limpieza_duplicados_startup()
    # 👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆
    
    print(f"🌐 Servidor iniciado en: http://0.0.0.0:{port}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)