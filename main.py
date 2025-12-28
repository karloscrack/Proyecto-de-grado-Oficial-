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
SMTP_PORT = 587
SMTP_EMAIL = "tu_correo_sistema@gmail.com"
SMTP_PASSWORD = "tu_contraseña_aplicacion"

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
            
            # 👇 ESTA ES LA LÍNEA QUE TE FALTA Y ARREGLA EL ERROR 👇
            ("Usuarios", "Fecha_Registro", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            
            ("Evidencias", "Tipo_Archivo", "TEXT DEFAULT 'documento'"),
            ("Evidencias", "Tamanio_KB", "REAL DEFAULT 0"),
            ("Evidencias", "Asignado_Automaticamente", "INTEGER DEFAULT 0"),
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
    """
    Envía un correo electrónico real usando SMTP
    Retorna True si fue exitoso, False si falló
    """
    try:
        # Si las credenciales no están configuradas, simular envío
        if "tu_correo" in SMTP_EMAIL or not SMTP_PASSWORD:
            print(f"📧 [SIMULACION EMAIL] A: {destinatario} | Asunto: {asunto}")
            print(f"   Mensaje: {mensaje[:100]}...")
            return True  # Simulamos éxito para desarrollo
            
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = destinatario
        msg['Subject'] = asunto
        
        if html:
            msg.attach(MIMEText(mensaje, 'html'))
        else:
            msg.attach(MIMEText(mensaje, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_EMAIL, destinatario, text)
        server.quit()
        
        logging.info(f"Correo enviado exitosamente a {destinatario}")
        return True
    except Exception as e:
        logging.error(f"❌ Error enviando email a {destinatario}: {e}")
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

def identificar_varios_rostros_aws(imagen_path: str, confidence_threshold: float = 80.0) -> List[str]:
    """
    Versión CORREGIDA: Lee el ExternalImageId (Cédula) en lugar del FaceId (UUID).
    """
    if not rekog:
        return []
    
    cedulas_encontradas = set()
    
    try:
        # 1. Cargar imagen con OpenCV
        img = cv2.imread(imagen_path)
        if img is None: return []
        height, width, _ = img.shape
        
        # 2. Leer bytes para AWS
        with open(imagen_path, 'rb') as image_file:
            image_bytes = image_file.read()
            
        # 3. Detectar TODAS las caras primero
        response_detect = rekog.detect_faces(Image={'Bytes': image_bytes})
        
        if not response_detect['FaceDetails']:
            return []

        # 4. Procesar cada cara encontrada
        for faceDetail in response_detect['FaceDetails']:
            bbox = faceDetail['BoundingBox']
            
            # Calcular recorte exacto
            x = int(bbox['Left'] * width)
            y = int(bbox['Top'] * height)
            w = int(bbox['Width'] * width)
            h = int(bbox['Height'] * height)
            
            # Ajustar márgenes
            x, y = max(0, x), max(0, y)
            w, h = min(width - x, w), min(height - y, h)
            
            face_crop = img[y:y+h, x:x+w]
            if face_crop.size == 0: continue

            # Convertir a jpg
            _, buffer = cv2.imencode('.jpg', face_crop)
            crop_bytes = buffer.tobytes()
            
            # 5. Buscar quién es esta persona
            try:
                search_res = rekog.search_faces_by_image(
                    CollectionId=COLLECTION_ID,
                    Image={'Bytes': crop_bytes},
                    MaxFaces=1, 
                    FaceMatchThreshold=confidence_threshold
                )
                
                if search_res['FaceMatches']:
                    # --- CORRECCIÓN AQUÍ ---
                    # Antes leía 'FaceId' (UUID), ahora lee 'ExternalImageId' (Cédula)
                    face_data = search_res['FaceMatches'][0]['Face']
                    ced = face_data.get('ExternalImageId') 
                    
                    if ced:
                        cedulas_encontradas.add(ced)
                        print(f"✅ Rostro identificado correctamente: {ced}")
                    else:
                        print("⚠️ Rostro encontrado pero sin Cédula asociada en AWS")
                        
            except Exception as e_inner:
                continue 

        return list(cedulas_encontradas)
        
    except Exception as e:
        print(f"Error IA: {e}")
        return []
    
def buscar_estudiantes_por_texto(imagen_path: str, conn) -> List[str]:
    """
    Versión DEFINITIVA (OCR Flexible):
    1. 'Bag of Words': Busca en TODA la imagen mezclada, no importa si el nombre está en dos renglones.
    2. 'Fuzzy Low': Acepta hasta 35% de error para entender letra Gótica (ej: 'Pcrez' = 'Pérez').
    3. Asignación Múltiple: Si hay dos nombres en la carátula, detectará a ambos.
    """
    if not rekog: return []
    cedulas_encontradas = set()
    
    try:
        with open(imagen_path, 'rb') as image_file:
            image_bytes = image_file.read()
            
        # 1. Detectar TODAS las palabras sueltas (Type='WORD')
        response = rekog.detect_text(Image={'Bytes': image_bytes})
        
        # Juntamos todas las palabras que la IA vio en una sola lista desordenada
        palabras_imagen = [t['DetectedText'].lower() for t in response.get('TextDetections', []) if t['Type'] == 'WORD']
        
        if not palabras_imagen: return []
        
        print(f"📖 Palabras leídas en imagen: {palabras_imagen}") # Debug en logs

        # 2. Traer todos los estudiantes activos
        estudiantes = conn.execute("SELECT Nombre, Apellido, CI FROM Usuarios WHERE Tipo=1").fetchall()
        
        for est in estudiantes:
            # Limpiamos el nombre del estudiante de la base de datos
            nombres = est['Nombre'].lower().split()   # Ej: ["juan", "carlos"]
            apellidos = est['Apellido'].lower().split() # Ej: ["perez", "lopez"]
            
            # --- ESTRATEGIA A: Nombre Completo (Más segura) ---
            # Busca "Juan", "Carlos", "Perez", "Lopez" en cualquier lugar de la hoja
            if coincidencia_difusa(nombres + apellidos, palabras_imagen, umbral=0.80):
                cedulas_encontradas.add(est['CI'])
                continue 
                
            # --- ESTRATEGIA B: Nombre Corto (Más flexible para Gótica) ---
            # Busca solo "Juan" y "Perez". 
            # Usamos umbral 0.65 para tolerar fallos de lectura en letra gótica.
            nombre_corto = [nombres[0], apellidos[0]]
            if coincidencia_difusa(nombre_corto, palabras_imagen, umbral=0.65):
                cedulas_encontradas.add(est['CI'])
                print(f"✅ Coincidencia difusa encontrada para: {est['Nombre']} {est['Apellido']}")

    except Exception as e:
        print(f"⚠️ Error OCR: {e}")
        
    return list(cedulas_encontradas)

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
    """Calcula estadísticas REALES sumando el peso exacto de la base de datos"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Contar usuarios activos
        c.execute("SELECT COUNT(*) FROM Usuarios WHERE Activo = 1")
        usuarios_activos = c.fetchone()[0]
        
        # Contar evidencias
        c.execute("SELECT COUNT(*) FROM Evidencias")
        total_evidencias = c.fetchone()[0]
        
        # CORRECCIÓN: Sumar el peso REAL (Tamanio_KB) de todas las evidencias
        c.execute("SELECT SUM(Tamanio_KB) FROM Evidencias")
        resultado_kb = c.fetchone()[0]
        total_kb = resultado_kb if resultado_kb else 0
        
        # Si la suma es 0 pero hay evidencias (archivos viejos sin peso registrado), usamos estimación
        # Esto corregirá tu problema de 0.17GB vs 1GB conforme subas archivos nuevos o se actualicen
        if total_kb == 0 and total_evidencias > 0:
            total_kb = total_evidencias * 2500 # Estimado 2.5MB solo si no hay datos
            nota_almacenamiento = "Estimado (sube archivos nuevos para corregir)"
        else:
            nota_almacenamiento = "Calculado exacto de DB"

        tamanio_total_mb = total_kb / 1024
        
        # Costos aproximados (Rekognition + S3)
        costo_rekognition = (total_evidencias / 1000) * 1.0
        costo_almacenamiento = (tamanio_total_mb / 1024) * 0.023
        
        # Solicitudes pendientes
        c.execute("SELECT COUNT(*) FROM Solicitudes WHERE Estado = 'PENDIENTE'")
        solicitudes_pendientes = c.fetchone()[0]
        
        conn.close()
        
        return {
            "usuarios_activos": usuarios_activos,
            "total_evidencias": total_evidencias,
            "almacenamiento_mb": round(tamanio_total_mb, 2),
            "almacenamiento_gb": round(tamanio_total_mb / 1024, 4), # 4 decimales para precisión
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
        return {}

# =========================================================================
# 4. CONFIGURACIÓN FASTAPI
# =========================================================================
app = FastAPI(title="Sistema Educativo Despertar", version="7.0")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",                                         # Permite todo (útil para desarrollo)
        "https://proyecto-grado-karlos.vercel.app",  # TU FRONTEND EN VERCEL
        "http://localhost:5500",                     # Por si pruebas en local
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
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
            fecha_registro
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
    """Elimina un usuario y todas sus evidencias asociadas"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Verificar si es admin (opcional, pero seguro)
        c.execute("SELECT Tipo FROM Usuarios WHERE CI = ?", (cedula,))
        user = c.fetchone()
        
        # Opcional: Impedir borrar al superadmin si quisieras
        # if user and user['Tipo'] == 0: ...

        # Borrar evidencias primero (limpieza)
        c.execute("DELETE FROM Evidencias WHERE CI_Estudiante = ?", (cedula,))
        
        # Borrar usuario
        c.execute("DELETE FROM Usuarios WHERE CI = ?", (cedula,))
        
        conn.commit()
        conn.close()
        
        registrar_auditoria("ELIMINACION_USUARIO", f"Usuario {cedula} eliminado permanentemente")
        return JSONResponse({"status": "ok", "mensaje": "Usuario eliminado"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# =========================================================================
# 8. ENDPOINTS DE EVIDENCIAS
# =========================================================================

# --- REEMPLAZA TU FUNCIÓN subir_evidencia_ia POR ESTA VERSIÓN DETALLADA ---

# --- REEMPLAZA TU FUNCIÓN subir_evidencia_ia POR ESTA VERSIÓN FINAL (ROSTROS + VIDEO + TEXTO) ---

@app.post("/subir_evidencia_ia")
async def subir_evidencia_ia(archivo: UploadFile = File(...)):
    try:
        # 1. Preparar archivo
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f: shutil.copyfileobj(archivo.file, f)
        
        ext = os.path.splitext(archivo.filename)[1].lower()
        es_imagen = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        es_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        cedulas_detectadas = set() 
        conn = get_db_connection() # Abrimos conexión aquí para usarla en lectura de texto
        
        # 2. ANÁLISIS INTELIGENTE
        if rekog:
            if es_imagen:
                # A) Buscar Rostros
                rostros = identificar_varios_rostros_aws(path)
                cedulas_detectadas.update(rostros)
                
                # B) NUEVO: Si no hay rostros (o además de ellos), buscar TEXTO
                # Esto sirve para carátulas, listas o fotos de trabajos con nombre
                textos = buscar_estudiantes_por_texto(path, conn)
                cedulas_detectadas.update(textos)
                
            elif es_video:
                # VIDEO: ANÁLISIS POR INTERVALOS
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: fps = 24
                
                intervalo_frames = int(fps * 2) # Cada 2 seg
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                while current_frame < frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_path = os.path.join(temp_dir, f"frame_{current_frame}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Analizar rostros en el video
                    rostros_frame = identificar_varios_rostros_aws(frame_path)
                    cedulas_detectadas.update(rostros_frame)
                    
                    # (Opcional) Podrías leer texto en el video también, pero consume muchos recursos.
                    # Por ahora lo dejamos solo en rostros para video para mantener velocidad.
                    
                    current_frame += intervalo_frames
                    if len(cedulas_detectadas) > 10: break 
                cap.release()

        # 3. Subir archivo original
        url_final = f"/local/{archivo.filename}"
        if s3_client:
            try:
                nube = f"evidencias/{int(ahora_ecuador().timestamp())}_{archivo.filename}"
                ct = 'video/mp4' if es_video else archivo.content_type
                s3_client.upload_file(path, BUCKET_NAME, nube, ExtraArgs={'ACL':'public-read', 'ContentType': ct})
                url_final = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nube}"
            except: pass

        # 4. Guardar en Base de Datos
        status = "alerta"
        msg = ""
        tipo_archivo = "video" if es_video else ("imagen" if es_imagen else "documento")
        
        if cedulas_detectadas:
            nombres = []
            for ced in cedulas_detectadas:
                u = conn.execute("SELECT Nombre, Apellido FROM Usuarios WHERE CI=?", (ced,)).fetchone()
                if u:
                    conn.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Estado, Tipo_Archivo, Asignado_Automaticamente) VALUES (?,?,1,?,1)", (ced, url_final, tipo_archivo))
                    nombres.append(f"{u['Nombre']} {u['Apellido']}")
            
            if nombres:
                status = "exito"
                # Mensaje personalizado según qué encontró
                msg = f"✅ Asignado a: {', '.join(nombres)}"
            else:
                conn.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Estado, Tipo_Archivo, Asignado_Automaticamente) VALUES ('PENDIENTE',?,1,?,0)", (url_final, tipo_archivo))
                msg = "⚠️ Se detectaron datos pero no coinciden con usuarios registrados."
        else:
            conn.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Estado, Tipo_Archivo, Asignado_Automaticamente) VALUES ('PENDIENTE',?,1,?,0)", (url_final, tipo_archivo))
            msg = "⚠️ No se detectaron rostros ni nombres escritos. Guardado en Pendientes."

        conn.commit()
        conn.close()
        shutil.rmtree(temp_dir)
        return JSONResponse({"status": status, "mensaje": msg})

    except Exception as e:
        return JSONResponse({"status": "error", "mensaje": str(e)})
    
@app.post("/subir_manual")
async def subir_manual(
    cedulas: str = Form(...), 
    archivo: UploadFile = File(...), 
    comentario: Optional[str] = Form(None)
):
    try:
        # Procesar lista de cédulas
        lista_cedulas = [c.strip() for c in cedulas.split(",") if c.strip()]
        if not lista_cedulas:
            return JSONResponse(content={"error": "Debe especificar al menos una cédula"})
        
        # Guardar archivo temporalmente
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(archivo.file, f)
            
        # Determinar URL (S3 o Local)
        url_archivo = f"/local/{archivo.filename}"
        if s3_client:
            try:
                nombre_nube = f"evidencias/manual_{int(ahora_ecuador().timestamp())}_{archivo.filename}"
                s3_client.upload_file(path, BUCKET_NAME, nombre_nube, ExtraArgs={'ACL': 'public-read'})
                url_archivo = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
            except: pass
            
        conn = get_db_connection()
        c = conn.cursor()
        
        # Guardar evidencia para cada estudiante
        count = 0
        for ced in lista_cedulas:
            # Verificar si existe el usuario
            if c.execute("SELECT CI FROM Usuarios WHERE CI=?", (ced,)).fetchone():
                c.execute("""
                    INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Estado, Tipo_Archivo, Tamanio_KB, Asignado_Automaticamente)
                    VALUES (?, ?, 1, 'documento', 0, 0)
                """, (ced, url_archivo))
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

@app.post("/optimizar_sistema")
async def optimizar_sistema():
    """Ejecuta tareas de optimización del sistema"""
    try:
        # Optimizar base de datos
        optimizado = optimizar_sistema_db()
        
        # Actualizar estadísticas
        stats = calcular_estadisticas_reales()
        
        # Registrar en métricas
        conn = get_db_connection()
        c = conn.cursor()
        
        fecha_hoy = ahora_ecuador().date().isoformat()
        c.execute("""
            INSERT OR REPLACE INTO Metricas_Sistema 
            (Fecha, Total_Usuarios, Total_Evidencias, Solicitudes_Pendientes, Almacenamiento_MB)
            VALUES (?, ?, ?, ?, ?)
        """, (
            fecha_hoy,
            stats.get("usuarios_activos", 0),
            stats.get("total_evidencias", 0),
            stats.get("solicitudes_pendientes", 0),
            stats.get("almacenamiento_mb", 0)
        ))
        
        conn.commit()
        conn.close()
        
        registrar_auditoria("OPTIMIZACION_SISTEMA", "Sistema optimizado y métricas actualizadas")
        
        return JSONResponse(content={
            "status": "ok",
            "mensaje": "Sistema optimizado correctamente",
            "optimizado": optimizado,
            "estadisticas": stats,
            "fecha": ahora_ecuador().isoformat()
        })
        
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
    
@app.post("/gestionar_solicitud")
async def gestionar_solicitud(
    id_solicitud: int = Form(...),
    accion: str = Form(...),
    mensaje: str = Form(""),
    id_admin: str = Form("Admin")
):
    """Gestiona una solicitud pendiente con envío de email real"""
    try:
        accion_norm = "APROBADA" if accion.upper() in ['APROBAR', 'ACEPTAR', 'APROBADA'] else "RECHAZADA"
        fecha_resolucion = ahora_ecuador()
        
        conn = get_db_connection()
        c = conn.cursor()
        
        # Obtener detalles de la solicitud
        c.execute("""
            SELECT s.*, u.Email, u.Nombre, u.Apellido 
            FROM Solicitudes s
            LEFT JOIN Usuarios u ON s.CI_Solicitante = u.CI
            WHERE s.id = ?
        """, (id_solicitud,))
        
        sol = c.fetchone()
        if not sol:
            conn.close()
            return JSONResponse(content={"error": "Solicitud no encontrada"})
        
        tipo = sol['Tipo']
        email_usuario = sol['Email']
        nombre_usuario = f"{sol['Nombre']} {sol['Apellido']}"
        
        # Procesar según tipo
        if tipo == 'SUBIDA':
            id_evidencia = sol['Id_Evidencia']
            if accion_norm == 'APROBADA':
                c.execute("UPDATE Evidencias SET Estado=1 WHERE id=?", (id_evidencia,))
                mensaje_email = f"Tu evidencia ha sido aprobada por el administrador. {mensaje}"
            else:
                c.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
                mensaje_email = f"Tu evidencia ha sido rechazada. Motivo: {mensaje}"
                
                # Eliminar archivo de S3 si existe
                if sol['Evidencia_Reportada_Url'] and s3_client and BUCKET_NAME in sol['Evidencia_Reportada_Url']:
                    try:
                        key = sol['Evidencia_Reportada_Url'].split(f"{BUCKET_NAME}/")[-1]
                        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                        print(f"✅ Archivo eliminado de S3: {key}")
                    except Exception as e:
                        print(f"⚠️ Error eliminando de S3: {e}")
        
        elif tipo == 'REPORTE':
            id_evidencia = sol['Id_Evidencia']
            if accion_norm == 'APROBADA':
                # Eliminar la evidencia reportada
                c.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
                mensaje_email = f"Tu reporte ha sido procesado. La evidencia ha sido eliminada del sistema. {mensaje}"
                
                # Eliminar archivo de S3
                if sol['Evidencia_Reportada_Url'] and s3_client and BUCKET_NAME in sol['Evidencia_Reportada_Url']:
                    try:
                        key = sol['Evidencia_Reportada_Url'].split(f"{BUCKET_NAME}/")[-1]
                        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                    except:
                        pass
            else:
                mensaje_email = f"Tu reporte ha sido rechazado. Motivo: {mensaje}"
        
        elif tipo == 'RECUPERACION':
            if accion_norm == 'APROBADA':
                # Enviar contraseña temporal o instrucciones
                temp_password = "Temp123!"  # En producción, generar aleatoria
                mensaje_email = f"""
                Hola {nombre_usuario},
                
                Tu solicitud de recuperación de contraseña ha sido aprobada.
                
                Contraseña temporal: {temp_password}
                
                Por favor, cambia tu contraseña después de iniciar sesión.
                
                {mensaje if mensaje else ''}
                
                Atentamente,
                Soporte U.E. Despertar
                """
            else:
                mensaje_email = f"""
                Hola {nombre_usuario},
                
                Tu solicitud de recuperación de contraseña ha sido rechazada.
                
                Motivo: {mensaje if mensaje else 'No cumple con los requisitos de seguridad.'}
                
                Por favor, contacta al administrador para más información.
                
                Atentamente,
                Soporte U.E. Despertar
                """
        
        # Actualizar solicitud
        c.execute("""
            UPDATE Solicitudes 
            SET Estado=?, Resuelto_Por=?, Respuesta=?, Fecha_Resolucion=?
            WHERE id=?
        """, (accion_norm, id_admin, mensaje, fecha_resolucion, id_solicitud))
        
        conn.commit()
        conn.close()
        
        # Enviar correo real al usuario si tiene email
        if email_usuario:
            asunto = f"Respuesta a tu solicitud - U.E. Despertar"
            enviado = enviar_correo_real(email_usuario, asunto, mensaje_email)
            
            if not enviado:
                print(f"⚠️ No se pudo enviar email a {email_usuario}")
        
        # Registrar auditoría
        registrar_auditoria(
            "GESTION_SOLICITUD",
            f"Solicitud {id_solicitud} ({tipo}) {accion_norm.lower()} por {id_admin}"
        )
        
        return JSONResponse(content={
            "status": "ok",
            "mensaje": f"Solicitud {accion_norm.lower()} correctamente",
            "email_enviado": bool(email_usuario),
            "fecha_resolucion": fecha_resolucion.isoformat()
        })
        
    except Exception as e:
        return JSONResponse(content={"status": "error", "mensaje": str(e)})

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
    print(f"🌐 Servidor iniciado en: http://0.0.0.0:{port}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)