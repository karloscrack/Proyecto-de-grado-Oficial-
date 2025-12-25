import shutil
import os
import sqlite3
import logging
import datetime
import zipfile
import hashlib
import boto3
import cv2 
import math
import difflib 
import numpy as np
import tempfile 
from typing import Optional
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from botocore.config import Config
from pydantic import BaseModel

# --- 1. CONFIGURACIÓN Y CREDENCIALES ---
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"
COLLECTION_ID = "estudiantes_db"

# Cliente Rekognition
try:
    rekog = boto3.client('rekognition', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo conectar con AWS Rekognition: {e}")
    rekog = None

# Backblaze B2
ENDPOINT_B2 = "https://s3.us-east-005.backblazeb2.com"
KEY_ID_B2 = "00508884373dab40000000001"
APP_KEY_B2 = "K005jvkLLmLdUKhhVis1qLcnU4flx0g"
BUCKET_NAME = "Proyecto-Grado-Karlos-2025"

try:
    my_config = Config(signature_version='s3v4', region_name='us-east-005')
    s3_client = boto3.client('s3', endpoint_url=ENDPOINT_B2, aws_access_key_id=KEY_ID_B2, aws_secret_access_key=APP_KEY_B2, config=my_config)
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo conectar con Backblaze B2: {e}")
    s3_client = None

# Logging y DB
logging.basicConfig(level=logging.INFO, format='%(message)s')
DB_NAME = "Bases_de_datos.db" 
VOLUMEN_PATH = "/app/datos_persistentes"

# Modelos Pydantic para recibir datos JSON si es necesario
class EstadoUsuarioRequest(BaseModel):
    cedula: str
    activo: int

# --- 2. INICIALIZACIÓN DE BASE DE DATOS ROBUSTA ---
def init_db_completa():
    """Inicializa TODAS las tablas necesarias si no existen"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # 1. Tabla Usuarios
        c.execute('''CREATE TABLE IF NOT EXISTS Usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Nombre TEXT NOT NULL,
            Apellido TEXT NOT NULL,
            CI TEXT UNIQUE NOT NULL,
            Password TEXT NOT NULL,
            Tipo INTEGER DEFAULT 1, -- 0: Admin, 1: Estudiante
            Foto TEXT,
            Activo INTEGER DEFAULT 1,
            TutorialVisto INTEGER DEFAULT 0,
            Fecha_Registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # 2. Tabla Evidencias
        c.execute('''CREATE TABLE IF NOT EXISTS Evidencias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            CI_Estudiante TEXT,
            Url_Archivo TEXT NOT NULL,
            Hash TEXT,
            Estado INTEGER DEFAULT 1, -- 1: Visible, 0: Oculto/Pendiente
            Fecha_Subida TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(CI_Estudiante) REFERENCES Usuarios(CI)
        )''')

        # 3. Tabla Solicitudes
        c.execute('''CREATE TABLE IF NOT EXISTS Solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Tipo TEXT,
            CI_Solicitante TEXT,
            Email TEXT,
            Detalle TEXT,
            Id_Evidencia INTEGER,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Estado TEXT DEFAULT 'PENDIENTE'
        )''')

        # 4. Tabla Auditoria
        c.execute('''CREATE TABLE IF NOT EXISTS Auditoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Accion TEXT,
            Detalle TEXT,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Crear Admin por defecto si no existe
        c.execute("SELECT CI FROM Usuarios WHERE Tipo=0")
        if not c.fetchone():
            print("--> Creando usuario Admin por defecto...")
            c.execute("INSERT INTO Usuarios (Nombre, Apellido, CI, Password, Tipo, Activo) VALUES (?,?,?,?,?,?)", 
                     ('Admin', 'Sistema', '9999999999', 'admin123', 0, 1))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error crítico inicializando DB: {e}")

# Gestión de Volumen (Railway)
if os.path.exists(VOLUMEN_PATH):
    db_en_volumen = os.path.join(VOLUMEN_PATH, "Bases_de_datos.db")
    if not os.path.exists(db_en_volumen):
        print("--> Inicializando Base de Datos en Volumen Persistente...")
        if os.path.exists("Bases_de_datos.db"):
            shutil.copy("Bases_de_datos.db", db_en_volumen)
    DB_NAME = db_en_volumen
    print(f"--> Usando Base de Datos Persistente en: {DB_NAME}")
else:
    print("--> Modo Local o Sin Volumen: Usando DB temporal.")

# ¡EJECUTAR INICIALIZACIÓN!
init_db_completa()

# --- VERIFICACIÓN DE REKOGNITION ---
def init_rekognition_collection():
    if rekog:
        try:
            print(f"--> Verificando colección Rekognition: {COLLECTION_ID}")
            rekog.create_collection(CollectionId=COLLECTION_ID)
            print(f"✅ Colección {COLLECTION_ID} creada correctamente.")
        except rekog.exceptions.ResourceAlreadyExistsException:
            print(f"✅ La colección {COLLECTION_ID} ya existe.")
        except Exception as e:
            print(f"⚠️ Error verificando colección: {e}")

init_rekognition_collection()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. FUNCIONES AUXILIARES ---
def registrar_auditoria(accion, detalle):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO Auditoria (Accion, Detalle) VALUES (?, ?)", (accion, detalle))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Error auditoría: {e}")

def eliminar_archivo_nube(url_archivo):
    """Borra el archivo físico de Backblaze para no dejar basura"""
    try:
        if not url_archivo or not s3_client: return
        nombre_clave = url_archivo.split(f"{BUCKET_NAME}/")[-1] 
        if "evidencias/" in url_archivo and "http" in url_archivo:
             nombre_clave = "evidencias/" + url_archivo.split("evidencias/")[-1]
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=nombre_clave)
        print(f"🗑️ Archivo eliminado de nube: {nombre_clave}")
    except Exception as e:
        print(f"⚠️ Error borrando de nube: {e}")

def calcular_hash(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(4096), b""): h.update(b)
    return h.hexdigest()

def es_parecido(palabra1, palabra2):
    return difflib.SequenceMatcher(None, palabra1, palabra2).ratio() >= 0.70

def forzar_compresion(ruta_archivo):
    try:
        Img = cv2.imread(ruta_archivo)
        if Img is None: return None
        calidad = 85; scale = 1.0
        while True:
            if scale < 1.0:
                h, w = Img.shape[:2]
                Img_temp = cv2.resize(Img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else: Img_temp = Img
            exito, buffer = cv2.imencode('.jpg', Img_temp, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])
            byte_data = buffer.tobytes()
            if (len(byte_data) / (1024 * 1024)) < 4.0: return byte_data
            calidad -= 10; scale -= 0.1
            if calidad < 30: return byte_data
    except Exception as e: print(f"Error comprimiendo: {e}"); return None

def procesar_foto_grupal(Img_bytes):
    encontrados = set()
    if not rekog: return encontrados
    try:
        # 1. Detectar todas las caras en la foto
        resp = rekog.detect_faces(Image={'Bytes': Img_bytes}, Attributes=['DEFAULT'])
        detalles = resp['FaceDetails']
        
        if not detalles: return encontrados
        
        # Preparamos la imagen para recortar las caritas
        nparr = np.frombuffer(Img_bytes, np.uint8)
        Img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig = Img_cv.shape[:2]
        
        # 2. Iterar sobre CADA cara detectada
        for idx, cara in enumerate(detalles):
            box = cara['BoundingBox']
            left = int(box['Left'] * w_orig); top = int(box['Top'] * h_orig)
            width = int(box['Width'] * w_orig); height = int(box['Height'] * h_orig)
            
            # Ajustes de límites para no salir de la imagen
            left = max(0, left); top = max(0, top)
            if left + width > w_orig: width = w_orig - left
            if top + height > h_orig: height = h_orig - top
            
            # Recortar la cara
            cara_recortada = Img_cv[top:top+height, left:left+width]
            _, buffer_cara = cv2.imencode('.jpg', cara_recortada)
            
            try:
                # 3. Buscar quién es esta cara específica con UMBRAL ALTO (95%)
                search_res = rekog.search_faces_by_image(
                    CollectionId=COLLECTION_ID, 
                    Image={'Bytes': buffer_cara.tobytes()}, 
                    FaceMatchThreshold=95 
                )
                
                matches = search_res['FaceMatches']
                # Si hay coincidencia, guardar al estudiante
                if matches:
                    persona = matches[0]['Face']['ExternalImageId']
                    encontrados.add(persona)
            except Exception as e:
                pass # Ignorar caras no identificadas o errores puntuales
                
    except Exception as e: print(f"❌ Error procesando grupo: {e}")
    return encontrados

def buscar_estudiantes_texto_smart(texto):
    encontrados = []
    if not texto: return []
    palabras = texto.upper().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U').replace(',','').replace('.','').split()
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT CI, Nombre, Apellido FROM Usuarios WHERE Tipo=1")
    users = c.fetchall(); conn.close()
    for ci, nom, ape in users:
        n_real = nom.upper().split()[0]; a_real = ape.upper().split()[0]
        match_n = False; match_a = False
        for p in palabras:
            if es_parecido(p, n_real): match_n = True
            if es_parecido(p, a_real): match_a = True
        if match_n and match_a: encontrados.append(ci)
    return encontrados

def borrar_ruta(path: str):
    try: 
        if os.path.isfile(path): os.remove(path)
        elif os.path.isdir(path): shutil.rmtree(path)
    except: pass

# --- 4. ENDPOINTS BÁSICOS ---
@app.post("/registrar_usuario")
async def registrar_usuario(nombre: str=Form(...), apellido: str=Form(...), cedula: str=Form(...), contrasena: str=Form(...), tipo_usuario: int=Form(...), foto: UploadFile=UploadFile(...)):
    try:
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("SELECT CI FROM Usuarios WHERE CI=?", (cedula,))
        if c.fetchone(): conn.close(); raise HTTPException(400, "Usuario ya existe")
        
        folder_local = f"perfiles_db/{cedula}"
        if not os.path.exists(folder_local): os.makedirs(folder_local)
        ext = foto.filename.split('.')[-1]
        fname = f"{folder_local}/principal.{ext}"
        
        with open(fname, "wb") as f: shutil.copyfileobj(foto.file, f)
        bytes_Img = forzar_compresion(fname)
        if not bytes_Img: 
            with open(fname, 'rb') as f: bytes_Img = f.read()

        if rekog:
            try: rekog.index_faces(CollectionId=COLLECTION_ID, Image={'Bytes': bytes_Img}, ExternalImageId=cedula, DetectionAttributes=['ALL'], QualityFilter='AUTO')
            except Exception as aws_err: os.remove(fname); raise HTTPException(400, f"Error AWS: {aws_err}")

        # Subir a S3/Backblaze
        s3_path = f"perfiles/perfil_{cedula}.{ext}"
        if s3_client:
            s3_client.upload_file(fname, BUCKET_NAME, s3_path)
            url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{s3_path}"
        else:
            url = ""
        
        c.execute("INSERT INTO Usuarios (Nombre,Apellido,CI,Password,Tipo,Foto,Activo) VALUES (?,?,?,?,?,?,1)", (nombre,apellido,cedula,contrasena,tipo_usuario,url))
        conn.commit(); conn.close()
        registrar_auditoria("REGISTRO", f"Usuario nuevo: {nombre} {apellido} ({cedula})")
        return {"mensaje":"Registrado correctamente", "url":url}
    except Exception as e: return {"error": str(e)}

@app.post("/agregar_referencia_facial")
async def agregar_referencia(cedula: str = Form(...), foto: UploadFile = UploadFile(...)):
    try:
        if not rekog: return {"error": "Servicio de reconocimiento facial no disponible"}
        folder_local = f"perfiles_db/{cedula}"
        if not os.path.exists(folder_local): os.makedirs(folder_local)
        fname = f"{folder_local}/ref_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        with open(fname, "wb") as f: shutil.copyfileobj(foto.file, f)
        bytes_Img = forzar_compresion(fname)
        if not bytes_Img: 
            with open(fname, 'rb') as f: bytes_Img = f.read()
        try:
            rekog.index_faces(CollectionId=COLLECTION_ID, Image={'Bytes': bytes_Img}, ExternalImageId=cedula, QualityFilter='AUTO')
            registrar_auditoria("ENTRENAMIENTO", f"Nueva foto de referencia para {cedula}")
            return {"mensaje": "Referencia agregada."}
        except Exception as e: return {"error": f"Error AWS: {e}"}
    except Exception as e: return {"error": str(e)}

@app.post("/iniciar_sesion")
async def iniciar_sesion(cedula: str = Form(...), contrasena: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Tipo, Nombre, Apellido, Activo FROM Usuarios WHERE CI=? AND Password=?", (cedula, contrasena))
    u = c.fetchone(); conn.close()
    if u: 
        if u[3] == 0: return {"encontrado": False, "mensaje": "Cuenta desactivada"}
        registrar_auditoria("LOGIN", f"Inicio de sesión: {u[1]} {u[2]} ({cedula})")
        return {"encontrado": True, "datos": {"tipo": u[0], "nombre": u[1], "apellido": u[2]}}
    return {"encontrado": False, "mensaje": "Credenciales incorrectas"}

@app.post("/marcar_tutorial_visto")
async def marcar_tutorial(cedula: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        c.execute("UPDATE Usuarios SET TutorialVisto=1 WHERE CI=?", (cedula,))
        conn.commit()
    except: pass
    conn.close()
    return {"status": "ok"}

# --- 5. ENDPOINTS DE SUBIDA Y GESTIÓN DE ARCHIVOS ---
@app.post("/subir_manual")
async def subir_manual(archivo: UploadFile = UploadFile(...), cedulas: str = Form(...)):
    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, f"manual_{archivo.filename.replace(' ','_')}")
    lista_cedulas = [c.strip() for c in cedulas.split(',')]
    try:
        with open(fname, "wb") as f: shutil.copyfileobj(archivo.file, f)
        fhash = calcular_hash(fname)
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("SELECT id FROM Evidencias WHERE Hash=?", (fhash,))
        if c.fetchone():
            conn.close(); borrar_ruta(temp_dir)
            return {"status": "alerta", "motivo": "duplicado", "mensaje": "Archivo ya existe."}
        
        nombre_limpio = os.path.basename(fname)
        s3_client.upload_file(fname, BUCKET_NAME, f"evidencias/{nombre_limpio}")
        url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/evidencias/{nombre_limpio}"
        
        nombres_asignados = []
        for ci in lista_cedulas:
            c.execute("SELECT id, Nombre, Apellido FROM Usuarios WHERE CI=?", (ci,))
            u = c.fetchone()
            if u:
                c.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado) VALUES (?,?,?,1)", (ci, url, fhash))
                nombres_asignados.append(f"{u[1]} {u[2]}")
        conn.commit(); conn.close(); borrar_ruta(temp_dir)
        if nombres_asignados:
            registrar_auditoria("SUBIDA MANUAL", f"Archivo: {archivo.filename} -> {', '.join(nombres_asignados)}")
            return {"mensaje": "OK", "asignado_a": nombres_asignados}
        else: return {"error": "No se encontraron los usuarios"}
    except Exception as e: borrar_ruta(temp_dir); return {"error": str(e)}

@app.post("/subir_evidencia")
async def subir_evidencia(cedula_destino: str = Form(None), archivo: UploadFile = UploadFile(...)):
    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, f"ev_{archivo.filename.replace(' ','_')}")
    dest = set()
    estado = "sin_coincidencia"; motivo = ""
    ext = fname.split('.')[-1].lower()
    
    try:
        valid_exts = ['mp4','mov','avi','webm','jpg','jpeg','png','pdf','docx','xlsx','pptx','txt']
        if ext not in valid_exts:
            borrar_ruta(temp_dir); return {"status": "error", "motivo": "Formato no soportado"}
        
        with open(fname,"wb") as f: shutil.copyfileobj(archivo.file,f)
        fhash = calcular_hash(fname)
        
        conn=sqlite3.connect(DB_NAME); c=conn.cursor()
        c.execute("SELECT id FROM Evidencias WHERE Hash=?",(fhash,))
        if c.fetchone(): 
            conn.close(); borrar_ruta(temp_dir)
            return {"status": "alerta", "motivo": "duplicado", "mensaje": "Archivo repetido"}
        conn.close()

        # --- LÓGICA DE DETECCIÓN MEJORADA ---
        if rekog and ext in ['jpg','jpeg','png']:
            Img_bytes = forzar_compresion(fname)
            if Img_bytes:
                personas = procesar_foto_grupal(Img_bytes)
                if personas: dest.update(personas)
                try:
                    resp_txt = rekog.detect_text(Image={'Bytes': Img_bytes})
                    txt = " ".join([t['DetectedText'] for t in resp_txt['TextDetections'] if t['Type'] == 'LINE'])
                    if txt: dest.update(buscar_estudiantes_texto_smart(txt))
                except: pass

        elif rekog and ext in ['mp4','mov','avi','webm']:
            try:
                cam = cv2.VideoCapture(fname)
                fps = cam.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_step = int(fps / 2) 
                if frame_step < 1: frame_step = 1
                curr = 0; scans = 0
                while curr < total_frames and scans < 60:
                    cam.set(cv2.CAP_PROP_POS_FRAMES, curr)
                    ret, frame = cam.read()
                    if not ret: break
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    try:
                        resp = rekog.search_faces_by_image(
                            CollectionId=COLLECTION_ID, 
                            Image={'Bytes': buffer.tobytes()}, 
                            FaceMatchThreshold=92 
                        )
                        for m in resp['FaceMatches']: dest.add(m['Face']['ExternalImageId'])
                    except: pass
                    curr += frame_step; scans += 1
                cam.release()
            except Exception as e: print(f"Error analizando video: {e}")

        # Subir a Nube y Guardar en BD
        nombre_limpio = os.path.basename(fname)
        if s3_client:
            s3_client.upload_file(fname, BUCKET_NAME, f"evidencias/{nombre_limpio}")
            url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/evidencias/{nombre_limpio}"
        else:
            url = f"/datos_persistentes/local_{nombre_limpio}" # Fallback local
        
        conn=sqlite3.connect(DB_NAME); c=conn.cursor()
        if dest:
            estado = "exito"
            nombres = []
            for ci in dest:
                c.execute("SELECT id, Nombre, Apellido FROM Usuarios WHERE CI=?", (ci,))
                ud = c.fetchone()
                if ud:
                    c.execute("INSERT INTO Evidencias (CI_Estudiante,Url_Archivo,Hash,Estado) VALUES (?,?,?,1)",(ci,url,fhash))
                    nombres.append(f"{ud[1]} {ud[2]}")
            registrar_auditoria("SUBIDA EXITOSA", f"IA detectó en {archivo.filename}: {', '.join(nombres)}")
        else:
            motivo = "Sin coincidencias claras (Umbral 95%)."
            registrar_auditoria("SUBIDA SIN ASIGNAR", f"Archivo: {archivo.filename}")
            
        conn.commit(); conn.close(); borrar_ruta(temp_dir)
        return {"status": estado, "asignado_a": list(dest), "motivo": motivo, "url": url}
    except Exception as e: borrar_ruta(temp_dir); return {"status": "error", "motivo": str(e)}

# --- 6. ENDPOINTS ESTUDIANTES (CORREGIDOS) ---
@app.post("/solicitar_recuperacion")
async def solicitar_recuperacion(cedula: str = Form(None), email: str = Form(...), mensaje: str = Form(None)):
    try:
        # Intenta usar email como cédula si no viene la cédula (para compatibilidad con frontend)
        ci_final = cedula if cedula else (email.split('@')[0] if '@' in email else email)
        
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("SELECT Nombre FROM Usuarios WHERE CI=?", (ci_final,))
        if not c.fetchone():
            # Permitir solicitud aunque no exista usuario (seguridad)
            pass
        
        hora = (datetime.datetime.now() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Email, Detalle, Estado, Fecha) VALUES (?,?,?,?,?,?)",
                  ("RECUPERACION", ci_final, email, mensaje or "Olvidé mi clave", "PENDIENTE", hora))
        conn.commit(); conn.close()
        registrar_auditoria("SOLICITUD CLAVE", f"Solicitud recuperación para {ci_final}")
        return {"status": "ok", "mensaje": "Solicitud enviada a los administradores."}
    except Exception as e: return {"status": "error", "mensaje": str(e)}

@app.post("/reportar_evidencia")
async def reportar_evidencia(cedula: str = Form(...), id_evidencia: int = Form(...), motivo: str = Form(...)):
    try:
        hora = (datetime.datetime.now() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Id_Evidencia, Detalle, Estado, Fecha) VALUES (?,?,?,?,?,?)",
                  ("REPORTE", cedula, id_evidencia, motivo, "PENDIENTE", hora))
        conn.commit(); conn.close()
        return {"status": "ok", "mensaje": "Reporte enviado."}
    except Exception as e: return {"status": "error", "mensaje": str(e)}

@app.post("/solicitar_subida")
async def solicitar_subida(cedula: str = Form(...), archivo: UploadFile = UploadFile(...)):
    temp_dir = tempfile.mkdtemp()
    fname = os.path.join(temp_dir, f"sol_{cedula}_{archivo.filename}")
    try:
        with open(fname, "wb") as f: shutil.copyfileobj(archivo.file, f)
        
        nombre_nube = os.path.basename(fname)
        if s3_client:
            s3_client.upload_file(fname, BUCKET_NAME, f"evidencias/{nombre_nube}")
            url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/evidencias/{nombre_nube}"
        else: url = f"/datos_persistentes/{nombre_nube}"
        fhash = calcular_hash(fname)
        
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado) VALUES (?,?,?,0)", (cedula, url, fhash))
        id_evidencia = c.lastrowid
        
        hora = (datetime.datetime.now() - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Id_Evidencia, Detalle, Estado, Fecha) VALUES (?,?,?,?,?,?)",
                  ("SUBIDA", cedula, id_evidencia, f"Solicitud de subida: {archivo.filename}", "PENDIENTE", hora))
        
        conn.commit(); conn.close(); shutil.rmtree(temp_dir)
        return {"status": "ok", "mensaje": "Archivo enviado a aprobación."}
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {"status": "error", "mensaje": str(e)}

# --- 7. ENDPOINTS ADMIN Y GESTIÓN (CORREGIDOS) ---
@app.get("/obtener_solicitudes")
async def obtener_solicitudes():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    query = """
        SELECT s.id, s.Tipo, s.CI_Solicitante, s.Detalle, s.Id_Evidencia, s.Fecha, 
               IFNULL(u.Nombre, 'Usuario'), IFNULL(u.Apellido, 'Desconocido'), e.Url_Archivo, s.Email
        FROM Solicitudes s
        LEFT JOIN Usuarios u ON s.CI_Solicitante = u.CI
        LEFT JOIN Evidencias e ON s.Id_Evidencia = e.id
        WHERE s.Estado = 'PENDIENTE'
        ORDER BY s.id DESC
    """
    c.execute(query)
    data = []
    for row in c.fetchall():
        data.append({
            "id": row[0], "Tipo": row[1], "cedula": row[2], 
            "Detalle": row[3], "id_evidencia": row[4], "fecha": row[5],
            "nombre": f"{row[6]} {row[7]}",
            "evidencia_url": row[8],
            "email": row[9]
        })
    conn.close()
    return data

@app.post("/responder_solicitud")
async def responder_solicitud_json(datos: dict):
    # Wrapper para compatibilidad con JSON body
    return await gestionar_solicitud(id_solicitud=datos.get('id'), accion=datos.get('estado'), id_admin=datos.get('resuelto_por'))

@app.post("/gestionar_solicitud")
async def gestionar_solicitud(id_solicitud: int = Form(...), accion: str = Form(...), id_admin: str = Form("Admin")):
    # Normalizar acción
    accion_normalizada = "APROBADA" if accion in ['APROBAR', 'aceptar', 'APROBADA'] else "RECHAZADA"
    
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Tipo, Id_Evidencia FROM Solicitudes WHERE id=?", (id_solicitud,))
    sol = c.fetchone()
    
    if not sol: conn.close(); return {"error": "Solicitud no encontrada"}
    tipo, id_evidencia = sol
    
    url_archivo = None
    if id_evidencia:
        c.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_evidencia,))
        row_ev = c.fetchone()
        if row_ev: url_archivo = row_ev[0]

    if accion_normalizada == 'APROBADA':
        if tipo == 'SUBIDA':
            c.execute("UPDATE Evidencias SET Estado=1 WHERE id=?", (id_evidencia,))
            registrar_auditoria("APROBACION", f"Admin {id_admin} aprobó subida {id_evidencia}")
        elif tipo == 'REPORTE':
            if url_archivo: eliminar_archivo_nube(url_archivo)
            c.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
            registrar_auditoria("APROBACION", f"Admin {id_admin} aceptó reporte y borró {id_evidencia}")

    elif accion_normalizada == 'RECHAZADA':
        if tipo == 'SUBIDA':
            if url_archivo: eliminar_archivo_nube(url_archivo)
            c.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
            registrar_auditoria("RECHAZO", f"Admin {id_admin} rechazó subida {id_evidencia}")
        elif tipo == 'REPORTE':
            registrar_auditoria("RECHAZO", f"Admin {id_admin} desestimó reporte {id_evidencia}")

    c.execute("UPDATE Solicitudes SET Estado=? WHERE id=?", (accion_normalizada, id_solicitud))
    conn.commit(); conn.close()
    return {"status": "ok", "mensaje": f"Solicitud procesada correctamente."}

@app.delete("/borrar_evidencia/{id_ev}")
async def del_ev(id_ev: int):
    return await eliminar_evidencia(id_ev)

@app.delete("/eliminar_evidencia/{id_ev}")
async def eliminar_evidencia(id_ev: int):
    conn = sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_ev,))
    ev = c.fetchone()
    if ev:
        eliminar_archivo_nube(ev[0]) 
        nom = ev[0].split('/')[-1]
    else: nom = "?"
    c.execute("DELETE FROM Evidencias WHERE id=?",(id_ev,))
    conn.commit(); conn.close()
    registrar_auditoria("ELIMINACIÓN", f"Evidencia borrada manualmente: {nom}")
    return {"status": "ok", "msg":"Eliminada"}

# --- 8. ENDPOINTS CONSULTA Y UTILIDADES ---
@app.get("/todas_evidencias")
async def todas(cedula: str = None):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    if cedula and cedula.strip():
        c.execute("SELECT id, Url_Archivo, CI_Estudiante, Tipo_Archivo FROM Evidencias WHERE CI_Estudiante=? AND Estado=1 ORDER BY id DESC", (cedula,))
    else:
        # Añadimos Tipo_Archivo para que el frontend no falle
        try:
             c.execute("SELECT id, Url_Archivo, CI_Estudiante, Tipo_Archivo FROM Evidencias WHERE Estado=1 ORDER BY id DESC LIMIT 50")
        except:
             # Fallback si no existe la columna Tipo_Archivo
             c.execute("SELECT id, Url_Archivo, CI_Estudiante FROM Evidencias WHERE Estado=1 ORDER BY id DESC LIMIT 50")
             
    # Adaptamos la respuesta
    res = []
    for x in c.fetchall():
        tipo = "documento"
        if len(x) > 3 and x[3]: tipo = x[3] # Si la columna existe
        res.append({"id":x[0], "url":x[1], "ci":x[2], "tipo": tipo})
        
    conn.close(); return res

@app.delete("/eliminar_usuario/{cedula}")
async def elim_user(cedula: str):
    conn = sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("DELETE FROM Usuarios WHERE CI=?",(cedula,))
    c.execute("DELETE FROM Evidencias WHERE CI_Estudiante=?",(cedula,))
    conn.commit(); conn.close()
    if os.path.exists(f"perfiles_db/{cedula}"): shutil.rmtree(f"perfiles_db/{cedula}")
    try: rekog.delete_faces(CollectionId=COLLECTION_ID, FaceIds=[cedula])
    except: pass
    registrar_auditoria("BAJA USUARIO", f"Usuario eliminado: {cedula}")
    return {"status": "ok", "msg":"Borrado"}

@app.post("/buscar_estudiante")
async def buscar_estudiante(cedula: str = Form(...), contrasena: Optional[str] = Form(None)):
    """
    Endpoint crucial para perfil.html y Login corregido.
    Devuelve datos del usuario, galería y TIPO (Admin/Estudiante).
    """
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; c = conn.cursor()
    
    # 1. Buscar Usuario
    c.execute("SELECT * FROM Usuarios WHERE CI=?", (cedula,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return {"encontrado": False, "mensaje": "Usuario no encontrado"}
    
    # Validar pass si se envía
    if contrasena and user["Password"] != contrasena:
        conn.close()
        return {"encontrado": False, "mensaje": "Contraseña incorrecta"}
        
    # 2. Buscar Galería
    c.execute("SELECT id, Url_Archivo as url FROM Evidencias WHERE CI_Estudiante=? AND Estado=1", (cedula,))
    evs = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Devolvemos TODO lo que necesitan los htmls
    return {
        "encontrado": True,
        "datos": {
            "nombre": user["Nombre"],
            "apellido": user["Apellido"],
            "cedula": user["CI"],
            "tipo": user["Tipo"], # <--- CRÍTICO PARA LOGIN
            "url_foto": user["Foto"],
            "galeria": evs
        }
    }

@app.get("/crear_backup")
async def backup(background_tasks: BackgroundTasks):
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"Backup_DB_{datetime.datetime.now().strftime('%Y%m%d')}.zip")
    with zipfile.ZipFile(zip_path,'w') as z:
        if os.path.exists(DB_NAME): z.write(DB_NAME, os.path.basename(DB_NAME))
    registrar_auditoria("BACKUP", "Copia de Base de Datos descargada")
    background_tasks.add_task(borrar_ruta, temp_dir)
    return FileResponse(zip_path, media_type="application/zip", filename=os.path.basename(zip_path))

@app.get("/descargar_multimedia_zip")
async def descargar_multimedia_zip(background_tasks: BackgroundTasks):
    temp_work_dir = tempfile.mkdtemp()
    zip_name = f"Multimedia_Full_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    zip_full_path = os.path.join(temp_work_dir, zip_name)
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT DISTINCT Url_Archivo FROM Evidencias")
    urls = c.fetchall(); conn.close()
    with zipfile.ZipFile(zip_full_path, 'w') as z:
        for row in urls:
            url = row[0]; filename = url.split('/')[-1]
            local_path = os.path.join(temp_work_dir, filename)
            try:
                if s3_client:
                    s3_client.download_file(BUCKET_NAME, f"evidencias/{filename}", local_path)
                    z.write(local_path, filename)
            except: pass
    registrar_auditoria("BACKUP FULL", f"Descarga multimedia")
    background_tasks.add_task(borrar_ruta, temp_work_dir)
    return FileResponse(zip_full_path, media_type="application/zip", filename=zip_name)

@app.get("/resumen_estudiantes_con_evidencias")
async def resumen_estudiantes():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    query = "SELECT u.Nombre, u.Apellido, u.CI, u.Foto, COUNT(e.id) as total FROM Usuarios u LEFT JOIN Evidencias e ON u.CI = e.CI_Estudiante AND e.Estado=1 WHERE u.Tipo=1 GROUP BY u.CI ORDER BY u.Apellido"
    c.execute(query)
    res = [{"nombre": x[0], "apellido": x[1], "cedula": x[2], "foto": x[3], "total_evidencias": x[4]} for x in c.fetchall()]
    conn.close(); return res

@app.post("/eliminar_evidencias_masivas")
async def eliminar_masivo(ids: str = Form(...)):
    lista_ids = ids.split(',')
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    borrados = 0
    try:
        for id_ev in lista_ids:
            c.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_ev,))
            row = c.fetchone()
            if row:
                eliminar_archivo_nube(row[0]) # Borrado nube
                c.execute("DELETE FROM Evidencias WHERE id=?", (id_ev,))
                borrados += 1
        conn.commit(); registrar_auditoria("ELIMINACION MASIVA", f"Se borraron {borrados} archivos.")
    except Exception as e: conn.close(); return {"status": "error", "mensaje": str(e)}
    conn.close(); return {"status": "ok", "borrados": borrados}

@app.post("/descargar_seleccion_zip")
async def descargar_seleccion_zip(ids: str = Form(...), background_tasks: BackgroundTasks = None):
    lista_ids = ids.split(',')
    if not lista_ids: raise HTTPException(400, "No hay archivos seleccionados")
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    placeholders = ','.join('?' for _ in lista_ids)
    query = f"SELECT Url_Archivo FROM Evidencias WHERE id IN ({placeholders})"
    c.execute(query, lista_ids)
    resultados = c.fetchall(); conn.close()
    if not resultados: raise HTTPException(404, "Archivos no encontrados")

    temp_dir = tempfile.mkdtemp()
    zip_name = f"Evidencias_Seleccion_{datetime.datetime.now().strftime('%H%M%S')}.zip"
    zip_path = os.path.join(temp_dir, zip_name)
    try:
        with zipfile.ZipFile(zip_path, 'w') as z:
            for row in resultados:
                url = row[0]; filename = url.split('/')[-1]
                local_path = os.path.join(temp_dir, filename)
                try:
                    if s3_client:
                        s3_client.download_file(BUCKET_NAME, f"evidencias/{filename}", local_path)
                        z.write(local_path, filename)
                except: pass
        return FileResponse(zip_path, media_type="application/zip", filename=zip_name)
    except Exception as e: return {"error": str(e)}
    finally:
        if background_tasks: background_tasks.add_task(shutil.rmtree, temp_dir)

@app.get("/listar_usuarios")
async def listar():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Nombre, Apellido, CI, Tipo, Password, Activo, Foto FROM Usuarios")
    res = [{"nombre":x[0], "apellido":x[1], "cedula":x[2], "tipo":x[3], "contrasena":x[4], "activo":x[5], "foto":x[6]} for x in c.fetchall()]
    conn.close(); return res

@app.post("/cambiar_estado_usuario")
async def cambiar_estado_usuario(datos: EstadoUsuarioRequest):
    """Necesario para el panel admin (Activar/Desactivar)"""
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute("UPDATE Usuarios SET Activo = ? WHERE CI = ?", (datos.activo, datos.cedula))
        conn.commit()
        return {"status": "ok", "mensaje": "Estado actualizado"}
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}
    finally:
        conn.close()

@app.get("/estadisticas_almacenamiento")
async def estadisticas_almacenamiento():
    """Necesario para el Dashboard del Admin"""
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM Usuarios WHERE Activo = 1")
    usuarios = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM Evidencias WHERE Estado=1")
    evidencias = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM Solicitudes WHERE Estado = 'PENDIENTE'")
    solicitudes = c.fetchone()[0]
    
    # Cálculo aproximado (3MB por evidencia)
    size_gb = (evidencias * 3.0) / 1024
    
    conn.close()
    
    return {
        "usuarios_activos": usuarios,
        "total_evidencias": evidencias,
        "solicitudes_pendientes": solicitudes,
        "almacenamiento_gb": size_gb,
        "imagenes_procesadas_mes": evidencias # Dato estimado
    }

@app.get("/obtener_logs")
async def obtener_logs():
    conn=sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("SELECT * FROM Auditoria ORDER BY id DESC LIMIT 50")
    l=[{"fecha":x[3],"accion":x[1],"detalle":x[2]} for x in c.fetchall()]
    conn.close(); return l

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"--> Servidor FINAL INICIADO en puerto {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)