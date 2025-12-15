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
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from botocore.config import Config

# --- CREDENCIALES ---
# CORREGIDO: Ya no tiene las claves escritas. Las tomará de Railway.
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"
COLLECTION_ID = "estudiantes_db"

# Cliente Rekognition
rekog = boto3.client('rekognition', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# --- BACKBLAZE ---
ENDPOINT_B2 = "https://s3.us-east-005.backblazeb2.com"
KEY_ID_B2 = "00508884373dab40000000001"
APP_KEY_B2 = "K005jvkLLmLdUKhhVis1qLcnU4flx0g"
BUCKET_NAME = "Proyecto-Grado-Karlos-2025"
my_config = Config(signature_version='s3v4', region_name='us-east-005')
s3_client = boto3.client('s3', endpoint_url=ENDPOINT_B2, aws_access_key_id=KEY_ID_B2, aws_secret_access_key=APP_KEY_B2, config=my_config)

# --- INICIALIZACIÓN ---
logging.basicConfig(level=logging.INFO, format='%(message)s')

# IMPORTANTE: Busca la DB en la misma carpeta que main.py
DB_NAME = "Bases_de_datos.db" 
TEMP_DIR = "temp_files" 
VOLUMEN_PATH = "/app/datos_persistentes"

# Verificamos si estamos en Railway (si existe la carpeta segura)
if os.path.exists(VOLUMEN_PATH):
    db_en_volumen = os.path.join(VOLUMEN_PATH, "Bases_de_datos.db")
    
    # Si la base de datos NO está en el volumen, copiamos la que subiste a GitHub
    if not os.path.exists(db_en_volumen):
        print("--> Inicializando Base de Datos en Volumen Persistente...")
        shutil.copy("Bases_de_datos.db", db_en_volumen)
    
    # Le decimos al sistema que use la base de datos del volumen
    DB_NAME = db_en_volumen
    print(f"--> Usando Base de Datos Persistente en: {DB_NAME}")
else:
    print("--> Modo Local o Sin Volumen: Usando DB temporal.")
app = FastAPI()

# --- CONFIGURACIÓN CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIONES AUXILIARES ---
def registrar_auditoria(accion, detalle):
    try:
        hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("INSERT INTO Auditoria (Accion, Detalle, Fecha) VALUES (?,?,?)", (accion, detalle, hora))
        conn.commit(); conn.close()
    except Exception as e: print(f"⚠️ Error guardando log: {e}")

def calcular_hash(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(4096), b""): h.update(b)
    return h.hexdigest()

def es_parecido(palabra1, palabra2):
    return difflib.SequenceMatcher(None, palabra1, palabra2).ratio() >= 0.70

def forzar_compresion(ruta_archivo):
    try:
        img = cv2.imread(ruta_archivo)
        if img is None: return None
        calidad = 85; scale = 1.0
        while True:
            if scale < 1.0:
                h, w = img.shape[:2]
                img_temp = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else: img_temp = img
            exito, buffer = cv2.imencode('.jpg', img_temp, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])
            byte_data = buffer.tobytes()
            peso_mb = len(byte_data) / (1024 * 1024)
            if peso_mb < 4.0: return byte_data
            calidad -= 10; scale -= 0.1
            if calidad < 30: return byte_data
    except Exception as e: print(f"Error comprimiendo: {e}"); return None

def procesar_foto_grupal(img_bytes):
    encontrados = set()
    try:
        resp = rekog.detect_faces(Image={'Bytes': img_bytes}, Attributes=['DEFAULT'])
        detalles = resp['FaceDetails']
        if not detalles: return encontrados
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig = img_cv.shape[:2]
        for idx, cara in enumerate(detalles):
            box = cara['BoundingBox']
            left = int(box['Left'] * w_orig); top = int(box['Top'] * h_orig)
            width = int(box['Width'] * w_orig); height = int(box['Height'] * h_orig)
            left = max(0, left); top = max(0, top)
            if left + width > w_orig: width = w_orig - left
            if top + height > h_orig: height = h_orig - top
            cara_recortada = img_cv[top:top+height, left:left+width]
            _, buffer_cara = cv2.imencode('.jpg', cara_recortada)
            try:
                search_res = rekog.search_faces_by_image(CollectionId=COLLECTION_ID, Image={'Bytes': buffer_cara.tobytes()}, FaceMatchThreshold=40)
                matches = search_res['FaceMatches']
                if matches:
                    persona = matches[0]['Face']['ExternalImageId']
                    encontrados.add(persona)
            except: pass
    except Exception as e: print(f"   ❌ Error grupo: {e}")
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
        if match_n and match_a:
            encontrados.append(ci)
    return encontrados

def borrar_ruta(path: str):
    try: 
        if os.path.isfile(path): os.remove(path)
        elif os.path.isdir(path): shutil.rmtree(path)
    except: pass

# --- ENDPOINTS ---
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
        
        bytes_img = forzar_compresion(fname)
        if not bytes_img: 
            with open(fname, 'rb') as f: bytes_img = f.read()

        try: rekog.index_faces(CollectionId=COLLECTION_ID, Image={'Bytes': bytes_img}, ExternalImageId=cedula, DetectionAttributes=['ALL'], QualityFilter='AUTO')
        except Exception as aws_err: os.remove(fname); raise HTTPException(400, f"Error AWS: {aws_err}")

        s3_path = f"perfiles/perfil_{cedula}.{ext}"
        s3_client.upload_file(fname, BUCKET_NAME, s3_path)
        url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{s3_path}"
        
        c.execute("INSERT INTO Usuarios (Nombre,Apellido,CI,Password,Tipo,Foto,Activo) VALUES (?,?,?,?,?,?,1)", (nombre,apellido,cedula,contrasena,tipo_usuario,url))
        conn.commit(); conn.close()
        registrar_auditoria("REGISTRO", f"Usuario nuevo: {nombre} {apellido} ({cedula})")
        return {"mensaje":"Registrado correctamente", "url":url}
    except Exception as e: return {"error": str(e)}

@app.post("/agregar_referencia_facial")
async def agregar_referencia(cedula: str = Form(...), foto: UploadFile = UploadFile(...)):
    try:
        folder_local = f"perfiles_db/{cedula}"
        if not os.path.exists(folder_local): os.makedirs(folder_local)
        fname = f"{folder_local}/ref_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        with open(fname, "wb") as f: shutil.copyfileobj(foto.file, f)
        
        bytes_img = forzar_compresion(fname)
        if not bytes_img: 
            with open(fname, 'rb') as f: bytes_img = f.read()
        
        try:
            rekog.index_faces(CollectionId=COLLECTION_ID, Image={'Bytes': bytes_img}, ExternalImageId=cedula, QualityFilter='AUTO')
            registrar_auditoria("ENTRENAMIENTO", f"Nueva foto de referencia para {cedula}")
            return {"mensaje": "Referencia agregada."}
        except Exception as e: return {"error": f"Error AWS: {e}"}
    except Exception as e: return {"error": str(e)}

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
                c.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash) VALUES (?,?,?)", (ci, url, fhash))
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
    estado = "sin_coincidencia"
    motivo = ""
    ext = fname.split('.')[-1].lower()
    es_video = ext in ['mp4', 'mov', 'avi', 'webm']
    es_imagen = ext in ['jpg', 'jpeg', 'png']
    es_documento = ext in ['pdf', 'docx', 'xlsx', 'pptx', 'txt']

    try:
        if not (es_video or es_imagen or es_documento): 
            borrar_ruta(temp_dir); return {"status": "error", "motivo": "Formato no soportado"}
        
        with open(fname,"wb") as f: shutil.copyfileobj(archivo.file,f)
        fhash = calcular_hash(fname)
        conn=sqlite3.connect(DB_NAME); c=conn.cursor()
        c.execute("SELECT id FROM Evidencias WHERE Hash=?",(fhash,))
        if c.fetchone(): 
            conn.close(); borrar_ruta(temp_dir)
            return {"status": "alerta", "motivo": "duplicado", "mensaje": "Archivo repetido"}
        conn.close()

        if es_documento: motivo = "Documento: Requiere asignación manual."
        elif es_video:
            try:
                cam = cv2.VideoCapture(fname)
                fps = cam.get(cv2.CAP_PROP_FPS) or 30
                frame_step = int(fps / 2) 
                if frame_step < 1: frame_step = 1
                total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
                curr = 0; scans = 0
                while curr < total_frames and scans < 40:
                    cam.set(cv2.CAP_PROP_POS_FRAMES, curr)
                    ret, frame = cam.read()
                    if not ret: break
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    try:
                        resp = rekog.search_faces_by_image(CollectionId=COLLECTION_ID, Image={'Bytes': buffer.tobytes()}, FaceMatchThreshold=40)
                        for m in resp['FaceMatches']: dest.add(m['Face']['ExternalImageId'])
                    except: pass
                    curr += frame_step; scans += 1
                cam.release()
            except Exception as e: print(f"Error video: {e}")
        elif es_imagen:
            img_bytes = forzar_compresion(fname)
            if img_bytes:
                personas = procesar_foto_grupal(img_bytes)
                if personas: dest.update(personas)
                try:
                    resp_txt = rekog.detect_text(Image={'Bytes': img_bytes})
                    txt = " ".join([t['DetectedText'] for t in resp_txt['TextDetections'] if t['Type'] == 'LINE'])
                    if txt: dest.update(buscar_estudiantes_texto_smart(txt))
                except: pass

        nombre_limpio = os.path.basename(fname)
        s3_client.upload_file(fname, BUCKET_NAME, f"evidencias/{nombre_limpio}")
        url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/evidencias/{nombre_limpio}"
        
        conn=sqlite3.connect(DB_NAME); c=conn.cursor()
        if dest:
            estado = "exito"
            nombres = []
            for ci in dest:
                c.execute("SELECT id, Nombre, Apellido FROM Usuarios WHERE CI=?", (ci,))
                ud = c.fetchone()
                if ud:
                    c.execute("INSERT INTO Evidencias (CI_Estudiante,Url_Archivo,Hash) VALUES (?,?,?)",(ci,url,fhash))
                    nombres.append(f"{ud[1]} {ud[2]}")
            registrar_auditoria("SUBIDA EXITOSA", f"Archivo: {archivo.filename} -> {', '.join(nombres)}")
        else:
            motivo = "Sin coincidencias."
            registrar_auditoria("SUBIDA FALLIDA", f"Archivo: {archivo.filename}")
        conn.commit(); conn.close(); borrar_ruta(temp_dir)
        return {"status": estado, "asignado_a": list(dest), "motivo": motivo, "url": url}
    except Exception as e: borrar_ruta(temp_dir); return {"status": "error", "motivo": str(e)}

@app.post("/iniciar_sesion")
async def iniciar_sesion(cedula: str = Form(...), contrasena: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Tipo, Nombre, Apellido, Activo FROM Usuarios WHERE CI=? AND Password=?", (cedula, contrasena))
    u = c.fetchone(); conn.close()
    if u: 
        if u[3] == 0: return {"encontrado": False, "mensaje": "Desactivado"}
        registrar_auditoria("LOGIN", f"Inicio de sesión: {u[1]} {u[2]} ({cedula})")
        return {"encontrado": True, "datos": {"tipo": u[0], "nombre": u[1], "apellido": u[2]}}
    return {"encontrado": False}

@app.get("/listar_usuarios")
async def listar():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Nombre, Apellido, CI, Tipo, Password, Activo FROM Usuarios")
    res = [{"nombre":x[0], "apellido":x[1], "cedula":x[2], "tipo":x[3], "contrasena":x[4], "activo":x[5]} for x in c.fetchall()]
    conn.close(); return res

@app.get("/todas_evidencias")
async def todas():
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT id, Url_Archivo, CI_Estudiante FROM Evidencias ORDER BY id DESC")
    res = [{"id":x[0], "url":x[1], "cedula":x[2]} for x in c.fetchall()]
    conn.close(); return res

@app.get("/obtener_logs")
async def obtener_logs():
    conn=sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("SELECT * FROM Auditoria ORDER BY id DESC LIMIT 50")
    l=[{"fecha":x[3],"accion":x[1],"detalle":x[2]} for x in c.fetchall()]
    conn.close(); return l

@app.delete("/borrar_evidencia/{id_ev}")
async def del_ev(id_ev: int):
    conn = sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_ev,))
    ev = c.fetchone(); nom = ev[0].split('/')[-1] if ev else "?"
    c.execute("DELETE FROM Evidencias WHERE id=?",(id_ev,))
    conn.commit(); conn.close()
    registrar_auditoria("ELIMINACIÓN", f"Evidencia borrada: {nom}")
    return {"msg":"Eliminada"}

@app.delete("/eliminar_usuario/{cedula}")
async def elim_user(cedula: str):
    conn = sqlite3.connect(DB_NAME); c=conn.cursor()
    c.execute("DELETE FROM Usuarios WHERE CI=?",(cedula,))
    c.execute("DELETE FROM Evidencias WHERE CI_Estudiante=?",(cedula,))
    conn.commit(); conn.close()
    if os.path.exists(f"perfiles_db/{cedula}"): shutil.rmtree(f"perfiles_db/{cedula}")
    
    # --- AQUÍ ESTABA EL ERROR, YA CORREGIDO EN LÍNEAS SEPARADAS ---
    try: 
        rekog.delete_faces(CollectionId=COLLECTION_ID, FaceIds=[cedula])
    except: 
        pass
    # -------------------------------------------------------------
    
    registrar_auditoria("BAJA USUARIO", f"Usuario eliminado: {cedula}")
    return {"msg":"Borrado"}

@app.post("/buscar_estudiante")
async def buscar_estudiante(cedula: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT Nombre, Apellido, Foto FROM Usuarios WHERE CI=?", (cedula,))
    p = c.fetchone()
    if p:
        c.execute("SELECT Url_Archivo FROM Evidencias WHERE CI_Estudiante=?", (cedula,))
        evs = [{"url": x[0]} for x in c.fetchall()]
        conn.close()
        return {"encontrado": True, "datos": {"nombre": p[0], "apellido": p[1], "url_foto": p[2], "galeria": evs}}
    conn.close(); return {"encontrado": False}

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
                s3_client.download_file(BUCKET_NAME, f"evidencias/{filename}", local_path)
                z.write(local_path, filename)
            except: pass
    registrar_auditoria("BACKUP FULL", f"Descarga multimedia")
    background_tasks.add_task(borrar_ruta, temp_work_dir)
    return FileResponse(zip_full_path, media_type="application/zip", filename=zip_name)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"--> Servidor FINAL INICIADO (DB dentro de Codigo)")
    uvicorn.run(app, host="0.0.0.0", port=port)