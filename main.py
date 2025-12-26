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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from botocore.config import Config
from pydantic import BaseModel

# --- 0. CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = "Bases_de_datos.db"
DB_ORIGINAL_PATH = os.path.join(BASE_DIR, DB_FILENAME)

# --- CONFIGURACIÓN DE CORREO ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "tu_correo_sistema@gmail.com"  # Cambiar por tu correo real
SMTP_PASSWORD = "tu_contraseña_aplicacion"   # Cambiar por tu contraseña de app

# --- 1. CONFIGURACIÓN Y CREDENCIALES AWS/B2 ---
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = "us-east-1"
COLLECTION_ID = "estudiantes_db"

try:
    if AWS_ACCESS_KEY:
        rekog = boto3.client('rekognition', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    else:
        rekog = None
except:
    rekog = None

ENDPOINT_B2 = "https://s3.us-east-005.backblazeb2.com"
KEY_ID_B2 = "00508884373dab40000000001"
APP_KEY_B2 = "K005jvkLLmLdUKhhVis1qLcnU4flx0g"
BUCKET_NAME = "Proyecto-Grado-Karlos-2025"

try:
    my_config = Config(signature_version='s3v4', region_name='us-east-005')
    s3_client = boto3.client('s3', endpoint_url=ENDPOINT_B2, aws_access_key_id=KEY_ID_B2, aws_secret_access_key=APP_KEY_B2, config=my_config)
except:
    s3_client = None

# Logging y DB
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- LÓGICA DE VOLUMEN PERSISTENTE ---
VOLUMEN_PATH = "/app/datos_persistentes"
DB_PATH_FINAL = DB_ORIGINAL_PATH 

if os.path.exists(VOLUMEN_PATH):
    db_en_volumen = os.path.join(VOLUMEN_PATH, DB_FILENAME)
    if not os.path.exists(db_en_volumen):
        if os.path.exists(DB_ORIGINAL_PATH):
            shutil.copy(DB_ORIGINAL_PATH, db_en_volumen)
    DB_PATH_FINAL = db_en_volumen

DB_NAME = DB_PATH_FINAL

class EstadoUsuarioRequest(BaseModel):
    cedula: str
    activo: int

# --- 2. INICIALIZACIÓN DE BASE DE DATOS ---
def init_db_completa():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS Usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Nombre TEXT NOT NULL,
            Apellido TEXT NOT NULL,
            CI TEXT UNIQUE NOT NULL,
            Password TEXT NOT NULL,
            Tipo INTEGER DEFAULT 1,
            Foto TEXT,
            Activo INTEGER DEFAULT 1,
            TutorialVisto INTEGER DEFAULT 0,
            Fecha_Registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Face_Encoding TEXT
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS Evidencias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            CI_Estudiante TEXT,
            Url_Archivo TEXT NOT NULL,
            Hash TEXT,
            Estado INTEGER DEFAULT 1,
            Tipo_Archivo TEXT,
            Fecha_Subida TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(CI_Estudiante) REFERENCES Usuarios(CI)
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS Solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Tipo TEXT,
            CI_Solicitante TEXT,
            Email TEXT,
            Detalle TEXT,
            Evidencia_Reportada_Url TEXT,
            Id_Evidencia INTEGER,
            Resuelto_Por TEXT,
            Respuesta TEXT,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Estado TEXT DEFAULT 'PENDIENTE'
        )''')
        
        try:
            c.execute("ALTER TABLE Solicitudes ADD COLUMN Respuesta TEXT")
        except:
            pass

        c.execute('''CREATE TABLE IF NOT EXISTS Auditoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Accion TEXT,
            Detalle TEXT,
            IP TEXT,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        c.execute("SELECT CI FROM Usuarios WHERE Tipo=0")
        if not c.fetchone():
            c.execute("INSERT INTO Usuarios (Nombre, Apellido, CI, Password, Tipo, Activo) VALUES (?,?,?,?,?,?)", 
                     ('Admin', 'Sistema', '9999999999', 'admin123', 0, 1))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error inicializando DB: {e}")

init_db_completa()

# --- 3. CONFIGURACIÓN DE CORS (CORREGIDA Y LIMPIA) ---
app = FastAPI()

# Permitimos todo para evitar errores de origen cruzado (CORS)
origins = [
    "https://proyecto-grado-karlos.vercel.app",    # <--- Tu Web en Vercel (INVITADO VIP)
    "http://127.0.0.1:5500",                         # <--- Tu PC
    "http://localhost:5500",                         # <--- Tu PC alternativo
    "https://proyecto-de-grado-oficial-production.up.railway.app" # <--- El mismo servidor
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # <--- Usamos la lista específica, NO ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIONES AUXILIARES ---
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def registrar_auditoria(accion, detalle):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO Auditoria (Accion, Detalle) VALUES (?, ?)", (accion, detalle))
        conn.commit(); conn.close()
    except: pass

def enviar_correo_real(destinatario, asunto, mensaje):
    try:
        if "tu_correo" in SMTP_EMAIL:
            print(f"📧 [SIMULACION EMAIL] A: {destinatario} | Msg: {mensaje}")
            return False
            
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = destinatario
        msg['Subject'] = asunto
        msg.attach(MIMEText(mensaje, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_EMAIL, destinatario, text)
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Error enviando email: {e}")
        return False

def calcular_hash(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(4096), b""): h.update(b)
    return h.hexdigest()

# --- ENDPOINTS PRINCIPALES ---

@app.get("/")
def home():
    return {
        "status": "online", 
        "backend": "Sistema Educativo Despertar V4.0",
        "cors_enabled": True,
        "mode": "PERMISSIVE_ALL"
    }

@app.post("/iniciar_sesion")
@app.post("/buscar_estudiante")
async def buscar_estudiante(cedula: str = Form(...), contrasena: Optional[str] = Form(None)):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM Usuarios WHERE CI=?", (cedula,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return JSONResponse(content={"encontrado": False, "mensaje": "Usuario no encontrado"})
    
    if contrasena and user["Password"] != contrasena:
        conn.close()
        return JSONResponse(content={"encontrado": False, "mensaje": "Contraseña incorrecta"})
    
    if user["Activo"] == 0:
        conn.close()
        return JSONResponse(content={"encontrado": False, "mensaje": "Cuenta desactivada por administración"})
        
    c.execute("SELECT id, Url_Archivo as url, Tipo_Archivo as tipo, Fecha_Subida FROM Evidencias WHERE CI_Estudiante=? AND Estado=1 ORDER BY Fecha_Subida DESC", (cedula,))
    evs = [dict(row) for row in c.fetchall()]

    c.execute("""SELECT Tipo, Estado, Respuesta, Fecha FROM Solicitudes 
                 WHERE CI_Solicitante=? AND Estado != 'PENDIENTE' 
                 ORDER BY Fecha DESC LIMIT 5""", (cedula,))
    notis = [dict(row) for row in c.fetchall()]

    conn.close()
    
    return JSONResponse(content={
        "encontrado": True,
        "datos": {
            "nombre": user["Nombre"],
            "apellido": user["Apellido"],
            "cedula": user["CI"],
            "tipo": user["Tipo"],
            "url_foto": user["Foto"],
            "galeria": evs,
            "notificaciones": notis
        }
    })

@app.post("/registrar_usuario")
async def registrar_usuario(nombre: str=Form(...), apellido: str=Form(...), cedula: str=Form(...), contrasena: str=Form(...), tipo_usuario: int=Form(...), foto: UploadFile=UploadFile(...)):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT CI FROM Usuarios WHERE CI=?", (cedula,))
        if c.fetchone(): 
            conn.close()
            return JSONResponse(content={"error": "Usuario ya existe"})
        
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, foto.filename)
        with open(path, "wb") as f: shutil.copyfileobj(foto.file, f)
        
        nombre_nube = f"perfiles/p_{cedula}_{foto.filename}"
        if s3_client:
            s3_client.upload_file(path, BUCKET_NAME, nombre_nube)
            url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
        else:
            url = f"/datos_persistentes/local_{foto.filename}"
        
        c.execute("INSERT INTO Usuarios (Nombre,Apellido,CI,Password,Tipo,Foto,Activo) VALUES (?,?,?,?,?,?,1)", 
                 (nombre,apellido,cedula,contrasena,tipo_usuario,url))
        conn.commit()
        conn.close()
        shutil.rmtree(temp_dir)
        
        return JSONResponse(content={"mensaje": "Registrado", "url": url})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.post("/solicitar_recuperacion")
async def solicitar_recuperacion(cedula: str = Form(None), email: str = Form(...), mensaje: str = Form(None)):
    try:
        ci_final = cedula if cedula else (email.split('@')[0] if '@' in email else email)
        conn = get_db_connection()
        user = conn.execute("SELECT Nombre, Apellido FROM Usuarios WHERE CI=?", (ci_final,)).fetchone()
        nombre_usuario = f"{user['Nombre']} {user['Apellido']}" if user else "Usuario Desconocido"
        
        conn.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Email, Detalle, Estado) VALUES (?,?,?,?,?)",
                  ("RECUPERACION", ci_final, email, mensaje or f"Solicitud de: {nombre_usuario}", "PENDIENTE"))
        conn.commit(); conn.close()
        return JSONResponse(content={"status": "ok", "mensaje": "Solicitud enviada a los administradores."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "mensaje": str(e)})

@app.post("/reportar_evidencia")
async def reportar_evidencia(cedula: str = Form(...), id_evidencia: int = Form(...), motivo: str = Form(...)):
    conn = get_db_connection()
    ev = conn.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_evidencia,)).fetchone()
    url = ev['Url_Archivo'] if ev else ""
    
    conn.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Id_Evidencia, Evidencia_Reportada_Url, Detalle, Estado) VALUES (?,?,?,?,?,?)",
              ("REPORTE", cedula, id_evidencia, url, motivo, "PENDIENTE"))
    conn.commit(); conn.close()
    return JSONResponse(content={"status": "ok", "mensaje": "Reporte enviado."})

@app.post("/reportar_problema")
async def reportar_problema(cedula: str = Form(...), mensaje: str = Form(...)):
    conn = get_db_connection()
    conn.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Detalle, Estado) VALUES (?,?,?,?)",
              ("PROBLEMA", cedula, mensaje, "PENDIENTE"))
    conn.commit(); conn.close()
    return JSONResponse(content={"status": "ok", "mensaje": "Problema reportado."})

@app.post("/solicitar_subida")
async def solicitar_subida(cedula: str = Form(...), archivo: UploadFile = UploadFile(...), comentario: str = Form(None)):
    try:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, archivo.filename)
        with open(path, "wb") as f: shutil.copyfileobj(archivo.file, f)
        
        import uuid
        nombre_nube = f"evidencias/pend_{uuid.uuid4().hex}_{archivo.filename}"
        if s3_client:
            s3_client.upload_file(path, BUCKET_NAME, nombre_nube)
            url = f"https://{BUCKET_NAME}.s3.us-east-005.backblazeb2.com/{nombre_nube}"
        else:
            url = f"/datos_persistentes/{archivo.filename}"
        
        fhash = calcular_hash(path)
        shutil.rmtree(temp_dir)
        
        conn = get_db_connection()
        cursor = conn.execute("INSERT INTO Evidencias (CI_Estudiante, Url_Archivo, Hash, Estado, Tipo_Archivo) VALUES (?,?,?,0,?)", 
                     (cedula, url, fhash, "documento"))
        id_evidencia = cursor.lastrowid
        
        detalle = f"Subida solicitada: {archivo.filename}. " + (comentario if comentario else "")
        conn.execute("INSERT INTO Solicitudes (Tipo, CI_Solicitante, Id_Evidencia, Evidencia_Reportada_Url, Detalle, Estado) VALUES (?,?,?,?,?,?)",
                  ("SUBIDA", cedula, id_evidencia, url, detalle, "PENDIENTE"))
        conn.commit(); conn.close()
        return JSONResponse(content={"status": "ok", "mensaje": "Archivo enviado a aprobación."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "mensaje": str(e)})

# --- GESTIÓN DE ADMIN ---

@app.get("/obtener_solicitudes")
async def obtener_solicitudes():
    conn = get_db_connection()
    c = conn.cursor()
    query = """
        SELECT s.id, s.Tipo, s.CI_Solicitante, s.Detalle, s.Id_Evidencia, s.Fecha, s.Estado, s.Email, s.Evidencia_Reportada_Url,
               IFNULL(u.Nombre, 'Usuario') as Nombre, IFNULL(u.Apellido, 'Desconocido') as Apellido
        FROM Solicitudes s
        LEFT JOIN Usuarios u ON s.CI_Solicitante = u.CI
        WHERE s.Estado = 'PENDIENTE'
        ORDER BY s.Fecha DESC
    """
    c.execute(query)
    data = [dict(row) for row in c.fetchall()]
    conn.close()
    return JSONResponse(content=data)

@app.post("/gestionar_solicitud")
async def gestionar_solicitud(
    id_solicitud: int = Form(...), 
    accion: str = Form(...), 
    mensaje: str = Form(""), 
    id_admin: str = Form("Admin")
):
    accion_norm = "APROBADA" if accion in ['APROBAR', 'ACEPTAR'] else "RECHAZADA"
    conn = get_db_connection()
    sol = conn.execute("SELECT Tipo, Id_Evidencia, CI_Solicitante, Email, Evidencia_Reportada_Url FROM Solicitudes WHERE id=?", (id_solicitud,)).fetchone()
    
    if not sol: 
        conn.close()
        return JSONResponse(content={"error": "Solicitud no encontrada"})
        
    tipo = sol['Tipo']
    id_evidencia = sol['Id_Evidencia']
    email_usuario = sol['Email']
    
    if tipo == 'SUBIDA':
        if accion_norm == 'APROBADA':
            conn.execute("UPDATE Evidencias SET Estado=1 WHERE id=?", (id_evidencia,))
        else:
            conn.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
            if sol['Evidencia_Reportada_Url']: eliminar_archivo_nube(sol['Evidencia_Reportada_Url'])

    elif tipo == 'REPORTE':
        if accion_norm == 'APROBADA':
            conn.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
            if sol['Evidencia_Reportada_Url']: eliminar_archivo_nube(sol['Evidencia_Reportada_Url'])

    elif tipo == 'RECUPERACION':
        if mensaje and email_usuario:
            asunto = "Respuesta a tu solicitud de Recuperación - U.E. Despertar"
            cuerpo = f"Hola,\n\nEl administrador ha respondido a tu solicitud:\n\n'{mensaje}'\n\nAtentamente,\nSoporte U.E. Despertar"
            enviar_correo_real(email_usuario, asunto, cuerpo)

    conn.execute("UPDATE Solicitudes SET Estado=?, Resuelto_Por=?, Respuesta=? WHERE id=?", 
                 (accion_norm, id_admin, mensaje, id_solicitud))
    conn.commit()
    conn.close()
    return JSONResponse(content={"status": "ok", "mensaje": "Solicitud procesada correctamente"})

def eliminar_archivo_nube(url):
    try:
        if s3_client and BUCKET_NAME in url:
            key = url.split(f"{BUCKET_NAME}/")[-1]
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
    except: pass

@app.get("/listar_usuarios")
async def listar():
    conn = get_db_connection()
    res = [dict(row) for row in conn.execute("SELECT Nombre, Apellido, CI, Tipo, Password, Activo, Foto FROM Usuarios").fetchall()]
    conn.close()
    return JSONResponse(content=res)

@app.get("/cors-test")
async def cors_test():
    return JSONResponse(content={
            "message": "CORS está funcionando correctamente",
            "timestamp": datetime.datetime.now().isoformat()
        })

@app.get("/static/{file_path:path}")
async def serve_static_file(file_path: str):
    static_path = os.path.join(BASE_DIR, "static", file_path)
    if not os.path.exists(static_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(static_path)

@app.get("/health")
async def health_check():
    return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "cors": "enabled",
            "database": "connected" if os.path.exists(DB_NAME) else "disconnected"
        })

@app.get("/todas_evidencias")
async def todas_evidencias(cedula: str):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT id, Url_Archivo as url, Tipo_Archivo as tipo, 
                   Fecha_Subida, Estado, Hash 
            FROM Evidencias 
            WHERE CI_Estudiante=? 
            ORDER BY Fecha_Subida DESC
        """, (cedula,))
        evs = [dict(row) for row in c.fetchall()]
        conn.close()
        return JSONResponse(content=evs)
    except Exception as e:
        return JSONResponse(content=[])

@app.get("/resumen_estudiantes_con_evidencias")
async def resumen_estudiantes():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        query = """
            SELECT u.Nombre as nombre, u.Apellido as apellido, u.CI as cedula, u.Foto as foto,
                   COUNT(e.id) as total_evidencias
            FROM Usuarios u
            LEFT JOIN Evidencias e ON u.CI = e.CI_Estudiante
            WHERE u.Tipo = 1
            GROUP BY u.CI
            ORDER BY total_evidencias DESC, u.Apellido ASC
        """
        c.execute(query)
        data = [dict(row) for row in c.fetchall()]
        conn.close()
        return JSONResponse(content=data)
    except Exception as e:
        print(f"Error resumen: {e}")
        return JSONResponse(content=[])

@app.get("/estadisticas_almacenamiento")
async def stats_storage():
    try:
        conn = get_db_connection()
        users = conn.execute("SELECT COUNT(*) FROM Usuarios WHERE Activo=1").fetchone()[0]
        files = conn.execute("SELECT COUNT(*) FROM Evidencias").fetchone()[0]
        gb_aprox = (files * 2.5) / 1024 
        conn.close()
        return JSONResponse(content={
                "usuarios_activos": users,
                "total_evidencias": files,
                "almacenamiento_gb": gb_aprox
            })
    except Exception as e:
        return JSONResponse(content={})

@app.get("/obtener_logs")
async def obtener_logs():
    try:
        conn = get_db_connection()
        logs = conn.execute("SELECT * FROM Auditoria ORDER BY Fecha DESC LIMIT 50").fetchall()
        data = [dict(row) for row in logs]
        conn.close()
        return JSONResponse(content=data)
    except:
        return JSONResponse(content=[])

@app.delete("/eliminar_evidencia/{id_evidencia}")
async def eliminar_evidencia_endpoint(id_evidencia: int):
    try:
        conn = get_db_connection()
        ev = conn.execute("SELECT Url_Archivo FROM Evidencias WHERE id=?", (id_evidencia,)).fetchone()
        if ev:
            url = ev['Url_Archivo']
            eliminar_archivo_nube(url)
            conn.execute("DELETE FROM Evidencias WHERE id=?", (id_evidencia,))
            conn.commit()
        conn.close()
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "detalle": str(e)})

@app.post("/cambiar_estado_usuario")
async def cambiar_estado_usuario(datos: EstadoUsuarioRequest):
    try:
        conn = get_db_connection()
        conn.execute("UPDATE Usuarios SET Activo=? WHERE CI=?", (datos.activo, datos.cedula))
        conn.commit()
        conn.close()
        return JSONResponse(content={"mensaje": "Estado actualizado"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"--> Servidor con CORS MEJORADO INICIADO en puerto {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)