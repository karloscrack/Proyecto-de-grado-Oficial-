import sqlite3

DB_NAME = "Bases_de_datos.db"

def reconstruir_tabla_solicitudes():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    print("🛠️ Eliminando tabla antigua 'Solicitudes'...")
    c.execute("DROP TABLE IF EXISTS Solicitudes")
    
    print("✨ Creando la NUEVA tabla 'Solicitudes' con superpoderes...")
    # Esta estructura sirve para TODO:
    # - Tipo: 'RECUPERACION', 'REPORTE', 'SUBIDA'
    # - Estado: 'PENDIENTE', 'APROBADO', 'RECHAZADO'
    c.execute('''
        CREATE TABLE Solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Tipo TEXT,
            CI_Solicitante TEXT,
            Email TEXT,
            Detalle TEXT,
            Id_Evidencia INTEGER,
            Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Estado TEXT DEFAULT 'PENDIENTE'
        )
    ''')
    
    # También necesitamos asegurarnos que la tabla Evidencias tenga la columna 'Estado'
    # Estado 1 = Visible, Estado 0 = Oculto (Pendiente de aprobación)
    try:
        c.execute("ALTER TABLE Evidencias ADD COLUMN Estado INTEGER DEFAULT 1")
        print("✅ Columna 'Estado' agregada a Evidencias.")
    except:
        print("ℹ️ La columna 'Estado' ya existía en Evidencias.")

    conn.commit()
    conn.close()
    print("🚀 ¡Base de datos lista! Ya puedes borrar este archivo.")

if __name__ == "__main__":
    reconstruir_tabla_solicitudes()