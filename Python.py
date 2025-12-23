import sqlite3

# Nombre exacto de tu base de datos
database_name = "Bases_de_datos.db"

try:
    conexion = sqlite3.connect(database_name)
    cursor = conexion.cursor()

    # Consultamos todas las columnas importantes
    cursor.execute("SELECT id, Nombre, Apellido, CI, Password, Tipo, Activo FROM Usuarios") 
    resultado = cursor.fetchall()

    print("\n=======================================================")
    print(f"  🔍 REPORTE DE USUARIOS EN '{database_name}'")
    print("=======================================================")

    if resultado:
        for fila in resultado:
            id_u, nombre, apellido, ci, password, tipo, activo = fila
            
            # Traducimos los códigos numéricos a texto legible
            rol = "👮 ADMINISTRADOR" if tipo == 0 else "🎓 ESTUDIANTE"
            estado = "✅ ACTIVO" if (activo == 1 or activo is None) else "🚫 DESACTIVADO"
            
            print(f"👤 {nombre} {apellido}")
            print(f"   🆔 Cédula: {ci}")
            print(f"   🔑 Clave:  {password}")
            print(f"   🔰 Rol:    {rol}")
            print(f"   {estado}")
            print("-------------------------------------------------------")
    else:
        print("  ⚠️ La tabla 'Usuarios' está vacía.")
        print("  -> Debes registrar un usuario nuevo desde la página web.")

except sqlite3.OperationalError as e:
    print(f"\n❌ Error: No se encuentra la tabla 'Usuarios' o la base de datos.")
    print(f"Detalle: {e}")
except Exception as e:
    print(f"\n❌ Error inesperado: {e}")
finally:
    if 'conexion' in locals():
        conexion.close()