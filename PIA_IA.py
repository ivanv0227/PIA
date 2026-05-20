import cv2
import face_recognition
import pickle
import numpy as np
import serial
import time

# =========================
# CONEXION SERIAL ARDUINO
# =========================

arduino = serial.Serial(
    'COM3',
    9600
)

time.sleep(2)

# =========================
# CARGAR EMBEDDINGS
# =========================

with open("rostros.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("Base de datos cargada")

# =========================
# ABRIR CAMARA
# =========================

cap = cv2.VideoCapture(0)

cap.set(3, 720)
cap.set(4, 640)

# =========================
# VARIABLES
# =========================

process_this_frame = True

face_locations = []
face_names = []

ultimo_estado = ""

# Tiempo requerido
TIEMPO_REQUERIDO = 2.0

# Tiempo pausa
TIEMPO_PAUSA = 5.0

# Temporizadores
inicio_reconocimiento = None
inicio_desconocido = None

# Tiempo bloqueo
tiempo_bloqueo = 0

# Porcentaje
porcentaje = 0

# Estados
rostro_desconocido = False
rostro_reconocido = False
reconocimiento_activo = False

# =========================
# LOOP PRINCIPAL
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # LEER MENSAJES ARDUINO
    # =========================

    if arduino.in_waiting:

        mensaje = (
            arduino.readline()
            .decode()
            .strip()
        )

        print(
            f"Arduino: {mensaje}"
        )

        # ACTIVAR IA

        if mensaje == "START":

            reconocimiento_activo = True

            print(
                "Reconocimiento ACTIVADO"
            )

    # MODO ESPERA

    if not reconocimiento_activo:

        cv2.putText(
            frame,
            "Presiona boton para iniciar",
            (80,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,255),
            2
        )

        cv2.imshow(
            "Reconocimiento Facial",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # =========================
    # PAUSA TEMPORAL
    # =========================

    if time.time() < tiempo_bloqueo:

        restante = int(
            tiempo_bloqueo - time.time()
        )

        cv2.putText(
            frame,
            f"Espera {restante}s",
            (180, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.imshow(
            "Reconocimiento Facial",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # REDUCIR FRAME

    small_frame = cv2.resize(
        frame,
        (0,0),
        fx=0.15,
        fy=0.15
    )

    # BGR -> RGB
    rgb_small = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    # PROCESAR FRAMES

    if process_this_frame:

        rostro_reconocido = False
        rostro_desconocido = False

        # Detectar rostros
        face_locations = face_recognition.face_locations(
            rgb_small,
            model="cnn"
        )

        # Embeddings
        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        face_names = []

        # COMPARAR ROSTROS

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding
            )

            face_distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            name = "Desconocido"

            best_match_index = np.argmin(
                face_distances
            )

            if matches[best_match_index]:

                name = known_names[
                    best_match_index
                ]

                rostro_reconocido = True

            else:

                rostro_desconocido = True

            face_names.append(name)

        # AUTORIZADO

        if rostro_reconocido:

            inicio_desconocido = None

            if inicio_reconocimiento is None:

                inicio_reconocimiento = time.time()

            tiempo_actual = time.time()

            tiempo_detectado = (
                tiempo_actual -
                inicio_reconocimiento
            )

            porcentaje = int(
                (tiempo_detectado /
                 TIEMPO_REQUERIDO) * 100
            )

            porcentaje = min(
                porcentaje,
                100
            )

            print(
                f"Autorizado... "
                f"{porcentaje}%"
            )

            # OPEN

            if porcentaje >= 100:

                if ultimo_estado != "OPEN":

                    arduino.write(
                        b'OPEN\n'
                    )

                    print(
                        "OPEN enviado"
                    )

                    ultimo_estado = "OPEN"

                    # Desactivar IA
                    reconocimiento_activo = False

                    # Pausa
                    tiempo_bloqueo = (
                        time.time() +
                        TIEMPO_PAUSA
                    )

                    # Reset
                    inicio_reconocimiento = None
                    porcentaje = 0

        # DESCONOCIDO

        elif rostro_desconocido:

            inicio_reconocimiento = None

            if inicio_desconocido is None:

                inicio_desconocido = time.time()

            tiempo_actual = time.time()

            tiempo_detectado = (
                tiempo_actual -
                inicio_desconocido
            )

            porcentaje = int(
                (tiempo_detectado /
                 TIEMPO_REQUERIDO) * 100
            )

            porcentaje = min(
                porcentaje,
                100
            )

            print(
                f"Desconocido... "
                f"{porcentaje}%"
            )

            # DENY

            if porcentaje >= 100:

                if ultimo_estado != "DENY":

                    arduino.write(
                        b'DENY\n'
                    )

                    print(
                        "DENY enviado"
                    )

                    ultimo_estado = "DENY"

                    # Desactivar IA
                    reconocimiento_activo = False

                    # Pausa
                    tiempo_bloqueo = (
                        time.time() +
                        TIEMPO_PAUSA
                    )

                    # Reset
                    inicio_desconocido = None
                    porcentaje = 0

        # NO HAY ROSTROS

        else:

            inicio_reconocimiento = None
            inicio_desconocido = None

            porcentaje = 0

            ultimo_estado = ""

    # Alternar procesamiento
    process_this_frame = (
        not process_this_frame
    )

    # DIBUJAR

    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        top = int(top * 6.666)
        right = int(right * 6.666)
        bottom = int(bottom * 6.666)
        left = int(left * 6.666)

        # Color
        if name == "Desconocido":

            color = (0,0,255)

        else:

            color = (0,255,0)

        # Rectangulo
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        # Nombre
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # BARRA
    
    if porcentaje > 0:

        cv2.rectangle(
            frame,
            (20,20),
            (320,60),
            (50,50,50),
            -1
        )

        barra = int(
            (porcentaje / 100) * 300
        )

        if rostro_desconocido:

            barra_color = (0,0,255)

        else:

            barra_color = (0,255,0)

        cv2.rectangle(
            frame,
            (20,20),
            (20 + barra,60),
            barra_color,
            -1
        )

        cv2.putText(
            frame,
            f"{porcentaje}%",
            (130,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

    # =========================
    # MOSTRAR VIDEO
    # =========================

    cv2.imshow(
        "Reconocimiento Facial",
        frame
    )

    # ESC salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# LIBERAR
# =========================

cap.release()
cv2.destroyAllWindows()

arduino.close()