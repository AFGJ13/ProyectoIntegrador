# 🍋 Clasificación de Limón Tahití con YOLOv8 y Raspberry Pi 3

Sistema de visión artificial para la **clasificación automática de limón Tahití** en tres categorías: **Desecho**, **Local** y **Exportación**. El sistema utiliza una red neuronal YOLOv8 entrenada con 3.000 imágenes, desplegada sobre una arquitectura embebida compuesta por una Raspberry Pi 3 Model B, un PC y un Arduino UNO.

## 🧠 Descripción del proyecto

### Objetivo

Clasificar limones Tahití en tiempo real en tres categorías mediante visión artificial:

| Categoría    | Descripción                                      |
|--------------|--------------------------------------------------|
| **Exportación** | Limón de calidad óptima para mercados internacionales |
| **Local**       | Limón apto para mercado nacional                |
| **Desecho**     | Limón no apto para comercialización             |

### Arquitectura del sistema

```
Cámara USB
    │
    ▼
Raspberry Pi 3
    │  (Ethernet / Serial)
    ▼
PC (Inferencia YOLO)
    │
    ▼
Arduino UNO
    │
    ▼
Actuadores (LEDs / Servo)
```

La Raspberry Pi captura video, lo transmite al PC donde corre el modelo YOLOv8, y el resultado se devuelve al Arduino para accionar los mecanismos físicos de clasificación.

---

## 📦 Carpeta `limones_yolo/`

Contiene el modelo YOLOv8 con su dataset y entrenamiento completo.

### Dataset

- **Total de imágenes:** 3.000
- **Distribución:** 1.000 imágenes por categoría (`exportacion`, `local`, `desecho`)
- **Anotaciones:** Bounding boxes en formato YOLO (`.txt`)
- **Entrenado con:** YOLOv8 (Ultralytics)

### Entrenamiento

El modelo se entrena usando YOLOv8 de Ultralytics. El archivo `data.yaml` define las rutas del dataset y las clases:

```yaml
# data.yaml
train: dataset/images/train
val:   dataset/images/val
nc: 3
names: ['exportacion', 'local', 'desecho']
```

Comando de entrenamiento:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

Los pesos entrenados (`best.pt` y `best.onnx`) se generan en `runs/detect/train/weights/`.

---

## 🍓 Carpeta `clasificador_limones/`

Módulo de clasificación diseñado para correr en la **Raspberry Pi 3** usando el modelo exportado en formato ONNX (`best.onnx`). No requiere GPU ni Ultralytics; solo `onnxruntime`.

### Archivos

| Archivo            | Descripción                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `main.py`          | Programa principal: captura frames, corre inferencia y muestra resultado    |
| `camera.py`        | Clase `Camera` — abstracción de la cámara USB con OpenCV                    |

### Instalación en Raspberry Pi

```bash
# Actualizar sistema
sudo apt update
sudo apt install python3-pip libatlas-base-dev -y

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
# Ejecutar clasificador principal
python main.py


---

## 💻 Carpeta `codigos/`

Scripts del **sistema embebido completo** que integra PC, Raspberry Pi y Arduino. Contiene múltiples versiones del pipeline según la configuración de red utilizada.

### Flujo completo del sistema

```
1. Arduino detecta limón (sensor ultrasónico)
2. Arduino envía "OBJETO" por serial a Raspberry
3. Raspberry toma foto con cámara USB
4. Raspberry envía imagen al PC por Ethernet (HTTP)
5. PC corre inferencia con YOLOv8
6. PC responde con la clasificación (desecho / local / exportacion)
7. Raspberry recibe el resultado
8. Raspberry reenvía al Arduino por serial
9. Arduino mueve servo / enciende LED según la categoría
```

### Mapeo de clases

| Clase YOLO    | Código enviado | Pin Arduino | Ángulo Servo |
|---------------|----------------|-------------|--------------|
| `desecho`     | 1              | Pin 10      | 150°         |
| `exportacion` | 2              | Pin 10      | 90°          |
| `local`       | 3              | Pin 10      | 0°           |

### Instalación en PC

```bash
pip install -r requirements.txt
```

### Ejecución

```bash
# Modo 1: PC directo con cámara USB y Arduino serial
python camera.py

# Modo 2: Sistema distribuido (Raspberry + PC)
#   En PC:
python detector.py
#   En Raspberry:
python cliente_raspberry.py

# Modo 3: Sistema con Arduino detector
#   Subir arduino_detector.ino al Arduino
#   En PC:
python detector.py
#   En Raspberry:
python main_pi.py

---

## ⚙️ Modelo `best.onnx`

El archivo `best.onnx` es el modelo YOLOv8 exportado desde PyTorch a formato ONNX para su uso en la Raspberry Pi. Este formato permite inferencia eficiente en CPU sin necesidad de instalar Ultralytics ni CUDA.

Fue entrenado con:

- **Arquitectura:** YOLOv8n (nano)
- **Dataset:** 3.000 imágenes (640×640, bounding boxes)
- **Clases:** `exportacion`, `local`, `desecho`
- **Framework:** Ultralytics YOLOv8
- **Exportación:**

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=320)
```

---

## 🔧 Hardware utilizado

- Raspberry Pi 3 Model B
- Arduino UNO
- Cámara USB (640×480)
- Cable Ethernet (conexión directa PC ↔ Raspberry)
- Sensor ultrasónico HC-SR04
- Servo motor

---

## 📡 Configuración de red

| Dispositivo   | IP              |
|---------------|-----------------|
| PC            | 192.168.137.1   |
| Raspberry Pi  | 192.168.137.2   |

La comunicación se realiza por Ethernet en red local (sin internet requerido).

## 📄 Licencia

Proyecto académico — uso libre con fines educativos.
