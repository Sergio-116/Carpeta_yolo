# Proyecto YOLO para deteccion de perifericos en Python y Raspberry Pi

## 1. Objetivo del proyecto

El objetivo de este proyecto es entrenar un modelo de deteccion de objetos con YOLO para reconocer perifericos de computador usando Python. El modelo se entrena en un computador y luego se puede ejecutar en una Raspberry Pi con camara.

El sistema fue preparado para detectar 10 clases:

| ID | Clase |
| --- | --- |
| 0 | teclado |
| 1 | mouse |
| 2 | monitor |
| 3 | impresora |
| 4 | parlantes |
| 5 | webcam |
| 6 | microfono |
| 7 | audifonos |
| 8 | memoria_usb |
| 9 | router |

## 2. Tecnologias utilizadas

### Python

Se uso Python como lenguaje principal del proyecto. Python permite trabajar facilmente con inteligencia artificial, camaras, imagenes y automatizacion de archivos.

Version usada durante el desarrollo:

```cmd
Python 3.14.5
```

### Ultralytics YOLO

Se uso la libreria `ultralytics`, que permite entrenar, validar, exportar y ejecutar modelos YOLO.

YOLO significa "You Only Look Once". Es un modelo de vision por computador que detecta objetos dentro de imagenes o video en tiempo real.

Modelo base usado:

```text
yolo11n.pt
```

Este modelo es pequeno y rapido, por eso es una buena opcion para pruebas y para correr despues en Raspberry Pi.

### OpenCV

Se uso `opencv-python` para leer imagenes, procesarlas y trabajar con camaras.

OpenCV permite:

- Abrir la camara.
- Leer imagenes.
- Dibujar cajas.
- Procesar imagenes para crear etiquetas automaticas aproximadas.

### NumPy

Se uso `numpy` para procesar matrices de imagenes y calcular areas, mascaras y coordenadas.

### PyYAML

Se uso `PyYAML` para leer archivos `.yaml`, especialmente el archivo de configuracion del dataset:

```text
data/perifericos.yaml
```

## 3. Ubicacion del proyecto

El proyecto esta ubicado en:

```text
C:\Users\ASUS\Documents\Codex\yolo-perifericos-raspberry
```

## 4. Estructura del proyecto

```text
yolo-perifericos-raspberry/
  data/
    perifericos.yaml
  dataset/
    Pendientes por etiqueta/
    images/
      train/
      val/
    labels/
      train/
      val/
    preview_auto_labels.jpg
  models/
  runs/
    detect/
  scripts/
    capture_images.py
    check_dataset.py
    detect.py
    export_raspberry.py
    make_label_preview.py
    prepare_auto_labels.py
    train.py
  requirements.txt
  requirements-raspberry.txt
  README.md
  README_PROYECTO.md
```

## 5. Que hace cada carpeta

### `data/`

Contiene la configuracion del dataset.

Archivo principal:

```text
data/perifericos.yaml
```

Este archivo le dice a YOLO donde estan las imagenes, donde estan las clases y como se llaman.

Contenido usado:

```yaml
path: C:/Users/ASUS/Documents/Codex/yolo-perifericos-raspberry/dataset
train: images/train
val: images/val

names:
  0: teclado
  1: mouse
  2: monitor
  3: impresora
  4: parlantes
  5: webcam
  6: microfono
  7: audifonos
  8: memoria_usb
  9: router
```

### `dataset/Pendientes por etiqueta/`

Aqui se dejaron las imagenes originales organizadas por periferico y separadas en `Train` y `Val`.

Ejemplo:

```text
Pendientes por etiqueta/
  Mouse Computadora/
    Train/
    Val/
  Teclado Computador/
    Train/
    Val/
```

Esta carpeta funciona como entrada del proceso de etiquetado automatico.

### `dataset/images/train/`

Aqui quedan las imagenes que YOLO usa para aprender.

### `dataset/images/val/`

Aqui quedan las imagenes que YOLO usa para validar si esta aprendiendo bien.

### `dataset/labels/train/`

Aqui quedan las etiquetas de las imagenes de entrenamiento.

### `dataset/labels/val/`

Aqui quedan las etiquetas de las imagenes de validacion.

Cada imagen debe tener un archivo `.txt` con el mismo nombre.

Ejemplo:

```text
dataset/images/train/mouse_computadora_train_0001.jpg
dataset/labels/train/mouse_computadora_train_0001.txt
```

## 6. Formato de etiquetas YOLO

YOLO necesita un archivo `.txt` por cada imagen.

Cada linea tiene este formato:

```text
class_id x_center y_center width height
```

Ejemplo:

```text
1 0.512000 0.488000 0.700000 0.820000
```

Esto significa:

| Valor | Significado |
| --- | --- |
| 1 | ID de la clase, en este caso mouse |
| 0.512000 | centro del objeto en X |
| 0.488000 | centro del objeto en Y |
| 0.700000 | ancho de la caja |
| 0.820000 | alto de la caja |

Las coordenadas estan normalizadas, es decir, van de `0` a `1`.

## 7. Como se hizo el etiquetado

Se creo un script llamado:

```text
scripts/prepare_auto_labels.py
```

Este script hace lo siguiente:

1. Lee las carpetas dentro de `dataset/Pendientes por etiqueta`.
2. Identifica la clase segun el nombre de la carpeta.
3. Lee las imagenes dentro de `Train` y `Val`.
4. Copia las imagenes a `dataset/images/train` o `dataset/images/val`.
5. Genera automaticamente un archivo `.txt` por cada imagen.
6. Calcula una caja aproximada usando procesamiento de imagen con OpenCV.
7. Guarda las etiquetas en `dataset/labels/train` o `dataset/labels/val`.

Comando usado para regenerar todo el dataset:

```cmd
python scripts\prepare_auto_labels.py --clean-output
```

La opcion `--clean-output` limpia las imagenes y etiquetas anteriores antes de generar el nuevo dataset.

## 8. Resultado del nuevo etiquetado

Despues de agregar mas imagenes y regenerar el dataset, el resultado fue:

```text
train: 2359 imagenes
val:   1709 imagenes
total: 4068 imagenes
```

Se saltaron 4 imagenes porque no se pudieron leer correctamente:

```text
Audifonos Computador\Train\1802-Audifonos-De-Cable-Con-Microfono-Para-Celular-y-Computa.jpg
Audifonos Computador\Train\1804-Audifonos-De-Cable-Con-Microfono-Para-Celular-y-Computa.jpg
Monitor Computadora\Val\PFGE2_2.jpg
Mouse Computadora\Train\2842-Kit-teclado-y-Mouse-Raton-Bluetooth-para-PC-Tablet-Celu.jpg
```

## 9. Como verificar que el dataset esta correcto

Se creo el script:

```text
scripts/check_dataset.py
```

Este script revisa:

- Que existan las imagenes.
- Que existan las etiquetas.
- Que cada etiqueta tenga 5 valores.
- Que la clase exista.
- Que las coordenadas esten entre `0` y `1`.

Comando:

```cmd
python scripts\check_dataset.py
```

Resultado esperado:

```text
Dataset listo para entrenar.
```

## 10. Vista previa del etiquetado

Se creo el script:

```text
scripts/make_label_preview.py
```

Este script genera una imagen de ejemplo con cajas dibujadas para revisar visualmente el etiquetado.

Comando:

```cmd
python scripts\make_label_preview.py
```

Archivo generado:

```text
dataset/preview_auto_labels.jpg
```

Para abrirlo:

```cmd
explorer dataset
```

## 11. Instalacion del entorno en Windows

Entrar al proyecto:

```cmd
cd C:\Users\ASUS\Documents\Codex\yolo-perifericos-raspberry
```

Crear entorno virtual:

```cmd
python -m venv .venv
```

Activar entorno:

```cmd
.venv\Scripts\activate
```

Instalar dependencias:

```cmd
pip install -r requirements.txt
```

Si falta alguna libreria:

```cmd
python -m pip install ultralytics opencv-python numpy PyYAML
```

## 12. Entrenamiento del modelo

Se creo el script:

```text
scripts/train.py
```

Este script usa `ultralytics` para entrenar YOLO con el dataset configurado.

Comando para entrenar la version mejorada:

```cmd
python scripts\train.py --model yolo11n.pt --epochs 70 --imgsz 640 --name perifericos_yolo_mejorado
```

Explicacion:

| Parametro | Significado |
| --- | --- |
| `--model yolo11n.pt` | Modelo base pequeño de YOLO |
| `--epochs 70` | Entrena durante 70 epocas |
| `--imgsz 640` | Tamano de imagen usado por YOLO |
| `--name perifericos_yolo_mejorado` | Nombre de la carpeta de resultados |

Cuando termina, el modelo entrenado queda en:

```text
runs/detect/perifericos_yolo_mejorado/weights/best.pt
```

El archivo `best.pt` es el modelo final que se usa para detectar objetos.

## 13. Como reanudar un entrenamiento

Si se apaga el computador o se detiene el entrenamiento, YOLO guarda un archivo:

```text
last.pt
```

Para reanudar:

```cmd
python scripts\train.py --model runs\detect\perifericos_yolo_mejorado\weights\last.pt --epochs 70 --imgsz 640
```

## 14. Validacion del entrenamiento

Durante el entrenamiento YOLO genera metricas en:

```text
runs/detect/perifericos_yolo_mejorado
```

Archivos importantes:

```text
results.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch0_pred.jpg
val_batch1_pred.jpg
val_batch2_pred.jpg
```

Para abrir la carpeta:

```cmd
explorer runs\detect\perifericos_yolo_mejorado
```

### Metricas principales

| Metrica | Que significa |
| --- | --- |
| Precision | De lo que detecto, cuanto fue correcto |
| Recall | De los objetos reales, cuantos encontro |
| mAP50 | Calidad de deteccion con criterio IoU 0.50 |
| mAP50-95 | Metrica mas exigente, evalua varios niveles de IoU |

## 15. Probar deteccion con imagenes

Se creo el script:

```text
scripts/detect.py
```

Para probar con las imagenes de validacion:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source dataset\images\val --save
```

Los resultados se guardan en:

```text
runs/detect/predict
```

Si ya existe `predict`, YOLO puede crear:

```text
predict-2
predict-3
predict-4
```

Para abrir resultados:

```cmd
explorer runs\detect\predict
```

Si la carpeta se llama `predict-2`:

```cmd
explorer runs\detect\predict-2
```

## 16. Probar deteccion con camara

Para abrir la camara:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source 0 --show --conf 0.65
```

Si la camara no abre:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source 1 --show --conf 0.65
```

El parametro `--conf 0.65` significa que solo muestra detecciones con confianza mayor o igual a 65%.

Para exigir mas confianza:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source 0 --show --conf 0.80
```

## 17. Problema encontrado durante la prueba con camara

En una prueba en vivo, el modelo detecto un objeto como `microfono` y dibujo una caja demasiado grande.

Esto puede pasar por estas razones:

1. Varias etiquetas fueron automaticas y no manuales.
2. Algunas cajas quedaron demasiado grandes.
3. Muchas imagenes originales eran tipo producto o catalogo.
4. La camara real tiene fondos, iluminacion y posiciones diferentes.
5. Algunas clases se parecen entre si, por ejemplo `parlantes` y `microfono`.

La solucion fue agregar mas imagenes reales y regenerar el dataset con mas cantidad de datos.

## 18. Recomendaciones para mejorar precision

Para mejorar la deteccion real con camara:

- Usar fotos tomadas con la misma camara donde correra el proyecto.
- Tomar fotos con fondos reales.
- Incluir objetos en la mano.
- Incluir objetos sobre mesa.
- Usar diferentes distancias.
- Usar diferentes angulos.
- Usar buena iluminacion y baja iluminacion.
- Revisar manualmente las etiquetas mas malas.
- Evitar que la caja cubra toda la imagen si el objeto es pequeno.

## 19. Exportar para Raspberry Pi

Se creo el script:

```text
scripts/export_raspberry.py
```

Para Raspberry Pi se recomienda exportar a NCNN porque suele ser mas liviano y rapido.

Comando:

```cmd
python scripts\export_raspberry.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --format ncnn --imgsz 416
```

Esto genera un modelo exportado para usar en Raspberry Pi.

## 20. Instalar en Raspberry Pi

En Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0
```

Entrar al proyecto:

```bash
cd ~/yolo-perifericos-raspberry
```

Crear entorno:

```bash
python3 -m venv .venv
```

Activar entorno:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements-raspberry.txt
```

Ejecutar deteccion:

```bash
python scripts/detect.py --weights runs/detect/perifericos_yolo_mejorado/weights/best.pt --source 0 --imgsz 416 --conf 0.65
```

Si se usa el modelo NCNN exportado, se apunta a la carpeta exportada:

```bash
python scripts/detect.py --weights runs/detect/perifericos_yolo_mejorado/weights/best_ncnn_model --source 0 --imgsz 416 --conf 0.65
```

## 21. Internet: cuando se necesita y cuando no

Se necesita internet para:

- Instalar librerias con `pip`.
- Descargar `yolo11n.pt` la primera vez.
- Descargar dependencias adicionales de exportacion.

No se necesita internet para:

- Detectar objetos con `best.pt`.
- Usar la camara si el entorno ya esta instalado.
- Ejecutar el modelo en Raspberry Pi si ya se copiaron dependencias y modelo.

## 22. Comandos principales resumidos

Entrar al proyecto:

```cmd
cd C:\Users\ASUS\Documents\Codex\yolo-perifericos-raspberry
```

Activar entorno:

```cmd
.venv\Scripts\activate
```

Regenerar etiquetas:

```cmd
python scripts\prepare_auto_labels.py --clean-output
```

Verificar dataset:

```cmd
python scripts\check_dataset.py
```

Crear vista previa:

```cmd
python scripts\make_label_preview.py
```

Entrenar:

```cmd
python scripts\train.py --model yolo11n.pt --epochs 70 --imgsz 640 --name perifericos_yolo_mejorado
```

Detectar con imagenes de validacion:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source dataset\images\val --save
```

Detectar con camara:

```cmd
python scripts\detect.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --source 0 --show --conf 0.65
```

Exportar para Raspberry Pi:

```cmd
python scripts\export_raspberry.py --weights runs\detect\perifericos_yolo_mejorado\weights\best.pt --format ncnn --imgsz 416
```

## 23. Estado actual del proyecto

El proyecto ya tiene:

- Estructura YOLO creada.
- Dataset organizado.
- Etiquetas generadas automaticamente.
- Verificador de dataset.
- Entrenamiento funcional.
- Deteccion por imagen.
- Deteccion por camara.
- Script de exportacion para Raspberry Pi.
- Nueva version del dataset con 4068 imagenes etiquetadas.

El siguiente paso recomendado es entrenar la version mejorada y probarla nuevamente con camara real.

