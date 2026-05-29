# YOLO Perifericos Raspberry

Proyecto base en Python para entrenar y ejecutar un detector YOLO de 10 perifericos de computador, pensado para correr luego en Raspberry Pi.

## Documentacion completa

Para ver el paso a paso completo de lo realizado, tecnologias usadas, estructura, etiquetado, entrenamiento, validacion, camara y Raspberry Pi, abre:

```text
README_PROYECTO.md
```

## Clases del modelo

El proyecto viene configurado para estas 10 clases:

1. teclado
2. mouse
3. monitor
4. impresora
5. parlantes
6. webcam
7. microfono
8. audifonos
9. memoria_usb
10. router

Puedes cambiar los nombres en `data/perifericos.yaml`, pero si ya empezaste a etiquetar imagenes conviene no cambiar el orden.

## Estructura

```text
yolo-perifericos-raspberry/
  data/
    perifericos.yaml
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/
  models/
  runs/
  scripts/
    capture_images.py
    check_dataset.py
    train.py
    detect.py
    export_raspberry.py
  requirements.txt
  requirements-raspberry.txt
```

## 1. Crear entorno en Windows o PC de entrenamiento

```powershell
cd C:\Users\ASUS\Documents\Codex\yolo-perifericos-raspberry
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Recolectar imagenes

Puedes capturar fotos con una webcam:

```powershell
python scripts/capture_images.py --class-name teclado --camera 0
```

Presiona:

- `c` para capturar una imagen.
- `q` para salir.

Repite con cada clase, por ejemplo:

```powershell
python scripts/capture_images.py --class-name mouse
python scripts/capture_images.py --class-name monitor
```

Despues etiqueta las imagenes en formato YOLO usando una herramienta como Label Studio, CVAT o Roboflow. Las etiquetas deben quedar asi:

```text
dataset/images/train/*.jpg
dataset/images/val/*.jpg
dataset/labels/train/*.txt
dataset/labels/val/*.txt
```

Cada archivo `.txt` debe tener lineas con este formato:

```text
class_id x_center y_center width height
```

Los valores de coordenadas deben estar normalizados de 0 a 1.

## 3. Revisar el dataset

```powershell
python scripts/check_dataset.py
```

## 4. Entrenar YOLO

Para empezar rapido:

```powershell
python scripts/train.py --model yolo11n.pt --epochs 50 --imgsz 640
```

El mejor modelo queda normalmente en:

```text
runs/detect/perifericos_yolo/weights/best.pt
```

## 5. Probar deteccion

Con camara:

```powershell
python scripts/detect.py --weights runs/detect/perifericos_yolo/weights/best.pt --source 0
```

Con imagen:

```powershell
python scripts/detect.py --weights runs/detect/perifericos_yolo/weights/best.pt --source ruta/a/imagen.jpg
```

## 6. Exportar para Raspberry Pi

Para Raspberry Pi suele ir mejor exportar a NCNN:

```powershell
python scripts/export_raspberry.py --weights runs/detect/perifericos_yolo/weights/best.pt --format ncnn
```

Tambien puedes exportar a ONNX:

```powershell
python scripts/export_raspberry.py --weights runs/detect/perifericos_yolo/weights/best.pt --format onnx
```

## 7. Instalar en Raspberry Pi

En la Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0
cd ~/yolo-perifericos-raspberry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-raspberry.txt
```

Ejecutar deteccion:

```bash
python scripts/detect.py --weights runs/detect/perifericos_yolo/weights/best.pt --source 0 --imgsz 416
```

Si usas modelo exportado NCNN, apunta `--weights` a la carpeta exportada, por ejemplo:

```bash
python scripts/detect.py --weights runs/detect/perifericos_yolo/weights/best_ncnn_model --source 0 --imgsz 416
```

## Consejos para buen resultado

- Usa minimo 100 imagenes por clase para una primera version decente.
- Toma fotos con diferentes fondos, luces, angulos y distancias.
- Mezcla imagenes con uno y varios objetos.
- Mantén `val` separado de `train`; no repitas las mismas fotos.
- Para Raspberry Pi empieza con `yolo11n.pt`, `imgsz 416` o `imgsz 320` si va lento.
