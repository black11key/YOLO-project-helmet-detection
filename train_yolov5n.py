from ultralytics import YOLO

def main():
    # Cargar el modelo YOLOv5n preentrenado
    model = YOLO('yolov5nu.pt')  # Asegúrate de que este modelo esté disponible

    # Iniciar el entrenamiento
    results = model.train(
        data='E:/Documents/Tesis 2/Vision Python/dataset (yolov5 pytorch)/data.yaml',  # Ruta correcta al archivo data.yaml
        epochs=100,
        batch=16,
        imgsz=640,
        project='runs/detect',
        name='yolov5nu_training',
        save=True,
        cache=False,
        device='cuda',  # Asegúrate de que tu GPU esté disponible
        workers=4,
        amp=False,
        optimizer='Adam',
    )
    print(results)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Para soporte en Windows
    main()