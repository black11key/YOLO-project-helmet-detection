from ultralytics import YOLO

def main():
    model = YOLO('yolov5su.pt')  # Cargar modelo YOLOv5n preentrenado
    results = model.train(
        data='E:/Documents/Tesis 2/Vision Python/dataset (yolov5 pytorch)/data.yaml', # Ruta a tu archivo de configuración del dataset
        epochs=100,  # Número de épocas (ajusta según tus necesidades)
        batch=16,  # Tamaño de lote (ajusta según la capacidad de tu GPU)
        imgsz=640,  # Tamaño de imagen (ajustable según el modelo y hardware)
        project='runs/detect',  # Carpeta donde se almacenan los resultados del entrenamiento
        name='yolov5su_training',  # Nombre del experimento para los resultados
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

    multiprocessing.freeze_support()
    main()