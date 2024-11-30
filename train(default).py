from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')  # Cargar modelo YOLOv8s preentrenado
    results = model.train(
        data='E:/Documents/Tesis 2/Vision Python/dataset (new)/data.yaml',  # Ruta a tu archivo de configuración del dataset
        epochs=100,  # Aumentar si deseas más refinamiento
        batch=16,  # Mantener fijo para la GPU RTX 2060 de 6GB
        imgsz=640,  # Mantener fijo el tamaño de imagen

        project='runs/detect',  # Carpeta para almacenar resultados
        name='yolov8s_training_last',  # Nombre del experimento

        save=True,  # Guardar los resultados del entrenamiento
        cache=False,  # Cacheo de datos desactivado
        device='cuda',  # Usar GPU (cuda) si está disponible
        workers=4,  # Número de trabajadores para carga de datos
        amp=False,  # Activar precisión mixta (mixed precision)
        optimizer='AdamW',  # Optimizador AdamW para mejor convergencia
        lr0=0.005,  # Tasa de aprendizaje inicial
        weight_decay=0.0001,  # Decaimiento de peso para evitar sobreajuste
        cos_lr=True,  # Usar el scheduler de tipo 'cosine'
    )
    print(results)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()