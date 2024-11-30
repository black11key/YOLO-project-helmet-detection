from ultralytics import YOLO

def main():
    model = YOLO('yolov9t.pt')  # Load a pretrained YOLOv8 model
    results = model.train(
        data='E:/Documents/Tesis 2/Vision Python/dataset (yolov9)/data.yaml',
        epochs=100,  # Número de epochs para entrenamiento
        batch=16,  # Tamaño del batch
        imgsz=640,  # Tamaño de las imágenes (640x640)

        project='runs/detect',  # Carpeta para almacenar resultados
        name='yolov9t_training_last',  # Nombre del experimento

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
    multiprocessing.freeze_support()  # For frozen applications like py2exe or pyinstaller
    main()