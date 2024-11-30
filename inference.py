from ultralytics import YOLO
import os
# Cargar el modelo preentrenado de YOLO
model = YOLO('runs/detect/yolov8s_training_last/weights/best.pt')  # Cambia esta ruta por el modelo que estás utilizando

# Ruta de las imágenes de prueba
input_folder = "data for prediction/Video 1.v1i.yolov9/test/images"
output_folder = "VIDEO RESULTS/video 1 (yolov8s)"  # Carpeta donde se guardarán los resultados

# Crear la carpeta de resultados si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos de imagen en la carpeta de entrada
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Realizar inferencia sobre cada imagen
for image_file in image_files:
    # Ruta completa de la imagen
    image_path = os.path.join(input_folder, image_file)

    # Realizar la inferencia
    results = model.predict(image_path, conf=0.25)

    # Obtener el primer resultado (en caso de que results sea una lista)
    result = results[0]

    # Ruta para guardar la imagen con las bounding boxes
    output_image_path = os.path.join(output_folder, image_file)

    # Guardar la imagen con las bounding boxes
    result.save(output_image_path)

    print(f"Resultado guardado: {output_image_path}")

print("Inferencias completadas y resultados guardados.")

