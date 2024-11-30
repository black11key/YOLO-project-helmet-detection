from ultralytics import YOLO
import cv2
import time
import os


def evaluate_latency(model_weights_path, test_data_folder, num_repeats=30):
    model = YOLO(model_weights_path)
    total_times = []  # Lista para almacenar los tiempos totales de cada repetición
    read_times = []  # Lista para almacenar los tiempos de lectura de imágenes
    inference_times = []  # Lista para almacenar los tiempos de inferencia de cada imagen

    for repeat in range(num_repeats):
        start_time = time.perf_counter()  # Tiempo de inicio para esta repetición
        total_read_time = 0  # Acumulador del tiempo total de lectura en esta repetición
        total_inference_time = 0  # Acumulador del tiempo total de inferencia en esta repetición

        for filename in os.listdir(test_data_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(test_data_folder, filename)
                print(f"Procesando imagen: {image_path}")

                # Medir el tiempo de lectura de la imagen
                read_start_time = time.perf_counter()
                image = cv2.imread(image_path)
                if image is None:
                    print(f"No se pudo cargar la imagen: {image_path}")
                    continue
                read_end_time = time.perf_counter()
                read_time = read_end_time - read_start_time
                total_read_time += read_time

                # Realizar inferencia en la imagen
                inference_start_time = time.perf_counter()
                results = model.predict(image)
                inference_end_time = time.perf_counter()

                inference_time = inference_end_time - inference_start_time
                total_inference_time += inference_time  # Acumulando el tiempo de inferencia

        # Calcular el tiempo total para esta repetición (lectura + inferencia)
        end_time = time.perf_counter()  # Tiempo de fin para esta repetición
        total_time = end_time - start_time
        total_times.append(total_time)
        read_times.append(total_read_time)
        inference_times.append(total_inference_time)

        print(f"Repetición {repeat + 1}: Tiempo total para todas las imágenes = {total_time:.4f} segundos")
        print(f"Tiempo total de lectura de imágenes en esta repetición: {total_read_time:.4f} segundos")
        print(f"Tiempo total de inferencia en esta repetición: {total_inference_time:.4f} segundos")

    avg_total_time = sum(total_times) / len(total_times) if total_times else 0
    avg_read_time = sum(read_times) / len(read_times) if read_times else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

    print(f"Tiempo promedio total para todas las imágenes: {avg_total_time:.4f} segundos")
    print(f"Tiempo promedio total de lectura de imágenes: {avg_read_time:.4f} segundos")
    print(f"Tiempo promedio total de inferencia: {avg_inference_time:.4f} segundos")


if __name__ == '__main__':
    model_weights_path = 'E:/Documents/Tesis 2/Vision Python/runs/detect/yolov5nu_training2/weights/best.pt'  # Ruta a los mejores pesos del modelo
    test_data_folder = 'E:/Documents/Tesis 2/Vision Python/dataset (yolov5 pytorch)/test/images'  # Ruta a la carpeta de test con imágenes

    evaluate_latency(model_weights_path, test_data_folder)  # No es necesario especificar num_repeats ahora

