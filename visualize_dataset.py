import os
import cv2
import argparse
from typing import List, Dict, Tuple

def parse_annotations(gt_file: str) -> Dict[int, List[Dict]]:
    """
    Парсит файл аннотаций и возвращает словарь с аннотациями для каждого кадра.
    
    :param gt_file: Путь к файлу gt.txt
    :return: Словарь, где ключ - frame_id, а значение - список аннотаций
    """
    annotations = {}
    with open(gt_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue  # Пропустить некорректные строки
            try:
                frame_id = int(parts[0])
                object_id = int(parts[1])
                bb_left = int(float(parts[2]))
                bb_top = int(float(parts[3]))
                bb_width = int(float(parts[4]))
                bb_height = int(float(parts[5]))
                confidence = float(parts[6])
                # Остальные части строки игнорируются
            except ValueError:
                continue  # Пропустить строки с некорректными значениями

            bbox = (bb_left, bb_top, bb_left + bb_width, bb_top + bb_height)

            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append({
                'object_id': object_id,
                'bbox': bbox,
                'confidence': confidence
            })
    return annotations

def load_frames(frames_dir: str) -> Dict[int, str]:
    """
    Загружает пути к кадрам из директории.

    :param frames_dir: Путь к директории с кадрами
    :return: Словарь, где ключ - frame_id, а значение - путь к изображению
    """
    frames = {}
    for filename in os.listdir(frames_dir):
        if filename.startswith('frame_') and filename.endswith('.jpg'):
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) != 2:
                continue
            try:
                frame_id = int(parts[1])
                frames[frame_id] = os.path.join(frames_dir, filename)
            except ValueError:
                continue
    return frames

def visualize_frames(frames: Dict[int, str], annotations: Dict[int, List[Dict]]):
    """
    Визуализирует кадры с аннотациями.

    :param frames: Словарь с путями к кадрам
    :param annotations: Словарь с аннотациями для каждого кадра
    """
    sorted_frame_ids = sorted(frames.keys())

    for frame_id in sorted_frame_ids:
        frame_path = frames[frame_id]
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Не удалось загрузить изображение: {frame_path}")
            continue

        if frame_id in annotations:
            for anno in annotations[frame_id]:
                object_id = anno['object_id']
                bbox = anno['bbox']
                confidence = anno['confidence']

                # Рисуем толстый черный прямоугольник для обводки
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 4)
                # Рисуем тонкий синий прямоугольник поверх
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # Добавляем текст с ID объекта
                text = f"ID: {object_id}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 4), 
                                      (bbox[0] + text_width, bbox[1]), (255, 0, 0), -1)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Отображаем кадр
        cv2.imshow('Visualization', frame)
        key = cv2.waitKey(0)  # Ожидание нажатия клавиши для перехода к следующему кадру
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def save_annotated_frames(frames: Dict[int, str], annotations: Dict[int, List[Dict]], output_dir: str):
    """
    Сохраняет аннотированные кадры в указанную директорию.

    :param frames: Словарь с путями к кадрам
    :param annotations: Словарь с аннотациями для каждого кадра
    :param output_dir: Путь к директории для сохранения аннотированных кадров
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_frame_ids = sorted(frames.keys())

    for frame_id in sorted_frame_ids:
        frame_path = frames[frame_id]
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Не удалось загрузить изображение: {frame_path}")
            continue

        if frame_id in annotations:
            for anno in annotations[frame_id]:
                object_id = anno['object_id']
                bbox = anno['bbox']
                confidence = anno['confidence']

                # Рисуем толстый черный прямоугольник для обводки
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 4)
                # Рисуем тонкий синий прямоугольник поверх
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # Добавляем текст с ID объекта
                text = f"ID: {object_id}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 4), 
                                      (bbox[0] + text_width, bbox[1]), (255, 0, 0), -1)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Сохранение аннотированного кадра
        output_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Сохранен аннотированный кадр: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Визуализация аннотированного датасета с использованием OpenCV.")
    parser.add_argument('--frames_dir', type=str, required=True,
                        help="Путь к директории с кадрами.")
    parser.add_argument('--gt_file', type=str, required=True,
                        help="Путь к файлу аннотаций (gt.txt).")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Путь к директории для сохранения аннотированных кадров. Если не указано, производится только визуализация.")
    parser.add_argument('--mode', type=str, choices=['visualize', 'save', 'both'], default='visualize',
                        help="Режим работы скрипта: 'visualize' для отображения кадров, 'save' для сохранения аннотированных кадров, 'both' для выполнения и того и другого.")
    args = parser.parse_args()

    frames_dir = args.frames_dir
    gt_file = args.gt_file
    output_dir = args.output_dir
    mode = args.mode

    # Проверка существования входных путей
    if not os.path.isdir(frames_dir):
        print(f"Ошибка: Директория с кадрами не существует: {frames_dir}")
        return
    if not os.path.isfile(gt_file):
        print(f"Ошибка: Файл аннотаций не существует: {gt_file}")
        return
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Ошибка при создании директории для сохранения: {output_dir}\n{e}")
            return

    # Парсинг аннотаций
    annotations = parse_annotations(gt_file)
    print(f"Загружено аннотаций для {len(annotations)} кадров.")

    # Загрузка путей к кадрам
    frames = load_frames(frames_dir)
    print(f"Найдено {len(frames)} кадров.")
    # Выполнение действий в зависимости от режима
    if mode in ['visualize', 'both']:
        print("Начинается визуализация аннотированных кадров...")
        visualize_frames(frames, annotations)
    
    if mode in ['save', 'both']:
        if not output_dir:
            print("Ошибка: Для режима 'save' необходимо указать --output_dir.")
            return
        print(f"Сохранение аннотированных кадров в директорию: {output_dir}")
        save_annotated_frames(frames, annotations, output_dir)

    print("Завершено.")

if __name__ == "__main__":
    main()
