import io
import os
from PIL import Image, ImageDraw
from ML.load import load_model

model = None

def load_ml():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

def save_image(image: bytes, file_name: str):
    try:
        img = Image.open(io.BytesIO(image))
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return None
    
    try:
        if not os.path.exists("savedIMG"):
            os.makedirs("savedIMG")
        path = f'savedIMG/{file_name}'
        img.save(path)
        return path
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        return None

def save_predict_image(image: bytes, file_name: str):
    try:
        img = Image.open(io.BytesIO(image))
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return None

    try:
        if not os.path.exists("predictedIMG"):
            os.makedirs("predictedIMG")
        path = f'predictedIMG/{file_name}'
        img.save(path)
        return path
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        return None

def get_result_from_ml(path):
    if model is None:
        print("Модель не загружена. Пожалуйста, загрузите модель перед выполнением предсказаний.")
        return None
    
    try:
        results = model.predict(source=path, conf=0.5)
        res = {}
        for r in results:
            boxes = r.boxes.data.tolist()
            for box in boxes:
                res["xmin"] = box[0]
                res["ymin"] = box[1]
                res["xmax"] = box[2]
                res["ymax"] = box[3]
                res["confidence"] = box[4]
                res["class"] = box[5]
                res["class_name"] = model.names[int(box[5])]
        return res
    except Exception as e:
        print(f"Ошибка при получении результатов из модели: {e}")
        return None

def highlight_damage(image_path: str, results: dict):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return None

    try:
        draw = ImageDraw.Draw(img)

        xmin = results.get("xmin")
        ymin = results.get("ymin")
        xmax = results.get("xmax")
        ymax = results.get("ymax")
        class_name = results.get("class_name")
        confidence = results.get("confidence")

        if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
            text = f"{class_name} ({confidence:.2f})"
            draw.text((xmin, ymin - 10), text, fill="red")
    except Exception as e:
        print(f"Ошибка при отрисовке: {e}")
        return None

    try:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()
    except Exception as e:
        print(f"Ошибка при попытке записи изображения в байты: {e}")
        return None