<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rubik+Spray+Paint&display=swap" rel="stylesheet">

<div align="center" style="display:flex;justify-content: center; align-items: center;">
    <img src="logo1.png">
    <img src="logo.png">
    <img src="logo2.png">
</div>

##

<div align="center" style="display:flex;justify-content: center; align-items: center;">

### 🧠 NeuroFaceID — Нейросистема распознавания лиц на основе FaceNet и KNN
[![License](https://img.shields.io/github/license/XRomanchikX/NeuroFaceID)](LICENSE) 
[![Build Status](https://img.shields.io/github/actions/workflow/status/XRomanchikX/NeuroFaceID/ci.yml?branch=main)](https://github.com/XRomanchikX/NeuroFaceID/actions)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/neurofaceid)](https://pypi.org/project/neurofaceid/)
##

</div>

### 🔍 Описание:
NeuroFaceID — это система распознавания лиц, реализованная с использованием:

- Максимально точной модели `Vggface2` (Точность: 90-95%)

- `MTCNN` для обнаружения лиц на изображении

- `InceptionResnetV1` (FaceNet) для генерации векторных эмбеддингов лиц

- `KNN` (k-ближайших соседей) для поиска схожих лиц по косинусному расстоянию
##

### 🛠️ Требования:
- Python 3.12+
- CUDA-совместимая видеокарта (рекомендуется)
- Минимум 8 ГБ ОЗУ
- Датасет изображений лиц в формате: png, jpg, jpeg

#

### 📦 Установка:

**Linux**:
```bash
git clone https://github.com/XRomanchikX/NeuroFaceID && cd NeuroFaceID && pip install -r requirements.txt
```

**Windows**:
```cmd
git clone https://github.com/XRomanchikX/NeuroFaceID && cd NeuroFaceID && pip install -r requirements.txt
```
#

### 🧪 Запуск:

Необходимо указать название папки в переменную - `dataset_dir`

Пример структуры каталога:

```
testdataset/ ### Формат изображения - png, jpg, jpeg.
    0/
        1.png 
        2.jpg
        3.jpeg
        ...
    1/
        1.jpg
        2.jpeg
        3.png
        ...
    ...
```

### Запуск:

 ```bash
 python main.py
 ```
#


### 📊 Архитектура системы:
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ MTCNN        │     │ FaceNet      │     │ KNN          │
│ Обнаружение  ├───► │ Эмбеддинги   ├───► │ Классификация│
└──────────────┘     └──────────────┘     └──────────────┘
```

#

### 🪶 Пример вывода:
```
dusty@archlinux ~/NeuroFaceID > python main.py 

The nearst image to: "testimage1.jpeg":
Filename: testdataset/1/2.png, Class: 1, Distance: 0.0647
Filename: testdataset/1/0.jpg, Class: 1, Distance: 0.0840
Filename: testdataset/1/5.jpeg, Class: 1, Distance: 0.0896
```
#

### 🛠 Проект находится исключительно в релизе! Идёт процесс разработки.
- Следующее обновление запланированно на 26.06.25
