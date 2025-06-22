<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rubik+Spray+Paint&display=swap" rel="stylesheet">

<div align="center" style="display:flex;justify-content: center; align-items: center;">
    <img src="logo1.png">
    <div class="text-gradient">NeuroFaceID</div>
    <img src="logo2.png">
</div>

##

### 🧠 NeuroFaceID — Нейросистема распознавания лиц на основе FaceNet и KNN
[![License](https://img.shields.io/github/license/XRomanchikX/NeuroFaceID)](LICENSE) 
[![Build Status](https://img.shields.io/github/actions/workflow/status/XRomanchikX/NeuroFaceID/ci.yml?branch=main)](https://github.com/XRomanchikX/NeuroFaceID/actions)
[![Coverage](https://img.shields.io/codecov/c/github/XRomanchikX/NeuroFaceID)](https://codecov.io/gh/XRomanchikX/NeuroFaceID)
[![Python Version](https://img.shields.io/badge/Python-3.12.7-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/neurofaceid)](https://pypi.org/project/neurofaceid/)

##

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
- Датасет изображений лиц в формате lfw

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

 ```bash
 python main.py
 ```

 Для тестирования - были загружены 2 фото (`testimage1.jpg` и `testimage2.jpg`) и датасет 
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

Nearst image's:
Filename: testdataset/6/12.jpg, Class: 6, Distance: 0.0647
Filename: testdataset/6/0.jpg, Class: 6, Distance: 0.0840
Filename: testdataset/6/13.jpg, Class: 6, Distance: 0.0896
```
#

### 🛠 Проект находится исключительно в релизе! Идёт процесс разработки.
- Следующее обновление запланированно на 24.06.25

<style>
.text-gradient {
    color: #1C6FFF;
    background-image: linear-gradient(180deg, #1C6FFF 3%, #C821FF 100%); 
    background-clip: text; 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    font-family: "Rubik Spray Paint", system-ui;
    font-size: 48px;
    font-weight: 400;
    font-style: normal
}
</style>