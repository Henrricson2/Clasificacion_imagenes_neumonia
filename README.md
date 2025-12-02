# Clasificación de Imágenes Médicas: Neumonía en Radiografías de Tórax

## Descripción del Proyecto
Este proyecto implementa y compara dos enfoques para la clasificación de neumonía en radiografías de tórax:
1. **Descriptores clásicos** (handcrafted features) con clasificadores tradicionales
2. **Redes neuronales convolucionales** (Deep Learning)

##  Objetivo
Desarrollar un pipeline completo de clasificación de imágenes médicas, explorando conceptos de visión por computadora y verificando su efectividad en un problema real.

## Dataset
- **Fuente**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Clases**: Normal vs Pneumonia
- **Tipo**: Radiografías de tórax

## Estructura del Proyecto
```
proyecto_clasificacion_neumonia/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
├── data/                        # Datos del proyecto
│   ├── raw/                     # Datos originales
│   └── processed/               # Datos preprocesados
├── src/                         # Código fuente
│   ├── preprocessing.py         # Preprocesamiento de imágenes
│   ├── feature_extraction.py   # Extracción de descriptores
│   ├── classification.py       # Clasificadores
│   └── utils.py                # Utilidades
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_classification.ipynb
└── results/                     # Resultados y visualizaciones
    ├── figures/                 # Gráficos y visualizaciones
    └── models/                  # Modelos entrenados
```

## Cómo ejecutar el proyecto

## 1.1 Crear entorno virtual 

python3 -m venv venv
source venv/bin/activate

### 1. Instalación de dependencias
```bash
pip install -r requirements.txt
```

### 2. Descargar los datos
1. Descargar el dataset de [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extraer en la carpeta `data/raw/`

### 3. Ejecutar el análisis
1. **Análisis exploratorio**: `notebooks/01_exploratory_analysis.ipynb`
2. **Extracción de características**: `notebooks/02_feature_extraction.ipynb`
3. **Clasificación**: `notebooks/03_classification.ipynb`


## Metodología

### Parte 1: Análisis Exploratorio y Preprocesamiento
- [ ] Carga y visualización del dataset
- [ ] Análisis de distribución de clases
- [ ] Pipeline de preprocesamiento (normalización, CLAHE)

### Parte 2: Extracción de Descriptores
#### Descriptores de Forma
- [ ] Histogram of Oriented Gradients (HOG)
- [ ] Momentos de Hu
- [ ] Descriptores de contorno

#### Descriptores de Textura
- [ ] Local Binary Patterns (LBP)
- [ ] Gray Level Co-occurrence Matrix (GLCM)
- [ ] Filtros de Gabor

### Parte 3: Clasificación
- [ ] SVM con diferentes kernels
- [ ] Random Forest
- [ ] k-Nearest Neighbors
- [ ] Regresión Logística
- [ ] Red Neuronal Convolucional

##  Métricas de Evaluación
- Accuracy, Precision, Recall, F1-Score
- Matriz de confusión
- Curva ROC y AUC
- Validación cruzada

## Equipo de Trabajo
- Laura Sanín Colorado
- Henrry Uribe Cabrera Ordoñez
- Sebastián Palacio Betancur
- Juan Manuel Sánchez Restrepo

## Referencias
- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---
**Curso**: Computer Vision - Maestría  
**Profesor**: Juan David Ospina Arango  
**Monitor**: Andrés Mauricio Zapata