# SAEM-sistema-de-Epidemologia


Este repositorio contiene un sistema completo para análisis epidemiológico que combina modelos matemáticos tradicionales (SIR, SEIR) con técnicas de aprendizaje profundo (CNN) para el análisis de imágenes médicas.

## Estructura del Proyecto

```
sistema-epidemiologico/
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── epidemic_cnn.py          # Modelo CNN para análisis de imágenes
│   │   ├── epidemic_models.py       # Modelos epidemiológicos (SIR, SEIR)
│   │   └── data_processor.py        # Procesamiento de datos e imágenes
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_interface.py        # Interfaz principal
│   │   ├── sir_tab.py               # Componentes de la pestaña SIR
│   │   ├── seir_tab.py              # Componentes de la pestaña SEIR
│   │   └── image_analysis_tab.py    # Componentes de análisis de imágenes
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_generator.py        # Generador de datos sintéticos
│       ├── visualization.py         # Funciones para visualización
│       └── logger.py                # Configuración de logging
│
├── data/
│   ├── casos/                       # Datos de casos epidémicos
│   ├── imagenes/                    # Imágenes para análisis CNN
│   ├── demograficos/                # Datos demográficos
│   └── modelos/                     # Modelos entrenados
│
├── notebooks/                       # Jupyter notebooks para análisis
│   ├── sir_model_exploration.ipynb
│   ├── seir_model_exploration.ipynb
│   └── cnn_training.ipynb
│
├── docs/                            # Documentación
│   ├── usage.md
│   ├── models.md
│   └── api.md
│
├── tests/                           # Tests unitarios e integración
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data_processor.py
│   └── test_utils.py
│
├── requirements.txt                 # Dependencias del proyecto
├── setup.py                         # Script de instalación
├── LICENSE                          # Licencia del proyecto
└── README.md                        # Documentación principal
```

## Instrucciones para Configurar el Repositorio Git

### 1. Inicializar el Repositorio

```bash
# Crear la estructura de directorios
mkdir -p sistema-epidemiologico/src/{models,ui,utils} sistema-epidemiologico/data/{casos,imagenes,demograficos,modelos} sistema-epidemiologico/notebooks sistema-epidemiologico/docs sistema-epidemiologico/tests

# Navegar al directorio del proyecto
cd sistema-epidemiologico

# Inicializar el repositorio Git
git init
```

### 2. Crear un archivo .gitignore

Crea un archivo `.gitignore` para excluir archivos innecesarios del control de versiones:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Datos
data/imagenes/*.png
data/imagenes/*.jpg
data/modelos/*.pt
data/modelos/*.pth

# Logs
*.log
logs/

# Entorno virtual
venv/
env/
ENV/
.env

# IDEs y editores
.idea/
.vscode/
*.swp
*.swo
*.swn
*~

# Sistema operativo
.DS_Store
Thumbs.db
```

### 3. Copiar los Archivos del Proyecto

Copia los siguientes archivos principales:

1. `src/models/epidemic_cnn.py`: La clase EpidemicCNN y relacionadas
2. `src/models/epidemic_models.py`: La clase ModelosEpidemiologicos
3. `src/models/data_processor.py`: La clase ProcesadorDatos
4. `src/ui/main_interface.py`: La clase InterfazEpidemiologia
5. `src/utils/data_generator.py`: El script de generación de datos
6. `setup.py`: Para la instalación del paquete

### 4. Crear el README.md

Usa este README.md como punto de partida:

```markdown
# Sistema de Análisis Epidemiológico con CNN

Un sistema integral para el análisis epidemiológico que combina modelos matemáticos clásicos (SIR, SEIR) con técnicas de aprendizaje profundo (CNN) para análisis de imágenes médicas.

## Características

- **Modelado Matemático**: Implementación de modelos SIR y SEIR para simulación de epidemias
- **Análisis de Imágenes**: Detección de patrones en imágenes mediante redes neuronales convolucionales
- **Interfaz Gráfica**: Visualización de resultados y configuración intuitiva de parámetros
- **Generación de Datos**: Herramientas para generar datos sintéticos para pruebas y entrenamiento

## Requisitos

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Tkinter

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/sistema-epidemiologico.git
cd sistema-epidemiologico

# Crear un entorno virtual (opcional)
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

## Uso

### Interfaz Gráfica

```bash
python -m src.ui.main_interface
```

### Generación de Datos Sintéticos

```bash
python -m src.utils.data_generator
```

## Documentación

Para más información sobre los modelos y la API, consulta la [documentación](docs/README.md).

## Contribuir

Las contribuciones son bienvenidas. Por favor, lee [CONTRIBUTING.md](CONTRIBUTING.md) para detalles sobre el proceso de pull request.

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
```

### 5. Crear el requirements.txt

```
numpy>=1.19.0
torch>=1.8.0
opencv-python>=4.5.0
matplotlib>=3.3.0
pandas>=1.1.0
scikit-learn>=0.24.0
Pillow>=8.0.0
```

### 6. Realizar el Primer Commit

```bash
# Añadir todos los archivos al staging
git add .

# Realizar el primer commit
git commit -m "Versión inicial del Sistema de Análisis Epidemiológico con CNN"
```

### 7. Conectar con un Repositorio Remoto (GitHub, GitLab, etc.)

```bash
# Para GitHub
git remote add origin https://github.com/tu-usuario/sistema-epidemiologico.git
git branch -M main
git push -u origin main
```

## Instrucciones para Ejecutar el Sistema

1. Primero, genera los datos sintéticos:
   ```bash
   python -m src.utils.data_generator
   ```

2. Luego, ejecuta la interfaz gráfica:
   ```bash
   python -m src.ui.main_interface
   ```

3. En la interfaz:
   - Carga los datos desde la carpeta `/data/casos/`
   - Configura los parámetros para los modelos SIR o SEIR
   - Ejecuta las simulaciones
   - Analiza imágenes desde la carpeta `/data/imagenes/`

## Desarrollo Futuro

Algunas áreas para expandir el proyecto:

1. Integración con bases de datos epidemiológicas reales
2. Implementación de modelos más complejos (ej. modelos por edad, red de contactos)
3. Mejora de la arquitectura CNN con modelos pre-entrenados
4. Incorporación de análisis espacial y geolocalización
5. Desarrollo de APIs para integración con otros sistemas
