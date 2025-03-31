import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import json
import threading
import queue

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("epidemiologia.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EpidemiologiaApp")

# Modelo CNN para análisis de imágenes médicas/epidemiológicas
class EpidemicCNN(nn.Module):
    def __init__(self, num_classes: int = 3, input_channels: int = 3):
        """
        Modelo CNN para análisis de imágenes relacionadas con epidemiología.
        
        Args:
            num_classes: Número de clases para clasificación (ej. niveles de severidad)
            input_channels: Número de canales de entrada (3 para RGB, 1 para escala de grises)
        """
        super(EpidemicCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Capas fully connected
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante del modelo."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Modelos matemáticos para epidemiología
class ModelosEpidemiologicos:
    def __init__(self):
        """Inicializa modelos matemáticos para análisis epidemiológico."""
        logger.info("Inicializando modelos epidemiológicos")
    
    @staticmethod
    def modelo_sir(
        poblacion: int, 
        susceptibles_init: int, 
        infectados_init: int, 
        recuperados_init: int, 
        beta: float, 
        gamma: float, 
        dias: int
    ) -> Dict[str, np.ndarray]:
        """
        Implementa el modelo SIR (Susceptible-Infected-Recovered) clásico.
        
        Args:
            poblacion: Tamaño total de la población
            susceptibles_init: Número inicial de personas susceptibles
            infectados_init: Número inicial de personas infectadas
            recuperados_init: Número inicial de personas recuperadas
            beta: Tasa de transmisión
            gamma: Tasa de recuperación
            dias: Número de días para la simulación
            
        Returns:
            Diccionario con los valores S, I, R para cada día
        """
        S = np.zeros(dias)
        I = np.zeros(dias)
        R = np.zeros(dias)
        
        # Condiciones iniciales
        S[0] = susceptibles_init
        I[0] = infectados_init
        R[0] = recuperados_init
        
        # Simulación del modelo
        for t in range(1, dias):
            # Nuevos infectados: beta * S * I / N
            nuevos_infectados = beta * S[t-1] * I[t-1] / poblacion
            # Nuevos recuperados: gamma * I
            nuevos_recuperados = gamma * I[t-1]
            
            S[t] = S[t-1] - nuevos_infectados
            I[t] = I[t-1] + nuevos_infectados - nuevos_recuperados
            R[t] = R[t-1] + nuevos_recuperados
        
        return {"S": S, "I": I, "R": R}
    
    @staticmethod
    def modelo_seir(
        poblacion: int, 
        susceptibles_init: int, 
        expuestos_init: int,
        infectados_init: int, 
        recuperados_init: int, 
        beta: float, 
        sigma: float,
        gamma: float, 
        dias: int
    ) -> Dict[str, np.ndarray]:
        """
        Implementa el modelo SEIR (Susceptible-Exposed-Infected-Recovered).
        
        Args:
            poblacion: Tamaño total de la población
            susceptibles_init: Número inicial de personas susceptibles
            expuestos_init: Número inicial de personas expuestas
            infectados_init: Número inicial de personas infectadas
            recuperados_init: Número inicial de personas recuperadas
            beta: Tasa de transmisión
            sigma: Tasa a la que los expuestos se vuelven infecciosos
            gamma: Tasa de recuperación
            dias: Número de días para la simulación
            
        Returns:
            Diccionario con los valores S, E, I, R para cada día
        """
        S = np.zeros(dias)
        E = np.zeros(dias)
        I = np.zeros(dias)
        R = np.zeros(dias)
        
        # Condiciones iniciales
        S[0] = susceptibles_init
        E[0] = expuestos_init
        I[0] = infectados_init
        R[0] = recuperados_init
        
        # Simulación del modelo
        for t in range(1, dias):
            # Nuevas exposiciones: beta * S * I / N
            nuevas_exposiciones = beta * S[t-1] * I[t-1] / poblacion
            # Nuevos infectados: sigma * E
            nuevos_infectados = sigma * E[t-1]
            # Nuevos recuperados: gamma * I
            nuevos_recuperados = gamma * I[t-1]
            
            S[t] = S[t-1] - nuevas_exposiciones
            E[t] = E[t-1] + nuevas_exposiciones - nuevos_infectados
            I[t] = I[t-1] + nuevos_infectados - nuevos_recuperados
            R[t] = R[t-1] + nuevos_recuperados
        
        return {"S": S, "E": E, "I": I, "R": R}
    
    @staticmethod
    def r0_efectivo(beta: float, gamma: float, porcentaje_susceptible: float) -> float:
        """
        Calcula el número reproductivo efectivo Re.
        
        Args:
            beta: Tasa de transmisión
            gamma: Tasa de recuperación
            porcentaje_susceptible: Porcentaje de la población susceptible
            
        Returns:
            Valor de Re
        """
        return (beta / gamma) * porcentaje_susceptible

    @staticmethod
    def crear_matriz_contacto(grupos_edad: int, datos_contacto: List[List[float]]) -> np.ndarray:
        """
        Crea una matriz de contacto para modelos estratificados por edad.
        
        Args:
            grupos_edad: Número de grupos de edad
            datos_contacto: Matriz con datos de contacto entre grupos
            
        Returns:
            Matriz de contacto numpy
        """
        return np.array(datos_contacto)

# Clase para procesamiento de datos e imágenes
class ProcesadorDatos:
    def __init__(self, device: str = "cpu"):
        """
        Inicializa el procesador de datos e imágenes.
        
        Args:
            device: Dispositivo para cálculos ("cpu" o "cuda")
        """
        self.device = device
        self.modelo_cnn = None
        
    def cargar_modelo(self, ruta_modelo: str) -> None:
        """
        Carga un modelo CNN pre-entrenado desde un archivo.
        
        Args:
            ruta_modelo: Ruta al archivo del modelo
        """
        try:
            self.modelo_cnn = torch.load(ruta_modelo, map_location=self.device)
            self.modelo_cnn.eval()
            logger.info(f"Modelo cargado desde {ruta_modelo}")
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            raise
    
    def preprocesar_imagen(self, imagen_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Preprocesa una imagen para la CNN.
        
        Args:
            imagen_path: Ruta a la imagen
            size: Tamaño objetivo para la imagen
            
        Returns:
            Tensor de imagen preprocesada
        """
        try:
            # Cargar imagen
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                raise ValueError(f"No se pudo leer la imagen: {imagen_path}")
            
            # Convertir de BGR a RGB
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            imagen = cv2.resize(imagen, size)
            
            # Normalizar y convertir a tensor
            imagen = imagen / 255.0
            imagen = np.transpose(imagen, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            tensor_imagen = torch.FloatTensor(imagen).unsqueeze(0)  # Añadir dimensión de batch
            
            return tensor_imagen.to(self.device)
        except Exception as e:
            logger.error(f"Error al preprocesar imagen: {e}")
            raise
    
    def detectar_patrones(self, imagen_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Utiliza el modelo CNN para detectar patrones en una imagen.
        
        Args:
            imagen_tensor: Tensor de imagen preprocesada
            
        Returns:
            Diccionario con resultados del análisis
        """
        if self.modelo_cnn is None:
            raise ValueError("Modelo CNN no cargado")
        
        with torch.no_grad():
            salida = self.modelo_cnn(imagen_tensor)
            prob = F.softmax(salida, dim=1)
            
        # Obtener la clase con mayor probabilidad
        _, predicted = torch.max(salida, 1)
        
        return {
            "clase_predicha": predicted.item(),
            "probabilidades": prob.cpu().numpy()[0]
        }
    
    @staticmethod
    def cargar_datos_csv(ruta_csv: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            ruta_csv: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            return pd.read_csv(ruta_csv)
        except Exception as e:
            logger.error(f"Error al cargar CSV: {e}")
            raise

# Clase para la interfaz gráfica
class InterfazEpidemiologia(tk.Tk):
    def __init__(self):
        """Inicializa la interfaz gráfica de la aplicación."""
        super().__init__()
        
        self.title("Sistema de Análisis Epidemiológico")
        self.geometry("1200x800")
        
        # Componentes de procesamiento
        self.procesador = ProcesadorDatos()
        self.modelos = ModelosEpidemiologicos()
        
        # Queue para comunicación con hilos
        self.queue = queue.Queue()
        
        # Crear componentes de la interfaz
        self.crear_menu()
        self.crear_pestanas()
        
        # Variables
        self.modelo_cargado = False
        self.datos_cargados = False
        self.ruta_datos = None
        
        logger.info("Interfaz inicializada")
    
    def crear_menu(self) -> None:
        """Crea el menú de la aplicación."""
        menu_principal = tk.Menu(self)
        self.config(menu=menu_principal)
        
        # Menú Archivo
        menu_archivo = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Archivo", menu=menu_archivo)
        
        menu_archivo.add_command(label="Cargar Datos", command=self.cargar_datos)
        menu_archivo.add_command(label="Cargar Modelo", command=self.cargar_modelo)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Guardar Resultados", command=self.guardar_resultados)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.quit)
        
        # Menú Análisis
        menu_analisis = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Análisis", menu=menu_analisis)
        
        menu_analisis.add_command(label="Modelo SIR", command=lambda: self.seleccionar_pestaña(1))
        menu_analisis.add_command(label="Modelo SEIR", command=lambda: self.seleccionar_pestaña(2))
        menu_analisis.add_command(label="Análisis de Imágenes", command=lambda: self.seleccionar_pestaña(3))
        
        # Menú Ayuda
        menu_ayuda = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Ayuda", menu=menu_ayuda)
        
        menu_ayuda.add_command(label="Acerca de", command=self.mostrar_acerca_de)
        menu_ayuda.add_command(label="Documentación", command=self.mostrar_documentacion)
    
    def crear_pestanas(self) -> None:
        """Crea el sistema de pestañas de la aplicación."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Pestaña principal
        self.tab_principal = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_principal, text="Principal")
        
        # Pestaña para modelo SIR
        self.tab_sir = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sir, text="Modelo SIR")
        
        # Pestaña para modelo SEIR
        self.tab_seir = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_seir, text="Modelo SEIR")
        
        # Pestaña para análisis de imágenes
        self.tab_imagenes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_imagenes, text="Análisis Imágenes")
        
        # Configurar cada pestaña
        self.configurar_tab_principal()
        self.configurar_tab_sir()
        self.configurar_tab_seir()
        self.configurar_tab_imagenes()
        
        # Log en la parte inferior
        self.crear_log()
    
    def configurar_tab_principal(self) -> None:
        """Configura la pestaña principal."""
        # Título
        lbl_titulo = ttk.Label(
            self.tab_principal, 
            text="Sistema de Análisis Epidemiológico", 
            font=("Helvetica", 16, "bold")
        )
        lbl_titulo.pack(pady=20)
        
        # Panel de estado
        frame_estado = ttk.LabelFrame(self.tab_principal, text="Estado del Sistema")
        frame_estado.pack(fill="x", padx=20, pady=10)
        
        # Indicadores de estado
        self.lbl_estado_datos = ttk.Label(
            frame_estado, 
            text="Datos: No cargados", 
            foreground="red"
        )
        self.lbl_estado_datos.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.lbl_estado_modelo = ttk.Label(
            frame_estado, 
            text="Modelo CNN: No cargado", 
            foreground="red"
        )
        self.lbl_estado_modelo.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Panel de acciones rápidas
        frame_acciones = ttk.LabelFrame(self.tab_principal, text="Acciones Rápidas")
        frame_acciones.pack(fill="x", padx=20, pady=10)
        
        btn_cargar_datos = ttk.Button(
            frame_acciones, 
            text="Cargar Datos", 
            command=self.cargar_datos
        )
        btn_cargar_datos.grid(row=0, column=0, padx=10, pady=10)
        
        btn_cargar_modelo = ttk.Button(
            frame_acciones, 
            text="Cargar Modelo CNN", 
            command=self.cargar_modelo
        )
        btn_cargar_modelo.grid(row=0, column=1, padx=10, pady=10)
        
        btn_ejecutar_sir = ttk.Button(
            frame_acciones, 
            text="Ejecutar Modelo SIR", 
            command=lambda: self.seleccionar_pestaña(1)
        )
        btn_ejecutar_sir.grid(row=1, column=0, padx=10, pady=10)
        
        btn_analizar_imagen = ttk.Button(
            frame_acciones, 
            text="Analizar Imagen", 
            command=lambda: self.seleccionar_pestaña(3)
        )
        btn_analizar_imagen.grid(row=1, column=1, padx=10, pady=10)
        
        # Panel de resumen
        frame_resumen = ttk.LabelFrame(self.tab_principal, text="Resumen de Análisis")
        frame_resumen.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.text_resumen = scrolledtext.ScrolledText(frame_resumen, height=10)
        self.text_resumen.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_resumen.insert(tk.END, "Cargue datos y ejecute análisis para ver resultados aquí.\n")
        self.text_resumen.config(state="disabled")
    
    def configurar_tab_sir(self) -> None:
        """Configura la pestaña del modelo SIR."""
        # Frame para parámetros
        frame_params = ttk.LabelFrame(self.tab_sir, text="Parámetros del Modelo SIR")
        frame_params.pack(fill="x", padx=20, pady=10)
        
        # Población total
        ttk.Label(frame_params, text="Población Total:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_poblacion_sir = ttk.Entry(frame_params)
        self.entry_poblacion_sir.grid(row=0, column=1, padx=10, pady=5)
        self.entry_poblacion_sir.insert(0, "10000")
        
        # Susceptibles iniciales
        ttk.Label(frame_params, text="Susceptibles Iniciales:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_s_init = ttk.Entry(frame_params)
        self.entry_s_init.grid(row=1, column=1, padx=10, pady=5)
        self.entry_s_init.insert(0, "9990")
        
        # Infectados iniciales
        ttk.Label(frame_params, text="Infectados Iniciales:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.entry_i_init = ttk.Entry(frame_params)
        self.entry_i_init.grid(row=2, column=1, padx=10, pady=5)
        self.entry_i_init.insert(0, "10")
        
        # Recuperados iniciales
        ttk.Label(frame_params, text="Recuperados Iniciales:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.entry_r_init = ttk.Entry(frame_params)
        self.entry_r_init.grid(row=3, column=1, padx=10, pady=5)
        self.entry_r_init.insert(0, "0")
        
        # Beta (tasa de transmisión)
        ttk.Label(frame_params, text="Beta (tasa de transmisión):").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.entry_beta_sir = ttk.Entry(frame_params)
        self.entry_beta_sir.grid(row=0, column=3, padx=10, pady=5)
        self.entry_beta_sir.insert(0, "0.3")
        
        # Gamma (tasa de recuperación)
        ttk.Label(frame_params, text="Gamma (tasa de recuperación):").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.entry_gamma_sir = ttk.Entry(frame_params)
        self.entry_gamma_sir.grid(row=1, column=3, padx=10, pady=5)
        self.entry_gamma_sir.insert(0, "0.1")
        
        # Días de simulación
        ttk.Label(frame_params, text="Días de simulación:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.entry_dias_sir = ttk.Entry(frame_params)
        self.entry_dias_sir.grid(row=2, column=3, padx=10, pady=5)
        self.entry_dias_sir.insert(0, "150")
        
        # Botón para ejecutar
        btn_ejecutar = ttk.Button(frame_params, text="Ejecutar Simulación", command=self.ejecutar_sir)
        btn_ejecutar.grid(row=4, column=0, columnspan=4, pady=15)
        
        # Frame para resultados
        frame_resultados = ttk.LabelFrame(self.tab_sir, text="Resultados")
        frame_resultados.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Área para gráficos
        self.frame_grafico_sir = ttk.Frame(frame_resultados)
        self.frame_grafico_sir.pack(fill="both", expand=True)
        
        # Información calculada
        frame_info = ttk.Frame(frame_resultados)
        frame_info.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(frame_info, text="R0:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.lbl_r0_sir = ttk.Label(frame_info, text="-")
        self.lbl_r0_sir.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(frame_info, text="Pico de Infectados:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.lbl_pico_sir = ttk.Label(frame_info, text="-")
        self.lbl_pico_sir.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        
        ttk.Label(frame_info, text="Día del Pico:").grid(row=0, column=4, padx=10, pady=5, sticky="w")
        self.lbl_dia_pico_sir = ttk.Label(frame_info, text="-")
        self.lbl_dia_pico_sir.grid(row=0, column=5, padx=10, pady=5, sticky="w")
        
    def configurar_tab_seir(self) -> None:
        """Configura la pestaña del modelo SEIR."""
        # Frame para parámetros
        frame_params = ttk.LabelFrame(self.tab_seir, text="Parámetros del Modelo SEIR")
        frame_params.pack(fill="x", padx=20, pady=10)
        
        # Población total
        ttk.Label(frame_params, text="Población Total:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_poblacion_seir = ttk.Entry(frame_params)
        self.entry_poblacion_seir.grid(row=0, column=1, padx=10, pady=5)
        self.entry_poblacion_seir.insert(0, "10000")
        
        # Susceptibles iniciales
        ttk.Label(frame_params, text="Susceptibles Iniciales:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_s_init_seir = ttk.Entry(frame_params)
        self.entry_s_init_seir.grid(row=1, column=1, padx=10, pady=5)
        self.entry_s_init_seir.insert(0, "9990")
        
        # Expuestos iniciales
        ttk.Label(frame_params, text="Expuestos Iniciales:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.entry_e_init = ttk.Entry(frame_params)
        self.entry_e_init.grid(row=2, column=1, padx=10, pady=5)
        self.entry_e_init.insert(0, "10")
        
        # Infectados iniciales
        ttk.Label(frame_params, text="Infectados Iniciales:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.entry_i_init_seir = ttk.Entry(frame_params)
        self.entry_i_init_seir.grid(row=3, column=1, padx=10, pady=5)
        self.entry_i_init_seir.insert(0, "0")
        
        # Recuperados iniciales
        ttk.Label(frame_params, text="Recuperados Iniciales:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.entry_r_init_seir = ttk.Entry(frame_params)
        self.entry_r_init_seir.grid(row=4, column=1, padx=10, pady=5)
        self.entry_r_init_seir.insert(0, "0")
        
        # Beta (tasa de transmisión)
        ttk.Label(frame_params, text="Beta (tasa de transmisión):").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.entry_beta_seir = ttk.Entry(frame_params)
        self.entry_beta_seir.grid(row=0, column=3, padx=10, pady=5)
        self.entry_beta_seir.insert(0, "0.3")
        
        # Sigma (tasa de incubación)
        ttk.Label(frame_params, text="Sigma (tasa de incubación):").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.entry_sigma = ttk.Entry(frame_params)
        self.entry_sigma.grid(row=1, column=3, padx=10, pady=5)
        self.entry_sigma.insert(0, "0.2")
        
        # Gamma (tasa de recuperación)
        ttk.Label(frame_params, text="Gamma (tasa de recuperación):").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.entry_gamma_seir = ttk.Entry(frame_params)
        self.entry_gamma_seir.grid(row=2, column=3, padx=10, pady=5)
        self.entry_gamma_seir.insert(0, "0.1")
        
        # Días de simulación
        ttk.Label(frame_params, text="Días de simulación:").grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.entry_dias_seir = ttk.Entry(frame_params)
        self.entry_dias_seir.grid(row=3, column=3, padx=10, pady=5)
        self.entry_dias_seir.insert(0, "150")
        
        # Botón para ejecutar
        btn_ejecutar = ttk.Button(frame_params, text="Ejecutar Simulación", command=self.ejecutar_seir)
        btn_ejecutar.grid(row=5, column=0, columnspan=4, pady=15)
        
        # Frame para resultados
        frame_resultados = ttk.LabelFrame(self.tab_seir, text="Resultados")
        frame_resultados.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Área para gráficos
        self.frame_grafico_seir = ttk.Frame(frame_resultados)
        self.frame_grafico_seir.pack(fill="both", expand=True)
        
        # Información calculada
        frame_info = ttk.Frame(frame_resultados)
        frame_info.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(frame_info, text="R0:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.lbl_r0_seir = ttk.Label(frame_info, text="-")
        self.lbl_r0_seir.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(frame_info, text="Pico de Infectados:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        self.lbl_pico_seir = ttk.Label(frame_info, text="-")
        self.lbl_pico_seir.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        
        ttk.Label(frame_info, text="Día del Pico:").grid(row=0, column=4, padx=10, pady=5, sticky="w")
        self.lbl_dia_pico_seir = ttk.Label(frame_info, text="-")
        self.lbl_dia_pico_seir.grid(row=0, column=5, padx=10, pady=5, sticky="w")
        
    def configurar_tab_imagenes(self) -> None:
        """Configura la pestaña de análisis de imágenes."""
        # Frame para selección de imagen
        frame_seleccion = ttk.LabelFrame(self.tab_imagenes, text="Selección de Imagen")
        frame_seleccion.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(frame_seleccion, text="Imagen:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_ruta_imagen = ttk.Entry(frame_seleccion, width=60)
        self.entry_ruta_imagen.grid(row=0, column=1, padx=10, pady=5)
        
        btn_examinar = ttk.Button(frame_seleccion, text="Examinar", command=self.seleccionar_imagen)
        btn_examinar.grid(row=0, column=2, padx=10, pady=5)
        
        btn_analizar = ttk.Button(frame_seleccion, text="Analizar Imagen", command=self.analizar_imagen)
        btn_analizar.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Frame para visualización y resultados
        frame_vis = ttk.Frame(self.tab_imagenes)
        frame_vis.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Área para la imagen
        frame_imagen = ttk.LabelFrame(frame_vis, text="Imagen")
        frame_imagen.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_imagen = ttk.Label(frame_imagen)
        self.lbl_imagen.pack(padx=10, pady=10)
        
        # Área para resultados
        frame_res_img = ttk.LabelFrame(frame_vis, text="Resultados del Análisis")
        frame_res_img.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.text_res_img = scrolledtext.ScrolledText(frame_res_img, width=40, height=20)
        self.text_res_img.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configurar grid
        frame_vis.columnconfigure(0, weight=1)
        frame_vis.columnconfigure(1, weight=1)
        frame_vis.rowconfigure(0, weight=1)
        
    def crear_log(self) -> None:
        """Crea el área de log en la parte inferior."""
        frame_log = ttk.LabelFrame(self, text="Log")
        frame_log.pack(fill="x", padx=10, pady=5)
        
        self.text_log = scrolledtext.ScrolledText(frame_log, height=5)
        self.text_log.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_log.config(state="disabled")
        
        # Configurar handler para redirigir logs a esta área
        self.log_handler = LogTextHandler(self.text_log)
        logger.addHandler(self.log_handler)
        
        logger.info("Sistema iniciado")
    
    def seleccionar_pestaña(self, indice: int) -> None:
        """
        Selecciona una pestaña por su índice.
        
        Args:
            indice: Índice de la pestaña a seleccionar
        """
        self.notebook.select(indice)
    
    def cargar_datos(self) -> None:
        """Abre un diálogo para cargar datos CSV."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            try:
                self.ruta_datos = ruta
                # Iniciar carga en hilo separado
                threading.Thread(target=self._cargar_datos_thread, args=(ruta,), daemon=True).start()
                
                logger.info(f"Iniciando carga de datos desde: {ruta}")
                self.mostrar_log(f"Cargando datos desde {ruta}...")
            except Exception as e:
                logger.error(f"Error al iniciar carga de datos: {e}")
                messagebox.showerror("Error", f"No se pudo iniciar la carga de datos: {e}")
    
    def _cargar_datos_thread(self, ruta: str) -> None:
        """
        Hilo para cargar datos.
        
        Args:
            ruta: Ruta al archivo de datos
        """
        try:
            df = self.procesador.cargar_datos_csv(ruta)
            self.queue.put(("datos_cargados", df))
            logger.info(f"Datos cargados con éxito: {len(df)} registros")
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            self.queue.put(("error", f"Error al cargar datos: {e}"))
        
        self.after(100, self.verificar_queue)
    
    def verificar_queue(self) -> None:
        """Verifica si hay mensajes en la cola."""
        try:
            while True:
                mensaje, datos = self.queue.get_nowait()
                
                if mensaje == "datos_cargados":
                    self.datos_cargados = True
                    self.lbl_estado_datos.config(text="Datos: Cargados", foreground="green")
                    self.mostrar_log(f"Datos cargados: {len(datos)} registros")
                    self.mostrar_resumen_datos(datos)
                    
                    # Actualizar texto de resumen
                    self.actualizar_resumen(f"Datos cargados desde: {self.ruta_datos}\n")
                    self.actualizar_resumen(f"Total de registros: {len(datos)}\n")
                    self.actualizar_resumen(f"Columnas: {', '.join(datos.columns)}\n")
                
                elif mensaje == "modelo_cargado":
                    self.modelo_cargado = True
                    self.lbl_estado_modelo.config(text="Modelo CNN: Cargado", foreground="green")
                    self.mostrar_log("Modelo CNN cargado con éxito")
                    
                    # Actualizar texto de resumen
                    self.actualizar_resumen("Modelo CNN cargado con éxito.\n")
                
                elif mensaje == "analisis_completo":
                    self.mostrar_log("Análisis completado")
                    self.mostrar_resultados_analisis(datos)
                
                elif mensaje == "error":
                    messagebox.showerror("Error", datos)
                
                self.queue.task_done()
        
        except queue.Empty:
            pass
        
        self.after(100, self.verificar_queue)
    
    def cargar_modelo(self) -> None:
        """Abre un diálogo para cargar un modelo CNN."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar modelo CNN",
            filetypes=[("Modelos PyTorch", "*.pt *.pth"), ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            try:
                # Iniciar carga en hilo separado
                threading.Thread(target=self._cargar_modelo_thread, args=(ruta,), daemon=True).start()
                
                logger.info(f"Iniciando carga de modelo desde: {ruta}")
                self.mostrar_log(f"Cargando modelo desde {ruta}...")
            except Exception as e:
                logger.error(f"Error al iniciar carga de modelo: {e}")
                messagebox.showerror("Error", f"No se pudo iniciar la carga del modelo: {e}")
    
    def _cargar_modelo_thread(self, ruta: str) -> None:
        """
        Hilo para cargar modelo.
        
        Args:
            ruta: Ruta al archivo del modelo
        """
        try:
            self.procesador.cargar_modelo(ruta)
            self.queue.put(("modelo_cargado", None))
            logger.info("Modelo cargado con éxito")
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            self.queue.put(("error", f"Error al cargar modelo: {e}"))
    
    def seleccionar_imagen(self) -> None:
        """Abre un diálogo para seleccionar una imagen."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if ruta:
            self.entry_ruta_imagen.delete(0, tk.END)
            self.entry_ruta_imagen.insert(0, ruta)
            
            # Mostrar la imagen
            try:
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                
                # Redimensionar para visualización
                h, w = imagen.shape[:2]
                max_dim = 300
                scale = min(max_dim / w, max_dim / h)
                
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                imagen = cv2.resize(imagen, (new_w, new_h))
                
                # Convertir a formato para tkinter
                imagen = Image.fromarray(imagen)
                imagen_tk = ImageTk.PhotoImage(image=imagen)
                
                # Guardar referencia para evitar garbage collection
                self.imagen_actual = imagen_tk
                
                # Mostrar en la interfaz
                self.lbl_imagen.configure(image=imagen_tk)
                
                logger.info(f"Imagen cargada: {ruta}")
            except Exception as e:
                logger.error(f"Error al mostrar la imagen: {e}")
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
    
    def analizar_imagen(self) -> None:
        """Analiza la imagen seleccionada con el modelo CNN."""
        ruta = self.entry_ruta_imagen.get()
        
        if not ruta:
            messagebox.showwarning("Advertencia", "Por favor, seleccione una imagen primero.")
            return
        
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Debe cargar un modelo CNN primero.")
            return
        
        try:
            # Iniciar análisis en hilo separado
            threading.Thread(target=self._analizar_imagen_thread, args=(ruta,), daemon=True).start()
            
            logger.info(f"Iniciando análisis de imagen: {ruta}")
            self.mostrar_log(f"Analizando imagen {ruta}...")
            
            # Limpiar resultados anteriores
            self.text_res_img.delete(1.0, tk.END)
        except Exception as e:
            logger.error(f"Error al iniciar análisis de imagen: {e}")
            messagebox.showerror("Error", f"No se pudo iniciar el análisis: {e}")
    
    def _analizar_imagen_thread(self, ruta: str) -> None:
        """
        Hilo para analizar una imagen.
        
        Args:
            ruta: Ruta a la imagen
        """
        try:
            # Preprocesar imagen
            tensor_imagen = self.procesador.preprocesar_imagen(ruta)
            
            # Detectar patrones
            resultados = self.procesador.detectar_patrones(tensor_imagen)
            
            self.queue.put(("analisis_completo", resultados))
            logger.info("Análisis de imagen completado")
        except Exception as e:
            logger.error(f"Error al analizar imagen: {e}")
            self.queue.put(("error", f"Error al analizar la imagen: {e}"))
    
    def ejecutar_sir(self) -> None:
        """Ejecuta una simulación con el modelo SIR."""
        try:
            # Obtener parámetros
            poblacion = int(self.entry_poblacion_sir.get())
            s_init = int(self.entry_s_init.get())
            i_init = int(self.entry_i_init.get())
            r_init = int(self.entry_r_init.get())
            beta = float(self.entry_beta_sir.get())
            gamma = float(self.entry_gamma_sir.get())
            dias = int(self.entry_dias_sir.get())
            
            # Validar que la suma sea igual a la población
            if s_init + i_init + r_init != poblacion:
                messagebox.showwarning(
                    "Advertencia", 
                    "La suma de S + I + R debe ser igual a la población total."
                )
                return
            
            # Ejecutar modelo
            resultados = self.modelos.modelo_sir(
                poblacion=poblacion,
                susceptibles_init=s_init,
                infectados_init=i_init,
                recuperados_init=r_init,
                beta=beta,
                gamma=gamma,
                dias=dias
            )
            
            # Calcular R0
            r0 = beta / gamma
            
            # Encontrar pico de infectados
            pico_infectados = np.max(resultados["I"])
            dia_pico = np.argmax(resultados["I"])
            
            # Actualizar interfaz
            self.lbl_r0_sir.config(text=f"{r0:.2f}")
            self.lbl_pico_sir.config(text=f"{pico_infectados:.0f} ({(pico_infectados / poblacion * 100):.2f}%)")
            self.lbl_dia_pico_sir.config(text=f"{dia_pico}")
            
            # Graficar resultados
            self.graficar_resultados_sir(resultados, dias)
            
            # Actualizar resumen
            self.actualizar_resumen(f"\n--- Resultados Modelo SIR ---\n")
            self.actualizar_resumen(f"R0: {r0:.2f}\n")
            self.actualizar_resumen(f"Pico de infectados: {pico_infectados:.0f} ({(pico_infectados / poblacion * 100):.2f}%)\n")
            self.actualizar_resumen(f"Día del pico: {dia_pico}\n")
            
            logger.info("Simulación SIR completada")
            self.mostrar_log("Simulación SIR completada con éxito")
        except Exception as e:
            logger.error(f"Error al ejecutar modelo SIR: {e}")
            messagebox.showerror("Error", f"Error al ejecutar simulación: {e}")
    
    def ejecutar_seir(self) -> None:
        """Ejecuta una simulación con el modelo SEIR."""
        try:
            # Obtener parámetros
            poblacion = int(self.entry_poblacion_seir.get())
            s_init = int(self.entry_s_init_seir.get())
            e_init = int(self.entry_e_init.get())
            i_init = int(self.entry_i_init_seir.get())
            r_init = int(self.entry_r_init_seir.get())
            beta = float(self.entry_beta_seir.get())
            sigma = float(self.entry_sigma.get())
            gamma = float(self.entry_gamma_seir.get())
            dias = int(self.entry_dias_seir.get())
            
            # Validar que la suma sea igual a la población
            if s_init + e_init + i_init + r_init != poblacion:
                messagebox.showwarning(
                    "Advertencia", 
                    "La suma de S + E + I + R debe ser igual a la población total."
                )
                return
            
            # Ejecutar modelo
            resultados = self.modelos.modelo_seir(
                poblacion=poblacion,
                susceptibles_init=s_init,
                expuestos_init=e_init,
                infectados_init=i_init,
                recuperados_init=r_init,
                beta=beta,
                sigma=sigma,
                gamma=gamma,
                dias=dias
            )
            
            # Calcular R0
            r0 = beta / gamma
            
            # Encontrar pico de infectados
            pico_infectados = np.max(resultados["I"])
            dia_pico = np.argmax(resultados["I"])
            
            # Actualizar interfaz
            self.lbl_r0_seir.config(text=f"{r0:.2f}")
            self.lbl_pico_seir.config(text=f"{pico_infectados:.0f} ({(pico_infectados / poblacion * 100):.2f}%)")
            self.lbl_dia_pico_seir.config(text=f"{dia_pico}")
            
            # Graficar resultados
            self.graficar_resultados_seir(resultados, dias)
            
            # Actualizar resumen
            self.actualizar_resumen(f"\n--- Resultados Modelo SEIR ---\n")
            self.actualizar_resumen(f"R0: {r0:.2f}\n")
            self.actualizar_resumen(f"Pico de infectados: {pico_infectados:.0f} ({(pico_infectados / poblacion * 100):.2f}%)\n")
            self.actualizar_resumen(f"Día del pico: {dia_pico}\n")
            
            logger.info("Simulación SEIR completada")
            self.mostrar_log("Simulación SEIR completada con éxito")
        except Exception as e:
            logger.error(f"Error al ejecutar modelo SEIR: {e}")
            messagebox.showerror("Error", f"Error al ejecutar simulación: {e}")
    
    def graficar_resultados_sir(self, resultados: Dict[str, np.ndarray], dias: int) -> None:
        """
        Grafica los resultados de la simulación SIR.
        
        Args:
            resultados: Diccionario con los resultados (S, I, R)
            dias: Número de días simulados
        """
        # Limpiar gráfico anterior
        for widget in self.frame_grafico_sir.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Eje X (días)
        dias_eje = np.arange(dias)
        
        # Graficar cada curva
        ax.plot(dias_eje, resultados["S"], label="Susceptibles", color="blue")
        ax.plot(dias_eje, resultados["I"], label="Infectados", color="red")
        ax.plot(dias_eje, resultados["R"], label="Recuperados", color="green")
        
        # Configurar gráfico
        ax.set_xlabel("Días")
        ax.set_ylabel("Población")
        ax.set_title("Modelo SIR")
        ax.legend()
        ax.grid(True)
        
        # Integrar en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_grafico_sir)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def graficar_resultados_seir(self, resultados: Dict[str, np.ndarray], dias: int) -> None:
        """
        Grafica los resultados de la simulación SEIR.
        
        Args:
            resultados: Diccionario con los resultados (S, E, I, R)
            dias: Número de días simulados
        """
        # Limpiar gráfico anterior
        for widget in self.frame_grafico_seir.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Eje X (días)
        dias_eje = np.arange(dias)
        
        # Graficar cada curva
        ax.plot(dias_eje, resultados["S"], label="Susceptibles", color="blue")
        ax.plot(dias_eje, resultados["E"], label="Expuestos", color="orange")
        ax.plot(dias_eje, resultados["I"], label="Infectados", color="red")
        ax.plot(dias_eje, resultados["R"], label="Recuperados", color="green")
        
        # Configurar gráfico
        ax.set_xlabel("Días")
        ax.set_ylabel("Población")
        ax.set_title("Modelo SEIR")
        ax.legend()
        ax.grid(True)
        
        # Integrar en tkinter
        canvas = FigureCanvasTkAgg(fig, self.frame_grafico_seir)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def guardar_resultados(self) -> None:
        """Guarda los resultados del análisis actual."""
        ruta = filedialog.asksaveasfilename(
            title="Guardar Resultados",
            defaultextension=".json",
            filetypes=[
                ("Archivo JSON", "*.json"),
                ("Archivo de Texto", "*.txt"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if ruta:
            try:
                # Recopilar datos
                datos = {
                    "fecha_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "resumen": self.text_resumen.get(1.0, tk.END)
                }
                
                # Añadir datos específicos según la pestaña activa
                pestaña_actual = self.notebook.index(self.notebook.select())
                
                if pestaña_actual == 1:  # SIR
                    datos["tipo_modelo"] = "SIR"
                    datos["parametros"] = {
                        "poblacion": self.entry_poblacion_sir.get(),
                        "susceptibles_init": self.entry_s_init.get(),
                        "infectados_init": self.entry_i_init.get(),
                        "recuperados_init": self.entry_r_init.get(),
                        "beta": self.entry_beta_sir.get(),
                        "gamma": self.entry_gamma_sir.get(),
                        "dias": self.entry_dias_sir.get()
                    }
                    datos["resultados"] = {
                        "r0": self.lbl_r0_sir.cget("text"),
                        "pico_infectados": self.lbl_pico_sir.cget("text"),
                        "dia_pico": self.lbl_dia_pico_sir.cget("text")
                    }
                
                elif pestaña_actual == 2:  # SEIR
                    datos["tipo_modelo"] = "SEIR"
                    datos["parametros"] = {
                        "poblacion": self.entry_poblacion_seir.get(),
                        "susceptibles_init": self.entry_s_init_seir.get(),
                        "expuestos_init": self.entry_e_init.get(),
                        "infectados_init": self.entry_i_init_seir.get(),
                        "recuperados_init": self.entry_r_init_seir.get(),
                        "beta": self.entry_beta_seir.get(),
                        "sigma": self.entry_sigma.get(),
                        "gamma": self.entry_gamma_seir.get(),
                        "dias": self.entry_dias_seir.get()
                    }
                    datos["resultados"] = {
                        "r0": self.lbl_r0_seir.cget("text"),
                        "pico_infectados": self.lbl_pico_seir.cget("text"),
                        "dia_pico": self.lbl_dia_pico_seir.cget("text")
                    }
                
                # Guardar como JSON
                with open(ruta, "w") as f:
                    json.dump(datos, f, indent=4)
                
                logger.info(f"Resultados guardados en: {ruta}")
                self.mostrar_log(f"Resultados guardados en: {ruta}")
                messagebox.showinfo("Éxito", f"Resultados guardados en: {ruta}")
            except Exception as e:
                logger.error(f"Error al guardar resultados: {e}")
                messagebox.showerror("Error", f"No se pudieron guardar los resultados: {e}")
    
    def mostrar_resumen_datos(self, df: pd.DataFrame) -> None:
        """
        Muestra un resumen de los datos cargados.
        
        Args:
            df: DataFrame con los datos
        """
        # Implementar según el formato de datos esperado
        pass
    
    def mostrar_resultados_analisis(self, resultados: Dict[str, Any]) -> None:
        """
        Muestra los resultados del análisis de imagen.
        
        Args:
            resultados: Diccionario con los resultados del análisis
        """
        # Limpiar resultados anteriores
        self.text_res_img.config(state="normal")
        self.text_res_img.delete(1.0, tk.END)
        
        # Clase predicha
        clase = resultados["clase_predicha"]
        self.text_res_img.insert(tk.END, f"Clase predicha: {clase}\n\n")
        
        # Probabilidades
        self.text_res_img.insert(tk.END, "Probabilidades por clase:\n")
        for i, prob in enumerate(resultados["probabilidades"]):
            self.text_res_img.insert(tk.END, f"Clase {i}: {prob:.4f} ({prob*100:.2f}%)\n")
        
        self.text_res_img.config(state="disabled")
        
        # También agregar al resumen
        self.actualizar_resumen("\n--- Análisis de Imagen ---\n")
        self.actualizar_resumen(f"Clase predicha: {clase}\n")
        self.actualizar_resumen(f"Probabilidad: {resultados['probabilidades'][clase]:.4f}\n")
    
    def mostrar_log(self, mensaje: str) -> None:
        """
        Añade un mensaje al área de log.
        
        Args:
            mensaje: Mensaje a añadir
        """
        self.text_log.config(state="normal")
        self.text_log.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {mensaje}\n")
        self.text_log.see(tk.END)
        self.text_log.config(state="disabled")
    
    def actualizar_resumen(self, texto: str) -> None:
        """
        Actualiza el área de resumen.
        
        Args:
            texto: Texto a añadir
        """
        self.text_resumen.config(state="normal")
        self.text_resumen.insert(tk.END, texto)
        self.text_resumen.see(tk.END)
        self.text_resumen.config(state="disabled")
    
    def mostrar_acerca_de(self) -> None:
        """Muestra información sobre la aplicación."""
        mensaje = """Sistema de Análisis Epidemiológico

Versión 1.0

Esta aplicación implementa modelos matemáticos 
para el análisis epidemiológico y utiliza redes 
neuronales convolucionales para el análisis de imágenes.

Modelos implementados:
- SIR (Susceptible-Infected-Recovered)
- SEIR (Susceptible-Exposed-Infected-Recovered)
"""
        messagebox.showinfo("Acerca de", mensaje)
    
    def mostrar_documentacion(self) -> None:
        """Muestra la documentación de la aplicación."""
        mensaje = """Documentación

1. Carga de datos:
   - Use el menú Archivo > Cargar Datos para importar datos en formato CSV.

2. Modelos epidemiológicos:
   - SIR: Modelo básico para enfermedades sin período de incubación.
   - SEIR: Modelo que incluye período de exposición (incubación).

3. Análisis de imágenes:
   - Requiere un modelo CNN pre-entrenado.
   - Las imágenes pueden ser analizadas para detectar patrones epidemiológicos.

Para más información, consulte el manual completo.
"""
        messagebox.showinfo("Documentación", mensaje)

# Handler personalizado para redirigir logs al área de texto
class LogTextHandler(logging.Handler):
    def __init__(self, text_widget: scrolledtext.ScrolledText):
        """
        Inicializa el handler.
        
        Args:
            text_widget: Widget de texto donde se mostrarán los logs
        """
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emite un registro de log al widget de texto.
        
        Args:
            record: Registro de log
        """
        msg = self.format(record)
        
        def _emit():
            self.text_widget.config(state="normal")
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
            self.text_widget.config(state="disabled")
        
        # Ejecutar en el hilo principal de tkinter
        self.text_widget.after(0, _emit)

# Punto de entrada
if __name__ == "__main__":
    # Importar aquí para evitar problemas de dependencia circular
    from PIL import Image, ImageTk
    
    # Verificar disponibilidad de GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("GPU no disponible, usando CPU")
    
    # Iniciar aplicación
    app = InterfazEpidemiologia()
    app.mainloop()
