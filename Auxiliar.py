import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

class Auxiliar:
    def __init__(self):
        """Clase auxiliar para manejar operaciones comunes como carga de datos, preprocesamiento y visualización."""
        pass

    def cargarDatos(self, fuente, tipo='csv', agregar_columna_unos=False, 
                    categoria_idx=0, conservar_nombres=True, **kwargs):
        """
        Carga datos desde un archivo CSV o una matriz previamente cargada.

        Args:
            fuente (str o np.ndarray): Ruta del archivo CSV o matriz de datos.
            tipo (str): Tipo de fuente. Opciones:
                - 'csv': Carga desde un archivo CSV.
                - 'matriz': Carga desde una matriz en memoria.
            agregar_columna_unos (bool): Si True, agrega una columna de unos al inicio de los datos.
            categoria_idx (int): Índice de la columna que contiene las categorías (etiquetas).
                Si es negativo, cuenta desde el final.
                Si es igual al número de columnas, no se consideran categorías, solo se devuelven los datos (Se usa para predicciones).
            conservar_nombres (bool): Si True, conserva los nombres de las columnas (solo para CSV).
            **kwargs: Argumentos adicionales para `pd.read_csv` si se usa un archivo CSV.

        Returns:
            tuple: (datos, categorias, nombres) donde:
                - datos (np.ndarray): Matriz de características.
                - categorias (np.ndarray): Vector de etiquetas.
                - nombres (list): Lista de nombres de las columnas (si aplica).

        Raises:
            ValueError: Si el índice de categoría está fuera de rango o el tipo de fuente no es válido.
        """

        def _verificar_numericos(matriz):
            if not np.issubdtype(matriz.dtype, np.number):
                print("\n¡CUIDADO! Si la data contiene cadenas, categorías, fechas y cualquier otro tipo no numérico, el codigo fallará (No tiene soporte para eso)")
                print("Para poder usar este código, convierta las cadenas a un número o a una categoría numérica (haciendo un one-hot encoding o similar)")
                return False
            return True
        
        def _cargar_desde_csv(ruta, agregar_columna_unos=True, categoria_idx=0, 
                      conservar_nombres=False, **kwargs):
            df = pd.read_csv(ruta, **kwargs)
            datos = df.values

            if not _verificar_numericos(datos):
                raise ValueError("Los datos deben ser numéricos")

            if abs(categoria_idx) > datos.shape[1]:
                raise ValueError("Índice de categoría fuera de rango")

            nombres = df.columns.tolist() if conservar_nombres else None

            if categoria_idx == datos.shape[1]:  # No hay categorías
                
                if agregar_columna_unos:
                    datos = np.hstack((np.ones((datos.shape[0], 1)), datos))
                
                    if nombres:
                        nombres.insert(0, 'Intercepto')
                
                return (datos, nombres) if conservar_nombres else datos

            categorias = datos[:, categoria_idx]
            datos = np.delete(datos, categoria_idx, axis=1)

            if agregar_columna_unos:
                datos = np.hstack((np.ones((datos.shape[0], 1)), datos))
            if nombres:
                nombres.insert(0, 'Intercepto')
            if conservar_nombres:
                nombres.pop(categoria_idx)
                return datos, categorias, nombres

            return datos, categorias

        def _cargar_desde_matriz(matriz, agregar_columna_unos=True, categoria_idx=0):
            if not isinstance(matriz, np.ndarray):
                matriz = np.array(matriz)
            
            if abs(categoria_idx) >= matriz.shape[1]:
                raise ValueError("Índice de categoría fuera de rango")
            
            categorias = matriz[:, categoria_idx]
            datos = np.delete(matriz, categoria_idx, axis=1)
            
            if agregar_columna_unos:
                datos = np.hstack((np.ones((datos.shape[0], 1)), datos))
            
            return datos, categorias

        if tipo == 'csv':
            return _cargar_desde_csv(fuente, agregar_columna_unos, categoria_idx, 
                                     conservar_nombres, **kwargs)
        elif tipo == 'matriz':
            return _cargar_desde_matriz(fuente, agregar_columna_unos, categoria_idx)
        else:
            raise ValueError(f"Tipo de fuente '{tipo}' no soportado. Opciones válidas: 'csv' o 'matriz'")
        

    @staticmethod
    def dividir_y_preprocesar(datos, categorias, prueba_size=0.2, aleatorio=True, semilla=None, normalizar=True):
        """
        Divide los datos en conjuntos de entrenamiento y pr, y los normalizaueba, y los normaliza.

        Args:
            datos (np.ndarray): Matriz de características.
            categorias (np.ndarray): Vector de etiquetas.
            prueba_size (float): Proporción de datos para el conjunto de prueba (0 < prueba_size < 1).
            aleatorio (bool): Si True, divide los datos aleatoriamente.
            semilla (int): Semilla para la generación de números aleatorios (si aleatorio=True).
            normalizar (bool): Si True, normaliza las características.

        Returns:
            tuple: (X_train, y_train, X_test, y_test) donde:
                - X_train (np.ndarray): Características de entrenamiento.
                - y_train (np.ndarray): Etiquetas de entrenamiento.
                - X_test (np.ndarray): Características de prueba.
                - y_test (np.ndarray): Etiquetas de prueba.

        Raises:
            ValueError: Si los datos o las categorías no son np.ndarray o si no tienen la misma longitud.
        """
        if not isinstance(datos, np.ndarray) or not isinstance(categorias, np.ndarray):
            raise ValueError("Los datos y las categorías deben ser np.ndarray")
        if len(datos) != len(categorias):
            raise ValueError("Los datos y las categorías deben tener la misma longitud")
        
        # Dividir datos en entrenamiento y prueba
        if aleatorio:
            if semilla is not None:
                np.random.seed(semilla)
            indices = np.random.permutation(len(datos))
        else:
            indices = np.arange(len(datos))
        
        prueba_len = int(len(datos) * prueba_size)
        prueba_indices = indices[:prueba_len]
        entrenamiento_indices = indices[prueba_len:]
        
        X_train = datos[entrenamiento_indices]
        y_train = categorias[entrenamiento_indices]
        X_test = datos[prueba_indices]
        y_test = categorias[prueba_indices]

        # Normalizar características si corresponde
        if normalizar:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test

    @staticmethod
    def graficarConfusion(matriz, etiquetas=['Clase 0', 'Clase 1'], guardar=False, nombre_sin_ext='matriz_confusion', titulo='Matriz de Confusión'):
        """
        Muestra la matriz de confusión usando un mapa de calor y permite guardar el resultado.

        Args:
            matriz (np.ndarray): Matriz de confusión (2x2).
            etiquetas (list): Etiquetas para las clases (debe tener 2 elementos).
            guardar (bool): Si True, guarda la gráfica en un archivo.
            nombre_sin_ext (str): Nombre del archivo (sin extensión) si guardar=True.
            titulo (str): Título de la gráfica.

        Raises:
            ValueError: Si la matriz no es 2x2 o las etiquetas no tienen 2 elementos.
        """
        if not isinstance(matriz, np.ndarray):
            raise ValueError("La matriz de confusión debe ser un np.ndarray")
        if matriz.shape != (2, 2):
            raise ValueError("La matriz de confusión debe ser de tamaño 2x2")
        if not isinstance(etiquetas, list) or len(etiquetas) != 2:
            raise ValueError("Las etiquetas deben ser una lista de dos elementos")

        fig, ax = plt.subplots()
        cax = ax.matshow(matriz, cmap=plt.cm.Blues)
        plt.colorbar(cax)

        # Ajustar el color del texto según el brillo de la celda
        for (i, j), val in np.ndenumerate(matriz):
            color = 'white' if cax.get_cmap()(matriz[i, j] / matriz.max())[0:3] < (0.5, 0.5, 0.5) else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center', color=color)

        ax.set_xticks(range(len(etiquetas)))
        ax.set_yticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas)
        ax.set_yticklabels(etiquetas)

        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title(titulo)

        if guardar:
            ruta_guardado = f'resultados/{nombre_sin_ext}.jpg'
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            plt.savefig(ruta_guardado, bbox_inches='tight')
        
        plt.show()

    @staticmethod
    def _calcularExactitud(real, pred):
        """
        Calcula la exactitud de las predicciones.

        Args:
            real (np.ndarray): Etiquetas reales.
            pred (np.ndarray): Etiquetas predichas.

        Returns:
            float: Exactitud de las predicciones.

        Raises:
            ValueError: Si las etiquetas reales y predichas no tienen la misma forma.
        """
        if not isinstance(real, np.ndarray) or not isinstance(pred, np.ndarray):
            raise ValueError("Las etiquetas reales y predichas deben ser np.ndarray")
        if real.shape != pred.shape:
            raise ValueError("Las etiquetas reales y predichas deben tener la misma forma")
        
        return np.mean(real == pred)

    @staticmethod
    def _matrizConfusion(real, pred):
        """
        Calcula la matriz de confusión para las etiquetas reales y predichas.

        Args:
            real (np.ndarray): Etiquetas reales.
            pred (np.ndarray): Etiquetas predichas.

        Returns:
            np.ndarray: Matriz de confusión (2x2).

        Raises:
            ValueError: Si las etiquetas reales y predichas no tienen la misma forma.
        """
        if not isinstance(real, np.ndarray) or not isinstance(pred, np.ndarray):
            raise ValueError("Las etiquetas reales y predichas deben ser np.ndarray")
        if real.shape != pred.shape:
            raise ValueError("Las etiquetas reales y predichas deben tener la misma forma")
        
        tp = np.sum((real == 1) & (pred == 1))
        tn = np.sum((real == 0) & (pred == 0))
        fp = np.sum((real == 0) & (pred == 1))
        fn = np.sum((real == 1) & (pred == 0))
        
        return np.array([[tn, fp], [fn, tp]])

    def encontrar_umbral_optimo(self, y_real, y_prob, C_FP: float =1, C_FN: float =1):
        """
        Encuentra el umbral que minimiza el costo total.

        Args:
            y_real (np.ndarray): Etiquetas reales.
            y_prob (np.ndarray): Probabilidades predichas.
            C_FP (float): Costo de un falso positivo.
            C_FN (float): Costo de un falso negativo.

        Returns:
            float: Umbral óptimo que minimiza el costo.

        Raises:
            ValueError: Si las etiquetas reales y probabilidades no tienen la misma forma.
        """
        if not isinstance(y_real, np.ndarray) or not isinstance(y_prob, np.ndarray):
            raise ValueError("y_real y y_prob deben ser np.ndarray")
        if y_real.shape != y_prob.shape:
            raise ValueError("y_real y y_prob deben tener la misma forma")
        
        thresholds = np.linspace(0, 1, 100)
        costos = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            fp = np.sum((y_pred == 1) & (y_real == 0))
            fn = np.sum((y_pred == 0) & (y_real == 1))
            costo = C_FP * fp + C_FN * fn
            costos.append(costo)
        
        umbral_optimo = thresholds[np.argmin(costos)]
        
        return umbral_optimo
