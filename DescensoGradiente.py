import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class DescensoGradiente:
    def __init__(self, funcion: object) -> None:
        """
        Inicializa el optimizador con una función objetivo.

        Args:
            funcion (object): Objeto que debe implementar los métodos:
                - evaluar(x): Evalúa la función objetivo en el punto x.
                - gradiente(x): Calcula el gradiente en el punto x.
                - hessiano(x) (opcional): Calcula la Hessiana en el punto x.
                - dimension (int): Dimensión del espacio del problema.
        """
        self._validar_funcion(funcion)
        self.funcion = funcion
        self.dimension = funcion.dimension

    def calcular_paso(self, x_actual: np.ndarray, metodo_paso: str = "exacto", 
                      paso_fijo: float = 1e-3, alpha_inicial: float = None, 
                      factor_contraccion: float = None, c1: float = None,
                      max_iter_backtracking: int = 1000) -> float:
        """
        Calcula el tamaño de paso para la iteración actual.

        Args:
            x_actual (np.ndarray): Punto actual en el espacio de búsqueda.
            metodo_paso (str): Método para calcular el paso. Opciones:
                - "exacto" o "e": Paso fijo.
                - "gradiente-hessiano", "gh", "g-h": Basado en gradiente y Hessiana.
                - "backtracking" o "b": Backtracking con condición de Armijo.
            paso_fijo (float): Tamaño de paso fijo (para "exacto"). Debe ser positivo.
            alpha_inicial (float): Valor inicial de alpha (para "backtracking").
            factor_contraccion (float): Factor de contracción (0 < rho < 1).
            c1 (float): Constante de Armijo (0 < c1 < 1).
            max_iter_backtracking (int): Máximo de iteraciones para backtracking.

        Returns:
            float: Tamaño de paso calculado.
        """
        metodo_paso = metodo_paso.lower()
        metodos_validos = {"exacto", "e", "gradiente-hessiano", "gh", "g-h", "backtracking", "b"}

        if metodo_paso not in metodos_validos:
            raise ValueError(f"Método de paso inválido. Opciones: {metodos_validos}")

        if metodo_paso in {"exacto", "e"}:
            if paso_fijo <= 0:
                raise ValueError("El paso fijo debe ser positivo")
            return paso_fijo

        elif metodo_paso in {"gradiente-hessiano", "gh", "g-h"}:
            if not hasattr(self.funcion, 'hessiano'):
                raise AttributeError("Este método requiere una Hessiana definida")
            
            grad = self.funcion.gradiente(x_actual)
            hess = self.funcion.hessiano(x_actual)
            numerador = np.dot(grad, grad)
            denominador = np.dot(grad, hess @ grad)
            
            if np.abs(denominador) < 1e-12:
                return 0.0
            
            return numerador / denominador

        elif metodo_paso in {"backtracking", "b"}:
            self._validar_parametros_backtracking(alpha_inicial, factor_contraccion, c1, max_iter_backtracking)
            
            grad = self.funcion.gradiente(x_actual)
            direccion = self._calcular_direccion_precondicionada(x_actual, grad)
            
            alpha = alpha_inicial
            iteracion = 0
            
            while self._condicion_armijo(x_actual, alpha, direccion, c1):
                alpha *= factor_contraccion
                iteracion += 1
                if iteracion > max_iter_backtracking:
                    self._advertencia_bucle_infinito(x_actual, alpha, direccion, c1)
                    break
            
            return alpha

    def optimizar(self, x_inicial: np.ndarray, criterio_parada: str = "kmax",
                  max_iteraciones: int = 1000, tolerancia: float = 1e-3,
                  metodo_paso: str = "exacto", max_iter_optimizacion: int = 1e5,
                  paso_fijo: float = 1e-3, alpha_inicial: float = 1.0,
                  factor_contraccion: float = 0.3, c1: float = 0.3,
                  max_iter_backtracking: int = 1000,
                  modo_visualizacion: str = "resumen",
                  guardar_resultados: bool = False,
                  archivo_datos: str = "archivo_no_especificado") -> np.ndarray:
        """
        Ejecuta el algoritmo de optimización.

        Args:
            x_inicial (np.ndarray): Punto inicial para la optimización.
            criterio_parada (str): Criterio de parada. Opciones:
                - "kmax": Máximo de iteraciones.
                - "tolx": Cambio en x menor que tolerancia.
                - "tolfunc": Cambio en f(x) menor que tolerancia.
                - "tolgrad": Norma del gradiente menor que tolerancia.
            max_iteraciones (int): Máximo de iteraciones para el criterio "kmax".
            tolerancia (float): Tolerancia para los criterios de parada.
            metodo_paso (str): Método para calcular el paso (ver calcular_paso).
            max_iter_optimizacion (int): Máximo de iteraciones de seguridad.
            paso_fijo (float): Tamaño de paso fijo (para "exacto").
            alpha_inicial (float): Valor inicial de alpha (para "backtracking").
            factor_contraccion (float): Factor de contracción (0 < rho < 1).
            c1 (float): Constante de Armijo (0 < c1 < 1).
            max_iter_backtracking (int): Máximo de iteraciones para backtracking.
            modo_visualizacion (str): Modo de visualización. Opciones:
                - "resumen": Muestra un resumen.
                - "tabular": Muestra una tabla.
                - "graficos": Genera gráficos.
                - "todo": Muestra todo.
            guardar_resultados (bool): Si True, guarda los resultados en archivos.
            archivo_datos (str): Nombre del archivo usado para la optimización.

        Returns:
            np.ndarray: Punto óptimo encontrado.
        """
        self._validar_punto_inicial(x_inicial)
        
        historial = {
            'iteraciones': [],
            'valores_funcion': [],
            'normas_gradiente': [],
            'diferencias_x': []
        }
        
        x_actual = x_inicial.copy()
        grad_actual = -self.funcion.gradiente(x_actual)
        iteracion = 0

        # Iniciar temporizador
        tiempo_inicio = time.time()
        
        while not self._verificar_criterio_parada(x_actual, grad_actual, iteracion, 
                                                  criterio_parada, max_iteraciones,
                                                  tolerancia, historial):
            
            paso = self.calcular_paso(
                x_actual, metodo_paso, paso_fijo,
                alpha_inicial, factor_contraccion, c1, max_iter_backtracking
            )
            
            x_nuevo = x_actual + paso * grad_actual
            
            self._registrar_metricas(historial, iteracion, x_actual, x_nuevo)
            
            x_actual = x_nuevo
            grad_actual = -self.funcion.gradiente(x_actual)
            iteracion += 1
            
            if iteracion > max_iter_optimizacion:
                print("Advertencia: Máximo de iteraciones de seguridad alcanzado")
                break

            if iteracion % 500 == 0:
                print(f"Iteración {iteracion}: f(x) = {self.funcion.evaluar(x_actual)}, ||∇f(x)|| = {np.linalg.norm(grad_actual)}")
        
        # Calcular tiempo total
        tiempo_total = time.time() - tiempo_inicio

        # Visualizar resultados
        self._visualizar_resultados(historial, modo_visualizacion, 
                                    iteracion, criterio_parada, 
                                    metodo_paso, paso_fijo, x_actual, tiempo_total, 
                                    guardar_resultados, tolerancia, archivo_datos,
                                    alpha_inicial, factor_contraccion, c1, 
                                    max_iter_backtracking, max_iter_optimizacion, max_iteraciones)
        
        return x_actual

    # Métodos privados de validación y utilidad
    def _validar_funcion(self, funcion: object) -> None:
        """
        Valida que la función objetivo tenga los métodos y atributos requeridos.

        Args:
            funcion (object): Objeto que representa la función objetivo.

        Raises:
            TypeError: Si falta algún método requerido.
            AttributeError: Si falta el atributo 'dimension'.
        """
        metodos_requeridos = ["evaluar", "gradiente"]
        for metodo in metodos_requeridos:
            if not hasattr(funcion, metodo) or not callable(getattr(funcion, metodo)):
                raise TypeError(f"Función no implementa el método requerido: {metodo}")
        
        if not hasattr(funcion, "dimension") or not isinstance(funcion.dimension, int):
            raise AttributeError("La función debe tener un atributo 'dimension' entero")

    def _validar_punto_inicial(self, x_inicial: np.ndarray) -> None:
        """
        Valida que el punto inicial tenga la dimensión correcta.

        Args:
            x_inicial (np.ndarray): Punto inicial.

        Raises:
            ValueError: Si la dimensión no coincide con la del problema.
        """
        if x_inicial.shape != (self.dimension,):
            raise ValueError(
                f"Dimensión del punto inicial ({x_inicial.shape}) "
                f"no coincide con la dimensión del problema ({self.dimension})"
            )

    def _validar_parametros_backtracking(self, alpha: float, rho: float, c1: float, max_iter: int) -> None:
        """
        Valida los parámetros para el método de backtracking.

        Args:
            alpha (float): Valor inicial de alpha.
            rho (float): Factor de contracción (0 < rho < 1).
            c1 (float): Constante de Armijo (0 < c1 < 1).
            max_iter (int): Máximo de iteraciones.

        Raises:
            ValueError: Si algún parámetro es inválido.
        """
        if any(val is None for val in [alpha, rho, c1, max_iter]):
            raise ValueError("Parámetros incompletos para backtracking")
        
        if alpha <= 0 or rho <= 0 or rho >= 1 or c1 <= 0 or c1 >= 1:
            raise ValueError(
                "Parámetros inválidos para backtracking: "
                f"alpha={alpha}, rho={rho}, c1={c1}"
            )

    def _calcular_direccion_precondicionada(self, x_actual: np.ndarray, gradiente: np.ndarray) -> np.ndarray:
        """
        Calcula la dirección precondicionada para el descenso.

        Args:
            x_actual (np.ndarray): Punto actual.
            gradiente (np.ndarray): Gradiente en el punto actual.

        Returns:
            np.ndarray: Dirección precondicionada.
        """
        try:
            H = self.funcion.hessiano(x_actual)
            diag_H = np.diag(H)
            epsilon = 1e-8
            precondicionador = 1.0 / (np.abs(diag_H) + epsilon)
            return -precondicionador * gradiente
        except Exception as e:
            raise RuntimeError("Error calculando dirección precondicionada") from e

    def _condicion_armijo(self, x_actual: np.ndarray, alpha: float, direccion: np.ndarray, c1: float) -> bool:
        """
        Evalúa la condición de Armijo.

        Args:
            x_actual (np.ndarray): Punto actual.
            alpha (float): Tamaño de paso.
            direccion (np.ndarray): Dirección de descenso.
            c1 (float): Constante de Armijo.

        Returns:
            bool: True si no se cumple la condición de Armijo.
        """
        try:
            f_actual = self.funcion.evaluar(x_actual)
            f_nuevo = self.funcion.evaluar(x_actual + alpha * direccion)
            termino_armijo = f_actual + c1 * alpha * np.dot(self.funcion.gradiente(x_actual), direccion)
            return f_nuevo > termino_armijo
        except Exception as e:
            raise RuntimeError("Error evaluando condición de Armijo") from e

    def _registrar_metricas(self, historial: dict, iteracion: int, 
                            x_actual: np.ndarray, x_nuevo: np.ndarray) -> None:
        """
        Registra las métricas de la iteración actual.

        Args:
            historial (dict): Diccionario para almacenar las métricas.
            iteracion (int): Número de iteración actual.
            x_actual (np.ndarray): Punto actual.
            x_nuevo (np.ndarray): Nuevo punto calculado.
        """
        historial['iteraciones'].append(iteracion)
        historial['valores_funcion'].append(self.funcion.evaluar(x_nuevo))
        historial['normas_gradiente'].append(np.linalg.norm(self.funcion.gradiente(x_nuevo)))
        historial['diferencias_x'].append(np.linalg.norm(x_nuevo - x_actual))

    def _verificar_criterio_parada(self, x_actual: np.ndarray, gradiente: np.ndarray,
                                   iteracion: int, criterio: str, max_iter: int,
                                   tolerancia: float, historial) -> bool:
        """
        Verifica si se cumple el criterio de parada.

        Args:
            x_actual (np.ndarray): Punto actual.
            gradiente (np.ndarray): Gradiente en el punto actual.
            iteracion (int): Número de iteración actual.
            criterio (str): Criterio de parada.
            max_iter (int): Máximo de iteraciones.
            tolerancia (float): Tolerancia para el criterio.

        Returns:
            bool: True si se cumple el criterio de parada.
        """
        if criterio == "kmax":
            return iteracion >= max_iter
        
        if criterio == "tolx":
            return historial['diferencias_x'][-1] < tolerancia if iteracion > 0 else False
        
        if criterio == "tolfunc":
            return abs(historial['valores_funcion'][-1] - historial['valores_funcion'][-2]) < tolerancia if iteracion > 1 else False
        
        if criterio == "tolgrad":
            return np.linalg.norm(gradiente) < tolerancia
        
        return False

    def _visualizar_resultados(self, historial: dict, modo: str, 
                               iteraciones_totales: int, criterio_parada: str,
                               metodo_paso: str, paso_fijo: float, 
                               x_final: np.ndarray, tiempo_total: float,
                               guardar_resultados: bool, tolerancia: float,
                               archivo_datos: str,
                               alpha_inicial: float = None, factor_contraccion: float = None, 
                               c1: float = None, max_iter_backtracking: int = None, 
                               max_iter_optimizacion: int = None, max_iteraciones: int = None) -> None:
        """
        Visualiza los resultados de la optimización.

        Args:
            historial (dict): Historial de métricas.
            modo (str): Modo de visualización ("resumen", "tabular", "graficos", "todo").
            iteraciones_totales (int): Número total de iteraciones realizadas.
            criterio_parada (str): Criterio de parada utilizado.
            metodo_paso (str): Método de cálculo del paso.
            paso_fijo (float): Tamaño de paso fijo (si aplica).
            x_final (np.ndarray): Punto óptimo encontrado.
            tiempo_total (float): Tiempo total de ejecución.
            guardar_resultados (bool): Si True, guarda los resultados.
            tolerancia (float): Tolerancia utilizada.
            archivo_datos (str): Nombre del archivo usado para la optimización.
        """
        if guardar_resultados:
            os.makedirs("resultados", exist_ok=True)

        if modo in {"tabular", "todo"}:
            self._mostrar_tabla(historial, guardar_resultados)

        if modo in {"resumen", "todo"}:
            self._mostrar_resumen(iteraciones_totales, criterio_parada, metodo_paso, paso_fijo, x_final, tiempo_total, guardar_resultados, tolerancia, archivo_datos, alpha_inicial, factor_contraccion, c1, max_iter_backtracking, max_iter_optimizacion, max_iteraciones)
        
        if modo in {"graficos", "todo"}:
            self._generar_graficos(historial, guardar_resultados)

    def _mostrar_resumen(self, iteraciones: int, criterio: str, 
                         metodo_paso: str, paso: float, x_final: np.ndarray, 
                         tiempo_total: float, guardar_resultados: bool, 
                         tolerancia: float, archivo_datos: str,
                         alpha_inicial: float = None, factor_contraccion: float = None, 
                         c1: float = None, max_iter_backtracking: int = None, 
                         max_iter_optimizacion: int = None, max_iteraciones: int = None) -> None:
        """
        Muestra un resumen de los resultados de la optimización.

        Args:
            iteraciones (int): Número total de iteraciones realizadas.
            criterio (str): Criterio de parada utilizado.
            metodo_paso (str): Método de cálculo del paso.
            paso (float): Tamaño de paso fijo (si aplica).
            x_final (np.ndarray): Punto óptimo encontrado.
            tiempo_total (float): Tiempo total de ejecución.
            guardar_resultados (bool): Si True, guarda los resultados.
            tolerancia (float): Tolerancia utilizada.
            archivo_datos (str): Nombre del archivo usado para la optimización.
        """
        resumen = []
        resumen.append(f"\n\n{' RESUMEN DE OPTIMIZACIÓN ':=^80}")
        resumen.append(f"• Archivo de datos: {archivo_datos}")
        resumen.append(f"• Iteraciones completadas: {iteraciones}")
        resumen.append(f"• Criterio de parada: {criterio}")
        resumen.append(f"• Tolerancia: {tolerancia:.2e}")
        resumen.append(f"• Método de paso: {metodo_paso}")
        
        if metodo_paso == "exacto":
            resumen.append(f"• Tamaño de paso fijo: {paso:.2e}")
        elif metodo_paso in {"backtracking", "b"}:
            resumen.append(f"• Alpha inicial: {alpha_inicial:.2e}")
            resumen.append(f"• Factor de contracción: {factor_contraccion:.2e}")
            resumen.append(f"• c1: {c1:.2e}")
            resumen.append(f"• Máximo iteraciones backtracking: {max_iter_backtracking}")
        
        resumen.append(f"• Máximo iteraciones optimización: {max_iter_optimizacion}")
        if criterio == "kmax":
            resumen.append(f"• Máximo iteraciones criterio kmax: {max_iteraciones}")
        
        horas, resto = divmod(tiempo_total, 3600)
        minutos, segundos = divmod(resto, 60)
        resumen.append(f"• Tiempo total requerido: {int(horas):02}:{int(minutos):02}:{int(segundos):02}")
        resumen.append(f"• Valor función final: {self.funcion.evaluar(x_final):.4e}")
        resumen.append(f"• Norma del gradiente final: {np.linalg.norm(self.funcion.gradiente(x_final)):.4e}")
        resumen.append("=" * 80)

        print("\n".join(resumen))

        if guardar_resultados:
            with open("resultados/resumen.txt", "w") as f:
                f.write("\n".join(resumen))

    def _mostrar_tabla(self, historial: dict, guardar_resultados: bool) -> None:
        """
        Muestra una tabla con el historial de métricas.

        Args:
            historial (dict): Historial de métricas.
            guardar_resultados (bool): Si True, guarda la tabla en un archivo CSV.
        """
        try:
            df = pd.DataFrame({
                'Iteración': historial['iteraciones'],
                'Valor función': ["%.4e" % val for val in historial['valores_funcion']],
                'Norma gradiente': ["%.4e" % val for val in historial['normas_gradiente']],
                'Δx': ["%.4e" % val for val in historial['diferencias_x']]
            })
            print("\nHistorial de convergencia:")
            print(df.to_string(index=False))

            if guardar_resultados:
                df.to_csv("resultados/historial_tabla.csv", index=False)
        except Exception as e:
            print(f"Error generando tabla: {str(e)}")

    def _generar_graficos(self, historial: dict, guardar_resultados: bool) -> None:
        """
        Genera gráficos de las métricas de convergencia.

        Args:
            historial (dict): Historial de métricas.
            guardar_resultados (bool): Si True, guarda los gráficos en archivos PNG.
        """
        try:
            # Gráfico de convergencia de la función objetivo
            plt.figure(figsize=(6, 5))
            plt.plot(historial['iteraciones'], np.log(historial['valores_funcion']))
            plt.title("Convergencia de la función objetivo")
            plt.xlabel("Iteración")
            plt.ylabel("log(f(x))")
            plt.tight_layout()
            if guardar_resultados:
                plt.savefig("resultados/convergencia_funcion.png")
            plt.show()

            # Gráfico de evolución de la norma del gradiente
            plt.figure(figsize=(6, 5))
            plt.plot(historial['iteraciones'], np.log(historial['normas_gradiente']))
            plt.title("Evolución de la norma del gradiente")
            plt.xlabel("Iteración")
            plt.ylabel("log(||∇f(x)||)")
            plt.tight_layout()
            if guardar_resultados:
                plt.savefig("resultados/evolucion_gradiente.png")
            plt.show()

            # Gráfico de cambio en los parámetros
            plt.figure(figsize=(6, 5))
            plt.plot(historial['iteraciones'], np.log(historial['diferencias_x']))
            plt.title("Cambio en los parámetros")
            plt.xlabel("Iteración")
            plt.ylabel("log(||xₖ₊₁ - xₖ||)")
            plt.tight_layout()
            if guardar_resultados:
                plt.savefig("resultados/cambio_parametros.png")
            plt.show()
        except Exception as e:
            print(f"Error generando gráficos: {str(e)}")

    def _advertencia_bucle_infinito(self, x_actual: np.ndarray, alpha: float, 
                                    direccion: np.ndarray, c1: float) -> None:
        """
        Muestra una advertencia si se detecta un bucle infinito (o mayor al número de iteracciones permitidas) en el método de backtracking.

        Args:
            x_actual (np.ndarray): Punto actual.
            alpha (float): Tamaño de paso.
            direccion (np.ndarray): Dirección de descenso.
            c1 (float): Constante de Armijo.
        """
        print("\n¡ADVERTENCIA! Bucle infinito detectado en backtracking")
        print("Valores actuales:")
        print(f"• x_actual: {x_actual}")
        print(f"• alpha: {alpha}")
        print(f"• Dirección: {direccion}")
        print(f"• Condición Armijo: {self._condicion_armijo(x_actual, alpha, direccion, c1)}")