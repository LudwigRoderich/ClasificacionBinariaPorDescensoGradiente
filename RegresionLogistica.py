import numpy as np

class RegresionLogistica:
    def __init__(self, X: np.array = None, y: np.array = None, 
                 lambda_: float = 1.0, ajustar_intercepto: bool = True, mensaje_inicio: bool = False):
        """
        Implementación profesional de regresión logística con regularización L2.

        La regresión logística es un modelo de clasificación binaria que utiliza la función sigmoide
        para predecir probabilidades. Este modelo incluye regularización L2 para evitar el sobreajuste.

        Parámetros:
        - X (np.array): Matriz de características de dimensión (muestras, características). 
                        Cada fila representa una muestra y cada columna una característica.
        - y (np.array): Vector de etiquetas binarias (0 o 1) de tamaño (muestras, 1).
        - lambda_ (float): Hiperparámetro de regularización (debe ser ≥ 0). Controla la penalización
                           aplicada a los coeficientes del modelo.
        - ajustar_intercepto (bool): Si es True, añade automáticamente una columna de unos a X para
                                     incluir el término de intercepto en el modelo.
        - mensaje_inicio (bool): Si es True, imprime un mensaje con información inicial del modelo.

        Excepciones:
        - ValueError: Si X o y no son proporcionados, si sus dimensiones no coinciden, si y contiene
                      valores distintos de 0 y 1, o si lambda_ es negativo.
        """
        # Validaciones
        if X is None or y is None:
            raise ValueError("X e y son requeridos")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensiones incompatibles entre X e y")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y debe contener solo 0s y 1s")
        if lambda_ < 0:
            raise ValueError("lambda debe ser ≥ 0")

        # Preprocesamiento automático
        if ajustar_intercepto:
            if not np.all(X[:, 0] == 1):  # Evitar duplicar intercepto
                X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        self.X = X
        self.y = y
        self.m, self.n = X.shape  # m = muestras, n = características (incluye intercepto)
        self.ajustar_intercepto = ajustar_intercepto
        
        # Configuración de regularización
        self.lambda_efectivo = lambda_ / self.m  # Escalado por tamaño de dataset
        self.dimension = self.n  # Para optimizadores
        if mensaje_inicio:
            print("\n\n")
            print("=" * 50)
            print("¡Regresión Logística Inicializada!")
            print(f"• Número de muestras: {self.m}")
            print(f"• Número de características (sin incluir intercepto): {self.n - 1}")
            print(f"• Regularización λ: {lambda_}")
            print(f"• Intercepto ajustado automáticamente: {'Sí' if ajustar_intercepto else 'No'}")
            print("=" * 50)

    def _funcion_sigmoide(self, z: np.array) -> np.array:
        """
        Calcula la función sigmoide de manera numéricamente estable.

        Parámetros:
        - z (np.array): Vector o matriz de valores para los cuales se calculará la sigmoide.

        Retorna:
        - np.array: Valores transformados por la función sigmoide.
        """
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    def evaluar(self, theta: np.array) -> float:
        """
        Calcula el costo regularizado del modelo.

        Parámetros:
        - theta (np.array): Vector de parámetros del modelo de tamaño (características, 1).

        Retorna:
        - float: Valor del costo regularizado.

        Excepciones:
        - ValueError: Si theta no tiene la dimensión esperada.
        """
        if theta.shape != (self.n,):
            raise ValueError(f"theta debe tener dimensión {self.n}")
        
        z = self.X @ theta
        termino_verosimilitud = np.mean(
            (1 - self.y) * z + np.logaddexp(0, -z)
        )
        
        inicio_regularizacion = 1 if self.ajustar_intercepto else 0
        termino_regularizacion = 0.5 * self.lambda_efectivo * np.sum(theta[inicio_regularizacion:]**2)
        
        return termino_verosimilitud + termino_regularizacion

    def gradiente(self, theta: np.array) -> np.array:
        """
        Calcula el gradiente del costo regularizado.

        Parámetros:
        - theta (np.array): Vector de parámetros del modelo de tamaño (características, 1).

        Retorna:
        - np.array: Gradiente del costo con respecto a los parámetros theta.
        """
        z = self.X @ theta
        error = self._funcion_sigmoide(z) - self.y
        grad = (self.X.T @ error) / self.m
        
        if self.ajustar_intercepto:
            grad[1:] += self.lambda_efectivo * theta[1:]
        else:
            grad += self.lambda_efectivo * theta
            
        return grad

    def hessiano(self, theta: np.array) -> np.array:
        """
        Calcula la matriz Hessiana del costo regularizado.

        Parámetros:
        - theta (np.array): Vector de parámetros del modelo de tamaño (características,).

        Retorna:
        - np.array: Matriz Hessiana de tamaño (características, características).
        """
        z = self.X @ theta
        sigma = self._funcion_sigmoide(z)
        D = np.diag(sigma * (1 - sigma))
        
        hess = (self.X.T @ D @ self.X) / self.m
        
        if self.ajustar_intercepto:
            hess[1:, 1:] += self.lambda_efectivo * np.eye(self.n - 1)
        else:
            hess += self.lambda_efectivo * np.eye(self.n)
            
        return hess