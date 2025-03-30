import numpy as np
from RegresionLogistica import RegresionLogistica
from DescensoGradiente import DescensoGradiente
from Auxiliar import Auxiliar
from sklearn.preprocessing import StandardScaler
import random

def obtener_parametro(prompt, tipo, default, opciones_validas: list = None, limite_inferior=None, limite_superior=None):
    """
    Solicita un parámetro al usuario con un valor predeterminado.

    Args:
        prompt (str): Mensaje para el usuario.
        tipo (type): Tipo de dato esperado.
        default: Valor predeterminado.
        opciones_validas (list, optional): Lista de opciones válidas. Si se proporciona, 
                                            la entrada debe estar en esta lista.
        limite_inferior (optional): Límite inferior para valores numéricos.
        limite_superior (optional): Límite superior para valores numéricos.

    Returns:
        El valor ingresado por el usuario o el valor predeterminado.
    """
    entrada = input(f"{prompt} (Presiona Enter para usar {default}): ")
    if entrada.strip() == "":
        return default
    try:
        valor = tipo(entrada)
        if opciones_validas is not None and tipo == str and valor not in opciones_validas:
            print(f"Opción no válida. Usando valor predeterminado: {default}")
            return default
        if limite_inferior is not None and valor <= limite_inferior:
            print(f"El valor es menor o igual que el límite inferior ({limite_inferior}). Usando valor predeterminado: {default}")
            return default
        if limite_superior is not None and valor >= limite_superior:
            print(f"El valor es mayor o igual que el límite superior ({limite_superior}). Usando valor predeterminado: {default}")
            return default
        return valor
    except ValueError:
        print(f"Entrada inválida. Usando valor predeterminado: {default}")
        return default

def menu():
    """
    Menú interactivo para configurar parámetros y realizar pruebas.
    """
    aux = Auxiliar()
    archivo = "datos_default.csv"
    datos, categorias= None, None
    X_train, y_train, X_test, y_test = None, None, None, None
    modelo, optimizador, parametros_optimos = None, None, None

    while True:
        print("\n--- Menú Principal ---\n")
        
        print("1. Leer y preparar datos (csv solamente)")
        print("2. Configurar parámetros de optimización y entrenar modelo")
        print("3. Probar el modelo con datos de prueba o entrenamiento, o predecir nuevos datos")
        print("4. Salir\n")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            # Configurar parámetros del archivo a leer
            archivo_a_usar = input("\nNombre del archivo CSV (Presiona Enter para usar 'datos_default.csv'): ") or archivo
            categoria_idx = obtener_parametro("Índice de la columna de categorías", int, 0, limite_inferior=-0.0001)
            prueba_size = obtener_parametro("Porcentaje de datos para prueba (0.0 - 1.0)", float, 0.15, limite_inferior=0, limite_superior=1)

            # Cargar datos
            datos, categorias = aux.cargarDatos(
                fuente=archivo_a_usar,
                tipo="csv",
                agregar_columna_unos=True,
                categoria_idx=categoria_idx,
                conservar_nombres=False,
                sep=",",
                header=0
            )

            print("\nDatos cargados correctamente.")
            print("Número de observaciones: ", datos.shape[0])
            print("Número de características: ", datos.shape[1])

            # Dividir y preprocesar datos
            X_train, y_train, X_test, y_test = aux.dividir_y_preprocesar(
                datos, categorias, prueba_size=prueba_size, aleatorio=True, semilla=random.randint(0, 1000), normalizar=True
            )
            print("\nDatos cargados y divididos correctamente.")

        elif opcion == "2":

            if datos is None:
                print("\nPrimero debes cargar los datos.")
                continue

            # Configurar parámetros de optimización
            lambda1 = obtener_parametro("\nParámetro de regularización (lambda)", float, 0.01, limite_inferior=0)
            criterio_parada = obtener_parametro("Criterio de parada (tolgrad/tolx/tolfunc)", str, "tolgrad", opciones_validas=["tolgrad", "tolx", "tolfunc"])
            tolerancia = obtener_parametro("Tolerancia para la convergencia", float, 1e-5, limite_inferior=0)

            # Configurar método de paso
            metodo_paso = obtener_parametro("Método de paso (exacto/gradiente-hessiano/backtracking)", str, "backtracking", opciones_validas=["exacto", "gradiente-hessiano", "bactracking"])
            if metodo_paso == "exacto":
                paso_fijo = obtener_parametro("Tamaño de paso constante", float, 1e-3, limite_inferior=0)
            elif metodo_paso == "kmax":
                max_iteraciones = obtener_parametro("Número máximo de iteraciones para kmax", int, 5000, limite_inferior=100)
            elif metodo_paso == "backtracking":
                alpha_inicial = obtener_parametro("Alpha inicial", float, 1.0, limite_inferior=0)
                factor_contraccion = obtener_parametro("Factor de contracción", float, 0.3, limite_inferior=0, limite_superior=1)
                c1 = obtener_parametro("Condición de Armijo (c1)", float, 0.3)
                max_iter_backtracking = obtener_parametro("Máximo de iteraciones para backtracking", int, 1000, limite_inferior=50)

            max_iteraciones_opt = obtener_parametro("Máximo de iteraciones para optimización (En caso de convergencia lenta)", int, 5000, limite_inferior=100)
            
            # Configurar vector inicial
            usar_aleatorio = input("¿Usar vector inicial aleatorio? (s/n)(Enter se toma como 's'): ").lower() in ["s", ""]
            if usar_aleatorio:
                x_0 = np.random.normal(0, 0.01, X_train.shape[1])
            else:
                x_0 = np.array([float(x) for x in input("Ingresa el vector inicial separado por comas (Ej: 1,3,4.9): ").split(",")])
                if len(x_0) != X_train.shape[1]:
                    print(f"El vector inicial debe tener {X_train.shape[1]} elementos. Usando vector aleatorio.")
                    x_0 = np.random.normal(0, 0.01, X_train.shape[1])
            # Configurar visualización de resultados
            mostrar_resultados = obtener_parametro("Forma de mostrar resultados (tabular/resumen/graficos/todo)", str, "resumen", opciones_validas=["tabular", "resumen", "graficos", "todo"])
            guardar_resultados = input("¿Guardar resultados? (s/n)(Enter se toma como 's'): ").lower() in ["s", ""]

            # Crear modelo y optimizador
            modelo = RegresionLogistica(X=X_train, y=y_train, lambda_=lambda1, ajustar_intercepto=False, mensaje_inicio=True)
            optimizador = DescensoGradiente(funcion=modelo)

            # Optimizar
            parametros_optimos = optimizador.optimizar(
                x_inicial=x_0,
                criterio_parada=criterio_parada,
                max_iteraciones=max_iteraciones if metodo_paso == "kmax" else 1000,
                tolerancia=tolerancia,
                metodo_paso=metodo_paso,
                max_iter_optimizacion=max_iteraciones_opt,
                paso_fijo=paso_fijo if metodo_paso == "exacto" else 1e-03,
                alpha_inicial=alpha_inicial if metodo_paso == "backtracking" else 1,
                factor_contraccion=factor_contraccion if metodo_paso == "backtracking" else 0.3,
                c1=c1 if metodo_paso == "backtracking" else 0.3,
                max_iter_backtracking=max_iter_backtracking if metodo_paso == "backtracking" else 1000,
                modo_visualizacion=mostrar_resultados,
                guardar_resultados=guardar_resultados,
                archivo_datos=archivo,
            )
            print("\nOptimización completada.")
            if guardar_resultados:
                print("Los resultados se guardaron en la carpeta 'resultados'.")
        
            mostrar_parametros_optimos = input("¿Mostrar parámetros óptimos encontrados? (s/n): ").lower() == "s"
            
            if mostrar_parametros_optimos:
                print("\nParámetros óptimos encontrados:")
                for i, param in enumerate(parametros_optimos):
                    print(f"θ{i}: {param:.4f}")

        elif opcion == "3":

            if parametros_optimos is None:
                print("\nPrimero debes optimizar el modelo.")
                continue

            # Seleccionar datos para prueba
            tipo_datos = obtener_parametro("\n¿Probar con datos de prueba, entrenamiento o ingresar nuevos datos? (prueba/entrenamiento/nuevo)", str, "prueba", opciones_validas=["prueba", "entrenamiento", "nuevo"])

            if tipo_datos == "prueba":
                X, y = X_test, y_test
            elif tipo_datos == "entrenamiento":
                X, y = X_train, y_train
            elif tipo_datos == "nuevo":
                archivo_deseado = obtener_parametro("\nNombre del archivo CSV para datos nuevos", str, "datos_a_predecir.csv")
                nuevos_datos = aux.cargarDatos(
                    fuente=archivo_deseado,
                    tipo="csv",
                    agregar_columna_unos=True,
                    categoria_idx=datos.shape[1]-1,  # No hay categorías
                    conservar_nombres=False,
                    sep=",",
                    header=0
                )
                scaler = StandardScaler()
                nuevos_datos = scaler.fit_transform(nuevos_datos)
                X, y = nuevos_datos, None
                print("\nProceso de cargar datos nuevos terminado.")

            # Configurar umbral
            usar_umbral_optimo = input("¿Buscar umbral óptimo? (s/n)(Enter se toma como 's'): ").lower() in ["s", ""]
            if usar_umbral_optimo:
                z_entr = X_train @ parametros_optimos
                umbral_optimo = aux.encontrar_umbral_optimo(y_train, modelo._funcion_sigmoide(z_entr))
                print(f"Umbral óptimo encontrado: {umbral_optimo:.4f}")
            else:
                umbral_optimo = obtener_parametro("Ingresa el umbral deseado", float, 0.5, limite_inferior=0,  limite_superior=1)
            
            if tipo_datos == "prueba" or tipo_datos == "entrenamiento":
                # Configurar parámetros de graficarConfusion
                titulo_matriz = obtener_parametro("\nTítulo para matriz de confusión", str, "Matriz Confusión")
                etiquetas = input("Etiquetas separadas por comas (Presiona Enter para usar predeterminadas: 'Clase 0' y 'Clase 1'): ").split(",")
                etiquetas = etiquetas if etiquetas != [''] else ['Clase 0', 'Clase 1']
                guardar_graficos = input("¿Guardar gráfico? (s/n)(Enter se toma como 's'): ").lower() in ["s", ""]
                
                if guardar_graficos:
                    nombre_grafico = obtener_parametro("Nombre del gráfico a guardar (sin extensión)", str, "matriz_confusion")

            # Realizar predicciones y calcular métricas
            z = X @ parametros_optimos
            predicciones = (modelo._funcion_sigmoide(z) >= umbral_optimo).astype(int)
            if y is not None:
                exactitud = aux._calcularExactitud(y, predicciones)
                matriz_confusion = aux._matrizConfusion(y, predicciones)
                print(f"Exactitud: {exactitud:.4f}")
                aux.graficarConfusion(matriz_confusion, etiquetas=etiquetas, guardar=guardar_graficos, titulo=titulo_matriz, nombre_sin_ext=nombre_grafico)
            else:
                print("\n--- Predicciones ---\n")
                for i, prediccion in enumerate(predicciones, start=1):
                    print(f"Fila {i}: Clase Predicha -> {prediccion}")


        elif opcion == "4":
            print("\nSaliendo del programa...\n")
            break
        else:
            print("\nOpción inválida o no disponible. Intenta de nuevo.")

if __name__ == "__main__":
    menu()