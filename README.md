# Modelo de supervivencia del Titanic: red neuronal desde cero

## Autores

| Nombre | Correo |
|---|---|
| Jacobo Bascón Montalvo | jacobo.bascon@cunef.edu |
| Mencía Marzal Higuero | mencia.marzal@cunef.edu |
| Guillem Alcubilla Pueyo | guillem.alcubilla@cunef.edu |

---

## 1. Problema

El objetivo de este proyecto es predecir si un pasajero del Titanic sobrevivió o no al naufragio a partir de información disponible sobre sus características personales y de viaje.

Se trata de un problema de clasificación binaria, donde la variable objetivo es `Survived`, que toma los valores:

* 0 → el pasajero no sobrevivió
* 1 → el pasajero sobrevivió

El reto principal es encontrar patrones en los datos que expliquen la supervivencia. Variables como el sexo, la clase del billete o la edad son candidatas obvias, pero el objetivo es ver si una red neuronal construida a mano es capaz de capturar esas relaciones.

---

## 2. Datos

### 2.1 Origen del dataset

El dataset proviene del concurso **"Titanic: Machine Learning from Disaster"** de Kaggle:

[https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)

**Ventajas:**

* Dataset muy conocido y bien documentado, con bastante bibliografía sobre qué variables funcionan bien
* Combina distintos tipos de variables (numéricas, categóricas, texto libre) lo que permite trabajar todas las etapas del pipeline de forma completa
* El tamaño (891 registros) es manejable sin necesitar potencia de cómputo especial
* Los datos son reales, lo que da contexto para interpretar los resultados

**Desventajas:**

* El tamaño es pequeño para entrenar una red neuronal, lo que limita la complejidad del modelo sin riesgo de sobreajuste
* Alta tasa de nulos en `Cabin` (~77%), lo que obliga a tomar decisiones de imputación que pueden introducir sesgo
* Ligero desbalance de clases: ~38% supervivientes vs ~62% no supervivientes
* Algunas variables importantes (posición en el barco, orden de evacuación) no están disponibles en el dataset

### 2.2 Descripción del dataset

El archivo `test.csv` de Kaggle no incluye la variable objetivo (`Survived`), así que no sirve para evaluar el modelo. Por eso, se trabaja únicamente con `train.csv` y se hace un split interno 80/20 para tener un conjunto de test con etiquetas reales.

Las variables originales son:

| Variable | Descripción |
|---|---|
| `PassengerId` | Identificador único del pasajero |
| `Survived` | Variable objetivo (0 = no sobrevivió, 1 = sobrevivió) |
| `Pclass` | Clase del billete (1ª, 2ª o 3ª) |
| `Name` | Nombre completo del pasajero |
| `Sex` | Sexo |
| `Age` | Edad |
| `SibSp` | Número de hermanos o cónyuge a bordo |
| `Parch` | Número de padres o hijos a bordo |
| `Ticket` | Número de billete |
| `Fare` | Tarifa pagada |
| `Cabin` | Número de cabina |
| `Embarked` | Puerto de embarque (C = Cherburgo, Q = Queenstown, S = Southampton) |

### 2.3 Problemas del dataset

El dataset tiene bastantes cosas que arreglar antes de poder usarlo:

* `Age` tiene un ~20% de valores nulos
* `Cabin` tiene casi un 80% de nulos, lo que hace imposible imputarla de forma razonable
* `Embarked` tiene 2 nulos (fácil de resolver)
* `Name`, `Ticket` y `PassengerId` no aportan información directa al modelo
* Las variables numéricas están en escalas muy distintas (`Fare` puede llegar a 500, `Parch` llega a 6)
* Las variables categóricas hay que convertirlas a números
* `Fare` presenta outliers significativos (pasajeros con tarifas muy superiores a la media)

---

## 3. Planteamiento técnico

### 3.1 Enfoque elegido: red neuronal

Se aborda el problema como una **clasificación binaria** mediante una **red neuronal feedforward** implementada desde cero en numpy, sin usar librerías de deep learning.

La elección de una red neuronal frente a otros modelos más simples se justifica por:

* **Relaciones no lineales**: las interacciones entre variables (sexo + clase + edad + tarifa) no son estrictamente lineales. Una regresión logística aprende cada variable de forma independiente, pero no captura automáticamente que "mujer de tercera clase con familia grande" tiene una probabilidad de supervivencia diferente a la suma de sus partes. La red neuronal sí puede aprender estas combinaciones en las capas ocultas.
* **Objetivo académico**: implementar el backpropagation desde cero obliga a entender la matemática real detrás del entrenamiento, que es parte central del contenido de la asignatura.
* **Flexibilidad**: la arquitectura puede ajustarse fácilmente cambiando el número de capas y neuronas.

La contrapartida es que con ~700 ejemplos de entrenamiento el modelo puede tender al sobreajuste, lo que se controla con early stopping y una arquitectura moderada.

### 3.2 Métricas de evaluación

Se reportan cuatro métricas: **accuracy**, **precisión**, **recall** y **F1-score**.

* **Accuracy**: proporción total de predicciones correctas. Es la más intuitiva y funciona razonablemente aquí porque el desbalance de clases no es extremo (~38/62).
* **Precisión**: de los que predecimos que sobreviven, ¿cuántos realmente sobrevivieron? Penaliza los falsos positivos.
* **Recall (sensibilidad)**: de los que realmente sobrevivieron, ¿cuántos detectamos? Penaliza los falsos negativos.
* **F1-score**: media armónica de precisión y recall. Es más informativo que el accuracy cuando hay cierto desbalance, porque equilibra ambas métricas.

Usar solo accuracy sería engañoso: un modelo que prediga siempre "no sobrevive" obtendría ~62% de accuracy sin haber aprendido nada. El F1-score penaliza ese tipo de solución trivial.

---

## 4. Preprocesado y feature engineering

Todo el preprocesado está en `00_preprocesado.ipynb`. Lo más importante del proceso es que **primero se hace el split y luego el feature engineering**, para evitar data leakage. Si se hiciera al revés, información del test contaminaría el ajuste de imputaciones y el scaler, lo que daría métricas optimistas pero irreales.

### 4.1 Separación del dataset

Se aplica un **split estratificado 80/20** antes de cualquier transformación:

* **80% Train** (712 registros): se usa para ajustar todas las transformaciones (imputaciones, scaler, encodings) y entrenar el modelo.
* **20% Test** (179 registros): se reserva completamente hasta la evaluación final. No se toca durante el desarrollo.

La estratificación garantiza que la proporción de supervivientes (~38%) sea la misma en ambos conjuntos. Se elige 80/20 porque con solo 891 registros totales necesitamos la mayor cantidad posible de datos para entrenar, pero sin dejar el test tan pequeño que las métricas no sean representativas.

Además, en el notebook de modelado se hace un **segundo split** dentro del 80% de train: 80% para entrenamiento real y 20% para validación durante el entrenamiento (early stopping). Esto da tres conjuntos: train (~569), validación (~143) y test (179). El test nunca se usa para tomar decisiones de entrenamiento.

### 4.2 Feature engineering

Antes de limpiar nulos, se crean nuevas variables que pueden aportar información útil:

**Title** — El nombre de cada pasajero contiene el título (Mr., Mrs., Miss...). Esto refleja sexo, edad aproximada y estatus social a la vez. Los títulos poco frecuentes se agrupan en "Rare" y se normalizan variantes de otros idiomas (Mlle → Miss, Mme → Mrs).

**FamilySize y FamilyCategory** — `FamilySize = SibSp + Parch + 1`. Además se categoriza en Alone (1), Small (2-4) y Large (5+), ya que la relación con la supervivencia no es lineal: viajar solo o en familia muy grande reduce las probabilidades.

**Deck** — Se extrae la primera letra de `Cabin` (la cubierta del barco). Los nulos se marcan como "Unknown". La cubierta está relacionada con la ubicación física en el barco y la clase social.

**TicketGroup** — Número de pasajeros que comparten el mismo ticket. Captura grupos que viajaban juntos aunque no fueran familia. La frecuencia se calcula solo con train para no filtrar información de test.

**AgeGroup** — Se discretiza `Age` en 4 categorías: Child (0-12), Teenager (12-18), Adult (18-60) y Elderly (60+). La relación entre edad y supervivencia no es lineal, así que agrupar ayuda al modelo.

### 4.3 Gestión de nulos

* **Age**: se imputa con la mediana por grupo (`Pclass` + `Title`). Así, un Mr. de 3ª clase recibe la mediana de los Mr. de 3ª clase, no la mediana global.
* **Embarked**: se imputa con la moda (Southampton).
* **Cabin**: se elimina directamente. Con un 80% de nulos no tiene sentido imputarla; la información útil ya está en `Deck`.
* También se eliminan `PassengerId`, `Name` y `Ticket`.

### 4.4 Codificación y escalado

Se aplica **One-Hot Encoding** con `drop_first=True` a todas las variables categóricas para evitar multicolinealidad.

Train y test se concatenan temporalmente antes del encoding para garantizar que las dos matrices acaben con exactamente las mismas columnas.

Las variables numéricas se estandarizan con `StandardScaler` ajustado **solo con train**. Aplicar el scaler sobre test (sin re-ajustarlo) asegura que no se filtra información del test en la transformación.

---

## 5. Análisis exploratorio (EDA)

El EDA está en `01_eda.ipynb`. Los objetivos son: verificar la coherencia entre train y test, analizar distribuciones, detectar outliers y entender qué variables están más relacionadas con la supervivencia.

**Estructura**: ambos datasets tienen las mismas columnas, tipos y sin nulos tras el preprocesado.

**Distribuciones**: se comparan histogramas de todas las variables entre train y test. Las distribuciones son muy similares en ambos conjuntos, lo que indica que el split fue correcto.

**Outliers**: `Fare` es la variable con más valores extremos. Algunos pasajeros de primera clase pagaron tarifas muy superiores a la media (suites de lujo). Tras el StandardScaler, estos outliers aparecen como z-scores > 3. Dado que la red neuronal es menos sensible a outliers que un modelo lineal, se decide mantenerlos sin truncar.

**Variables vs Survived**:

* **Sexo**: las mujeres tienen una tasa de supervivencia mucho mayor que los hombres. Es la variable más discriminante.
* **Clase**: a mayor clase, mayor supervivencia. Los de primera clase tenían más acceso a los botes salvavidas.
* **Tarifa**: relacionada con la clase; a mayor tarifa pagada, más probabilidad de sobrevivir.
* **Edad**: sola no parece muy discriminante, aunque los niños tenían prioridad.
* **Tamaño de familia**: las familias pequeñas sobreviven más. Viajar solo o en grupo muy grande baja las probabilidades.

**Correlación con Survived**: las variables más correlacionadas son `Sex_male` (negativa), `Pclass` (negativa), `Title_Mr` (negativa) y `Fare` (positiva).

---

## 6. Modelo

El modelo está implementado completamente desde cero en `02_model.ipynb`, sin usar ninguna librería de deep learning. Solo numpy.

### 6.1 Arquitectura

Red neuronal feedforward con la siguiente arquitectura:

```
Entrada → 32 → 32 → 16 → 16 → 1 (salida)
```

* Capas ocultas: activación **ReLU**
* Capa de salida: activación **Sigmoid** (devuelve una probabilidad entre 0 y 1)

### 6.2 Inicialización de pesos

* Capas con ReLU: inicialización **He**
* Capa de salida con Sigmoid: inicialización **Xavier**
* Sesgos inicializados a cero

### 6.3 Entrenamiento

* **Función de coste**: entropía cruzada binaria
* **Optimización**: descenso del gradiente batch
* **Learning rate**: 0.001
* **Épocas máximas**: 10.000
* **Early stopping**: se detiene si el val loss no mejora más de 1e-4 en 50 épocas consecutivas. Se guardan los mejores pesos durante el entrenamiento y se restauran al final.

El backpropagation está implementado a mano. Para la capa de salida (Sigmoid + cross-entropy) el gradiente se simplifica a `A - y`. Para las capas ocultas con ReLU, la derivada necesita la preactivación `z` (no la activación `a`, que ha perdido el signo de los valores negativos).

### 6.4 Selección de variables

Antes de entrenar se eliminan las variables con correlación absoluta menor a 0.01 con `Survived` para reducir ruido.

---

## 7. Resultados

Los resultados finales del modelo sobre los tres conjuntos son aproximadamente:

| Conjunto | Accuracy | Precisión | Recall | F1-score |
|---|---|---|---|---|
| Train | ~0.87 | ~0.85 | ~0.83 | ~0.84 |
| Validación | ~0.83 | ~0.81 | ~0.79 | ~0.80 |
| Test | ~0.82 | ~0.80 | ~0.78 | ~0.79 |

La diferencia entre train y test es pequeña, lo que indica que el modelo generaliza bien y no hay overfitting grave. El early stopping ayuda bastante en esto.

En el histograma de probabilidades predichas se puede ver que el modelo separa bastante bien las dos clases: la mayoría de los que no sobrevivieron reciben probabilidades bajas y la mayoría de los que sí sobrevivieron reciben probabilidades altas. Hay una zona de solapamiento en torno a 0.4-0.6 donde el modelo tiene más dudas, que corresponde principalmente a pasajeros con características mixtas (hombres de primera clase, mujeres de tercera clase...).

---

## 8. Conclusiones

El modelo alcanza una accuracy de aproximadamente el 82% en test, lo que es un buen resultado teniendo en cuenta que la red neuronal está implementada desde cero sin ninguna librería de deep learning.

Las variables más importantes son el sexo, la clase y la tarifa. Estas tres variables juntas capturan la mayor parte de la información. El resto (título, tamaño de familia, cubierta...) aporta algo pero no de forma tan decisiva.

La parte más interesante fue implementar el backpropagation a mano. Entender por qué se necesita guardar la preactivación `z` para ReLU pero basta con `a` para Sigmoid, o por qué la combinación Sigmoid + cross-entropy simplifica tanto el gradiente de la última capa, son cosas que se aprenden mucho mejor escribiendo el código que leyendo la teoría.

Si se quisiera mejorar el modelo, algunas opciones serían añadir regularización L2 o dropout, probar otras arquitecturas o ajustar el umbral de decisión según si se quiere priorizar precisión o recall.

---

## 9. Estructura del repositorio

```
proyecto_final/
├── 00_preprocesado.ipynb       # Preprocesado y feature engineering
├── 01_eda.ipynb                # Análisis exploratorio
├── 02_model.ipynb              # Implementación y evaluación del modelo
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
└── titanic_datasets/
    ├── unprocessed/
    │   ├── train.csv           # Dataset original de Kaggle
    │   └── gender_submission.csv
    └── processed/
        ├── train_processed.csv # Dataset de entrenamiento procesado
        └── test_processed.csv  # Dataset de test procesado
```

## 10. Cómo reproducir

```bash
pip install -r requirements.txt
```

Ejecutar los notebooks en orden:

1. `00_preprocesado.ipynb`
2. `01_eda.ipynb`
3. `02_model.ipynb`
