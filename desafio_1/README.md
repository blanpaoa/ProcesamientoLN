# Desafío 1 — Vectorización de texto y Naïve Bayes

## Descripción

Trabajo práctico sobre vectorización de texto y clasificación con modelos Naïve Bayes utilizando el dataset **20 Newsgroups**.

## Contenido

El notebook `Desafio_1_LenguajeNatural.ipynb` cubre:

1. **Carga de datos** — Dataset 20 Newsgroups (~18 000 publicaciones en 20 categorías temáticas), separado en train/test.

2. **Vectorización** — Uso de `TfidfVectorizer` y `CountVectorizer` de scikit-learn para transformar texto en matrices documento-término dispersas (sparse).

3. **Similitud de documentos** — Cálculo de similitud coseno entre documentos usando sus representaciones TF-IDF.

4. **Clasificación Naïve Bayes** — Comparación sistemática de configuraciones (`MultinomialNB` vs `ComplementNB`, distintos parámetros de vectorización) evaluadas con F1-Score Macro.
   - **Mejor configuración encontrada:** TF-IDF con `min_df=2` + `ComplementNB` con `alpha=0.5` → **F1-Macro: 0.6980**

5. **Similitud entre palabras** — Transposición de la matriz documento-término para obtener vectores de contexto de palabras y medir su similitud semántica mediante similitud coseno.

## Dependencias

```
numpy
scikit-learn
```

Instalación:

```bash
pip install numpy scikit-learn
```

## Ejecución

```bash
jupyter notebook Desafio_1_LenguajeNatural.ipynb
```
