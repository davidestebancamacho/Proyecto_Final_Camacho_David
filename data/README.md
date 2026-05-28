# Datos

El dataset **TMDB Movies Metadata** no se incluye en el repositorio
por su tamaño (~228 MB). Descárgalo así:

## Opción A — Kaggle CLI (recomendada)

```bash
pip install kaggle
kaggle datasets download -d rounakbanik/the-movies-dataset
unzip the-movies-dataset.zip
mv movies_metadata.csv data/
```

> Necesitas una cuenta en kaggle.com y tu `~/.kaggle/kaggle.json` configurado.

## Opción B — Descarga manual

1. Ve a: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
2. Descarga `movies_metadata.csv`
3. Muévelo a esta carpeta: `data/movies_metadata.csv`

## Archivo esperado

```
data/
└── movies_metadata.csv   ← 45,466 filas × 24 columnas
```

El código busca el CSV en el directorio de trabajo o en `data/`.
Ajusta la ruta en la celda de carga si es necesario.
