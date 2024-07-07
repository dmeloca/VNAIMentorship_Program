# Ejecución del Notebook

Para ejecutar este notebook en tu computadora, es necesario tener Conda instalado. Puedes seguir estos pasos para crear el entorno virtual necesario:

1. **Crear el entorno virtual**:

    ```bash
    conda env create -f environment.yml
    ```
También puede cargar directamente este notebook en cola.

## Objetivo

El objetivo de esta tarea es modificar los pesos para conseguir la compuerta XOR.

### Input

```python
w = np.array([15, 8.5, 
              -3, 
              5, 6.28, 
              -1,
              -6.5, 0,
              12])

```
### Output esperado:

```python
[1, 0, 0, 1]