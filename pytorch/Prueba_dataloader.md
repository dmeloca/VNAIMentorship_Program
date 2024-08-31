## Prueba Técnica: Uso de DataLoaders en PyTorch

### Instrucciones

1. Responde cada pregunta con el código correspondiente.
2. Asegúrate de que el código esté correctamente comentado.
3. Al finalizar, ejecuta el código para verificar que funcione como se espera.

### Parte 1: Creación de un Dataset Personalizado

1. Define una clase `CustomDataset` que herede de `torch.utils.data.Dataset`.
2. Implementa los métodos `__init__`, `__len__` y `__getitem__` para esta clase.
3. El dataset debe contener datos ficticios de imágenes y etiquetas, donde:
    - Las imágenes son tensores de tamaño `(3, 64, 64)` generados aleatoriamente.
    - Las etiquetas son valores enteros entre 0 y 9.

### Parte 2: Uso de DataLoader

1. Crea una instancia del `CustomDataset`.
2. Usa `torch.utils.data.DataLoader` para crear un dataloader con las siguientes características:
    - `batch_size` de 8.
    - `shuffle` activado.
3. Itera sobre el dataloader y muestra el tamaño de las imágenes y etiquetas en cada batch.

### Parte 3: Transformaciones (Bonus)

1. Aplica una transformación a las imágenes del dataset para normalizarlas.
2. Muestra el efecto de la transformación en un batch de imágenes.

### Preguntas de Reflexión

1. ¿Por qué es importante utilizar `DataLoader` en PyTorch?
2. ¿Qué ventajas ofrece el uso de transformaciones en los datos?
3. ¿Cómo afecta el `shuffle` en el `DataLoader` al entrenamiento de un modelo?