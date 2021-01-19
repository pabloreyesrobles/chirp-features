# Chirp features extractor

Es necesario tener instalada la biblioteca de [scipy] o [anaconda]. Además es necesario instalar la librería [spikelib] directo de su repositorio o a través de pip:
```sh
$ pip install spikelib
```
Es necesario que las carpetas de los experimentos que se desean procesar posean la siguiente estructura
```sh
MR-0XXX
├── config.ini
├── data
│   ├── logs
│   ├── processed_protocols
│   ├── raw
│   ├── sorting
│   │   ├── MR-0XXX.result.hdf5
│   ├── stim
│   └── sync
│       ├── MR-0XXX
│       │   ├── event_list_MR-0XXX.csv
│       │   ├── repeated_frames_MR-0XXX.txt
│       │   ├── start_end_frames_MR-0XXX.txt
│       │   └── total_duration_MR-0XXX.txt
```
**config.ini** debe especificar claramente donde se encuentran los archivos de los directorios *sorting* y *sync* de otra forma el script fallará.
Una vez los experimentos se encuentren en orden deben ser ubicados en el mismo directorio que el respositorio (o mover extract_features.py y chirp.py) y ejecutar
```sh
$ python extract_features.py
```
Pronto será incorporado un notebook para previsualización de la respuesta de chirp y features

   [scipy]: <https://www.scipy.org/>
   [anaconda]: <https://www.anaconda.com/>
   [spikelib]: <https://github.com/creyesp/spikelib/>