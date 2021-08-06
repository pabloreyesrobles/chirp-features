# Chirp features extractor

Es necesario tener instalada la biblioteca de [scipy] o [anaconda]. Además es necesario instalar la biblioteca **spikelib**, se recomienda instalar el siguiente [fork]. Para esto clonar el repositorio e instalar directo desde la ruta donde ha sido descargado, luego por consola éste repositorio y los siguientes paquetes mediante pip:

```
$ git clone https://github.com/pabloreyesrobles/spikelib.git
$ cd spikelib
$ pip install .
$ pip install tqdm h5py seaborn ipywidgets
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

Una vez los experimentos se encuentren en orden la ruta a ellos debe ser especificada en el archivo **params.json** además del directorio donde se almacenarán los resultados. Luego ejecutar el script según:

```sh
$ python extract_features.py
```

   [scipy]: <https://www.scipy.org/>
   [anaconda]: <https://www.anaconda.com/>
   [fork]: <https://github.com/pabloreyesrobles/spikelib/>