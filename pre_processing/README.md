## Data pre-processing

Please first download the raw data of ABC dataset [here](https://pan.baidu.com/s/1N8E_8xhwKmE2UHuJhdFPZA?pwd=asdf), and unzip it under current folder. We split the models with multiple components. There will be 3 items for each model:

```
*.obj: surface mesh of the model.
*.yml: parametric curve and patch information.
*.fea: curve segments on the mesh. Starts with the number of curve segments n, followed by n lines, where each line contains the two vertex indices of a curve segment. Can be extracted from 'vert_indices' of curves in the *.yml files.
```

**Installation**

Please first install the Boost and eigen3 library, then run

        $ cd PATH_TO_NH-REP/pre_processing
        $ mkdir build && cd build
        $ cmake ..
        $ make

Then you can generate the training data:

        $ python gen_training_data_yaml.py

If you do not have a yaml file and want to generate sample points from meshes, you can prepare the *.fea file as sharp feature curves of the meshes, then run:

        $ python gen_training_data_mesh.py

Please make sure that you set 'data_path' in _gen_training_data_yaml.py_ and _gen_training_data_mesh.py_ correctly.


When patch decomposition is conducted, there will be *_fixtree.obj and *_fixtree.fea in 'data_path', which can be used for generating point samples in later round.