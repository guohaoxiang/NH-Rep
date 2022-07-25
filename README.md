# NH-Rep

<p align="center"> 
<img src="/images/nhrep_teaser.png" width="1000">
</p>

This is the official implementation of the following paper:

Guo H X, Liu Y, Pan H, Guo B N. NH-Rep: Neural Halfspace Representations for Implicit Conversion of B-Rep Solids. 

[Paper（to do）]() | [Project Page(to do)]()

Abstract: _We present a novel implicit representation  -- neural halfspace representation (NH-Rep), to convert manifold B-Rep solids to implicit representation. NH-Rep is a Boolean tree built on a set of implicit functions represented by neural network, and the composite Boolean function is capable of representing solid geometry while preserving sharp features. We propose an efficient algorithm to extract the Boolean tree from a Manifold B-Rep solid and devise a neural-network-based optimization approach to compute implicit functions.
We demonstrate the high quality offered by our conversion algorithm on ten thousand manifold B-Rep CAD models that contain various curved patches including NURBS, and the superiority of our learning approach over other representative implicit conversion algorithms in terms of surface reconstruction, sharp feature preservation, signed distance field approximation, and robustness to various surface geometry, as well as applications supported by NH-Rep._

The code has been tested on a Ubuntu 18.04 server with CUDA 10.2 installed.

## Quick try

Firstly setup the environment with conda:

        $ conda env create -f environment.yml
        $ conda activate nhrep
        
Then you can convert the points sampled on B-Rep model in _input\_data_ to implicit representation:

        $ cd conversion
        $ python run.py --conf setup.conf --pt output_data

Input models are listed in _setup.conf_. And the output pt files are in folder _output\_data_, whose zero surfaces can be extracted using ...

## Data downloading (to do)
We provide the pre-processed ABC dataset used for training and evaluating ComplexNet [here](https://pan.baidu.com/s/1PStVn2h_kkKtYsc-LYF7sQ?pwd=asdf), which can be extracted by [7-Zip](https://www.7-zip.org/). You can find the details of pre-processing pipelines in the [supplemental material](https://haopan.github.io/data/ComplexGen_supplemental.zip) of our paper.

The data contains surface points along with normals, and its ground truth B-Rep labels. After extracting the zip file under root directory, the data should be organized as the following structure:
```
ComplexGen
│
└─── data
    │
    └─── default
    │   │
    |   └─── train
    │   │
    |   └─── val
    │   │
    |   └─── test
    |   |   
    |   └─── test_point_clouds
    |        
    └─── partial
        │
        └─── ...
```

<!-- Here _noise_002_ and _noise_005_ means noisy point clouds with normal-distribution-perturbation of mean value _0.02_ and _0.05_ respectively. -->

**\[Optional\]** You can also find the output of each phase [here](https://pan.baidu.com/s/1vO0nTSBbCw52EWUDZI7X4g?pwd=asdf). For each test model, there will be 4 or 5 outputs:
```
*_input.ply: Input point cloud
*_prediction.pkl: Output of 'ComplexNet prediction' phase
*_prediction.complex: Visualizable file for *_prediction.pkl, elements with valid probability larger than 0.3 are kept.
*_extraction.complex: Output of 'complex extraction' phase
*_geom_refine.json: Output of 'geometric refinement' phase, which is also the final output.
```
The description and visualization of each file type can be found in [pickle description](docs/network_prediction_pickle_description.md), [complex description](docs/complex_extraction_complex_description.md) and [json description](docs/geometric_refinement_json_description.md). If you want to directly evaluate the provided output data of ComplexGen, please put the extracted _experiments_ folder under root folder _ComplexGen_, and conduct [Environment setup](https://github.com/guohaoxiang/ComplexGen#environment-setup-with-docker) and [Evaluation](https://github.com/guohaoxiang/ComplexGen#evaluation)

## Evaluation
