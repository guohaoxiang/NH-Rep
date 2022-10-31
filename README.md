# NH-Rep

<p align="center"> 
<img src="/images/nhrep_teaser.png" width="1000">
</p>

This is the official implementation of the following paper:

Guo H X, Liu Y, Pan H, Guo B N. NH-Rep: Neural Halfspace Representations for Implicit Conversion of B-Rep Solids. 

[Paper](https://arxiv.org/abs/2209.10191) | [Project Page](https://guohaoxiang.github.io/projects/nhrep.html)

Abstract: _We present a novel implicit representation  -- neural halfspace representation (NH-Rep), to convert manifold B-Rep solids to implicit representation. NH-Rep is a Boolean tree built on a set of implicit functions represented by neural network, and the composite Boolean function is capable of representing solid geometry while preserving sharp features. We propose an efficient algorithm to extract the Boolean tree from a Manifold B-Rep solid and devise a neural-network-based optimization approach to compute implicit functions.
We demonstrate the high quality offered by our conversion algorithm on ten thousand manifold B-Rep CAD models that contain various curved patches including NURBS, and the superiority of our learning approach over other representative implicit conversion algorithms in terms of surface reconstruction, sharp feature preservation, signed distance field approximation, and robustness to various surface geometry, as well as applications supported by NH-Rep._

The code has been tested on a Ubuntu 18.04 server with CUDA 10.2 installed.

## Installation

Please first clone this repo with its submodules:
        
        $ git clone --recursive https://github.com/guohaoxiang/NH-Rep.git

Then set up the environment via Docker or Anaconda.

**Via Docker**

This is the most convenient way to try _NH\_Rep_, everything is already settled down in the docker.

        $ docker pull horaceguo/pytorchigr:isg
        $ docker run --runtime=nvidia --ipc=host --net=host -v PATH_TO_NH-REP/:/workspace -t -i horaceguo/pytorchigr:isg
        $ cd /workspace
        
Then you can convert the points sampled on B-Rep model in _input\_data_ to implicit representation:

        $ cd code/conversion
        $ python run.py --conf setup.conf --pt ../data/output_data

The training will take about 8 minutes to finish. Currently we only support training with one gpu, you can set gpu id via _--gpu_ flag. The output neural implicit function (broken_bullet_50k_model_h.pt) are stored in folder _data/output\_data_, whose zero surface can be extracted with our iso-surface generator:

        $ cd PATH_TO_NH-REP/data/output_data
        $ /usr/myapp/ISG -i broken_bullet_50k_model_h.pt -o broken_bullet_50k.ply -d 8

You will find the feature-preserving zero-surface mesh (broken_bullet_50k.ply) in _data/output\_data_.

**Via Conda**

You can also setup the environment with conda:

        $ conda env create -f environment.yml
        $ conda activate nhrep

Meanwhile, you need to build iso-surface generator mannually, please refer [here](https://github.com/xueyuhanlang/IsoSurfacing). The built executable file lies in _code/IsoSurfacing/build/App/console_pytorch/ISG_console_pytorch_.

After that, you can conduct implicit conversion and iso-surface extraction as mentioned above.

## Data downloading
We provide the pre-processed ABC dataset used for training NH-Rep, you can download it from [BaiduYun](https://pan.baidu.com/s/1F8kKQM7AcOPBrl1oqLgJQA?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9XYOCbeISf0z5GuB?e=V5ppev), which can be extracted by [7-Zip](https://www.7-zip.org/). Please unzip it under the _data_ folder. For each model, there will be 3 input items:
```
*_50k.xyz: 50,000 sampled points of the input B-Rep, can be visualized with MeshLab.
*_50k_mask.txt: (patch_id + 1) of sampled points.
*_50k_csg.conf: Boolean tree built on the patches, stored in nested lists. 'flag_convex' indicates the convexity of the root node. 
```
For example, _data/input_data/broken_bullet_50k_csg.conf_ looks like:
```
csg{
    list = [0,1,[2,3,4,],],
    flag_convex = 1,
}
```
The operation of root node is op(convex) = max, the root node contains 2 patch leaf node 'p_0' and 'p_1', and a child tree node. The child tree node contains 3 patch leaf node 'p_2', 'p_3' and 'p_4'. So the Boolean tree looks like:

```
     max
   /  |  \
  /   |   \    
p_0  p_1  min 
        /  |  \
       /   |   \ 
      p_2 p_3  p_4
```

If you want to generate our training data from the raw ABC dataset, please refer [here](code/pre_processing/README.md).

**\[Optional\]** You can also download the output of NH-Rep from [BaiduYun](https://pan.baidu.com/s/1ogCm5SPUHzFmDOOKngisLQ?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9W26y62lhSLRwQKL?e=lHuHXe), and unzip it under _data_ folder. For each model, there will be 2 outputs:
```
*_50k_model_h.pt: implicit function of root node stored with TorchScript.
*_50k.ply: extracted zero surface of the implicit function.
```

With the provided output data, you can skip training and directly go to the evaluation part. 

## Training for the whole dataset

To convert the whole dataset to neural halfspace representation by training from scratch, run:

        $ cd PATH_TO_NH-REP/code/conversion
        $ python run.py --conf setup_all.conf --pt ../data/output_data

As there are totally over 10,000 models, the training will take a long long time. We recommend you to use multiple gpus for training. To do this, simply create more *.conf files and distribute 'fileprefix_list' of _setup_all.conf_ into each of them.

## Evaluation

To conduct evaluation, you need to firstly build a point-sampling tool.

        $ cd PATH_TO_NH-REP/code/evaluation/MeshFeatureSample
        $ mkdir build && cd build
        $ cmake ..
        $ make

Then you can evaluate the conversion quality (CD, HD, NAE, FCD, FAE) of the broken_bullet model:

        $ cd PATH_TO_NH-REP/code/evaluation
        $ python evaluation.py 

To evaluate the whole dataset, please download 'eval_data' from [BaiduYun](https://pan.baidu.com/s/1XEy9H_mI43Egl3-wYYutfQ?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9XFMHUGg_kOigBrN?e=dBNs6V), and unzip it under the _data_ folder, then run:

        $ python evaluation.py --name_list all_names.txt

Statistics will be stored in _eval_results.csv_.

The *.ptangle* file used for evaluation stores position and dihedral angle (in degree) of points uniformly sampled on sharp features of a model.

To evaluate the DE and IoU metric, you need to download ground truth mesh data from [BaiduYun](https://pan.baidu.com/s/1uob8xASuUbXzJyuo9EsOZA?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9XTzUmZ4XaDyXKkf?e=TBTgJR), and unzip it under the root folder. You also need to build _code/IsoSurfacing_, then switch to folder _PATH_TO_NH-REP/data/output_data_, run:

        $ python eval_de_iou.py

DE and IoU will be stored in the *_eval.txt files.

## Citation

If you use our code for research, please cite our paper:
```
@article{Guo2022nhrep,
  title={NH-Rep: Neural Halfspace Representations for Implicit Conversion of B-Rep Solids},
  author={Guo, Hao-Xiang and Yang, Liu and Pan, Hao and Guo, Baining},
  journal={ACM Transactions on Graphics (TOG)},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

## License

MIT Licence

## Contact

Please contact us (Haoxiang Guo guohaoxiangxiang@gmail.com, Yang Liu yangliu@microsoft.com) if you have any question about our implementation.

## Acknowledgement
This implementation takes [IGR](https://github.com/amosgropp/IGR) as references. Our codes also include [happly](https://github.com/nmwsharp/happly), [yaml-cpp](https://github.com/jbeder/yaml-cpp), [cxxopts](https://github.com/jarro2783/cxxopts) and [Geometric Tools](https://github.com/davideberly/GeometricTools). We thank the authors for their excellent work.
