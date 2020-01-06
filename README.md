# Image Classification Using Feature Ensembles

Image classification on the [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

___

**Running code**
To reproducibly run the code provided in this repository please use the associated Docker image: fgiuste/ml:graphviz. Software and hardware requirements include Ubuntu OS (tested on 18.04 LTS) and a Tensorflow/Keras compatible GPU. Requirements also include nvidia-graphics installed and latest version of Docker.

To run: ```$ docker run -p <port>:8888 --gpus '"device=0"' -v <repo>:/opt/ fgiuste/ml:graphviz```

*```<port>``` port you want to access the jupyter notebook service*

*```<repo>``` absolute path of clone repo*
    
One you run docker navigate to ```localhost:<port>/tree/opt``` in browser and copy paste the token that appears on terminal.

main.ipynb runs all the code available in repo.


