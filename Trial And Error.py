import cProfile
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from SpectralLayer import*
from utilsSimpleConv2D import*
from spectralconvolutions import *
from modelSpectralConv2D import *

from tensorflow.keras.layers import Layer, Dense
from typing import Tuple,List,Any,Dict




from sys import argv

# Sur le cluster on a 10 jobs; de 1 à 10.
num = int(argv[1])
model_pectral = ModelSpectral()
if 1 <= num <= 10:
    iteration = 0
    while iteration < 10:
        model_pectral.build_model()
        model_pectral.fit_model()
        with open(f"accuracy{num}.txt", "a") as f:
            f.write(str(model_pectral.accuracy[iteration][1]))
            f.write("\n")
        iteration += 1

    print("programme termine avec succes!")
    # Sur le cluster entrez comme argument num>0.
elif num == 0:
    iteration = 0
    print("Start...\n")
    while iteration < 1:
        model_pectral.build_model()
        model_pectral.fit_model()
        with open(f"accuracy{num}.txt", "a") as f:
            f.write(str(model_pectral.accuracy[iteration][1]))
            f.write("\n")
        iteration += 1
    print("programme termine avec succes!")
    # En local entrez comme argument 0.
else:
    raise NotImplemented