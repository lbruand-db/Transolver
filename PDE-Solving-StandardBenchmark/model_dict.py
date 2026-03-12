import sys
import os

from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D, Transolver_Structured_Mesh_3D

# Add parent directory to path so transolver3 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transolver3.model import Transolver3


def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
        'Transolver_Structured_Mesh_2D': Transolver_Structured_Mesh_2D,
        'Transolver_Structured_Mesh_3D': Transolver_Structured_Mesh_3D,
        'Transolver3': Transolver3,  # Transolver-3 with optimized Physics-Attention
    }
    return model_dict[args.model]
