"""Методы распределения задач (статья, раздел 3). Жадный, CBBA, «k-means + венгерский
метод», GACA, метод СКФУ."""
from .base import MethodResult, check_assignment, finalize
from .cbba import cbba
from .gaca import gaca, nn_chains
from .greedy import greedy
from .kmeans_hungarian import kmeans, kmeans_hungarian
from .skfu import sector_labels, skfu, strip_labels

METHODS = {
    "greedy": greedy,
    "cbba": cbba,
    "kmeans_hungarian": kmeans_hungarian,
    "gaca": gaca,
    "skfu": skfu,
}

__all__ = ["METHODS", "MethodResult", "check_assignment", "finalize", "greedy", "cbba",
           "kmeans", "kmeans_hungarian", "gaca", "nn_chains", "skfu", "sector_labels",
           "strip_labels"]
