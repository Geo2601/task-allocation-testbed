"""allocsim — стенд для сравнения способов распределения задач в группе аппаратов
с точным эталоном: генератор полей задач, показатель скученности, точное решение, методы, анализ."""
from .fields import FieldSpec, LEVELS, sample_field, reflect_into_box
from .clustering import clustering_index, mean_nn_distance, uniform_reference
from .instance import InstanceSpec, Instance, make_instance, main_field

__all__ = [
    "FieldSpec", "LEVELS", "sample_field", "reflect_into_box",
    "clustering_index", "mean_nn_distance", "uniform_reference",
    "InstanceSpec", "Instance", "make_instance", "main_field",
]
