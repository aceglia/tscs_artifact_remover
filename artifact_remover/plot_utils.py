import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class PlotSolution:
    def __init__(self):
        pass

    def initialize(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)