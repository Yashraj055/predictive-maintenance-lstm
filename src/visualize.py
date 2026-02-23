import matplotlib.pyplot as plt
import numpy as np

def plot_health_curve(errors, engine_id=None):
    plt.figure(figsize=(10, 4))
    plt.plot(errors, label="Reconstruction Error")
    plt.xlabel("Time Window")
    plt.ylabel("Error")
    plt.title("Health Degradation Curve")
    plt.legend()
    plt.show()