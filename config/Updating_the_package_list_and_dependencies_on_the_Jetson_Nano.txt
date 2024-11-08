#!/bin/bash

# Mettre à jour les paquets système
sudo apt-get update
sudo apt-get upgrade -y

# Installer Python3 et pip3
sudo apt-get install python3 python3-pip -y

# Installer les dépendances pour cuML (en fonction de la version de CUDA)
pip3 install cuml-cuda11

# Vérifier que cuML est installé
python3 -c "import cuml; print(cuml.__version__)"
