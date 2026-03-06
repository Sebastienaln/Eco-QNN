# Projet 4


## Analyse

L'article qu'on a lu : reuploading => + profondeur, - de qubits, donc + d'expressivité, + de bruit de décohérence, - de bruit de diaphonie.
Sans reuploading => beaucoup de qubits, + de bruit de diaphonie.
Soit on veut minimiser le nombre de qubits, soit le nombre de portes, soit le temps (i.e. diminuer le bruit, donc le nombre de shots).

Pour les barren plateaux : dans un circuit quantique idéal, ce phénomène est souvent dû à une trop grande intrication ou à une trop grande profondeur.
Cependant, cet article démontre que le bruit seul suffit à créer ces plateaux, indépendamment de la structure du circuit.

## Idée
On sait que l'encodage en angle et en amplitude consiste en deux paradigmes différents : réduire le nombre de qubits mais augmenter la profondeur, ou augmenter le nombre de qubits pour réduire la profondeur. Ici, on peut trouver un compromis avec le data reuploading.
Pour un problème de classification du style n-sphère, on pourrait évaluer la précision de différents modèles en profondeur avec du data reuploading (jusqu'à 1 qubit), ou en largeur avec n qubits !


### Tâches
* Faire un système qui, pour n features et m qubits, génère le circuit Qiskit.
* Faire un algo d'entraînement et d'inférence.
* Génération des datasets.
* Faire des courbes précision vs largeur, précision vs profondeur. Pour des problèmes de taille n.