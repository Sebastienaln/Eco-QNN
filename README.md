# Projet 4


## Analyse

L'article qu'on a lu : reuploading => + profondeur, - de qubits, donc + d'expressivité, + de bruit de décohérence, - de bruit de diaphonie.
Sans reuploading => beaucoup de qubits, + de bruit de diaphonie.
Soit on veut minimiser le nombre de qubits, soit le nombre de portes, soit le temps (i.e. diminuer le bruit, donc le nombre de shots).


Une limitation des algorithmes réside dans le bruit que le circuit contient. Celui-ci détériore les performances de l'algorithme et c'est quelque chose que nous devons prendre en compte lors de nos expérimentations

## Idée
Notre idée pour le projet était alors de nous donner un "challenge". En partant d'un classifieur quantique à n qubits avec une certaine précision, réussir à avoir un circuit avec moins de qubits et des performances similaires. C'est faisable en faisant beaucoup de re-uploading d'après l'article que nous avons lu initialement, cependant nous voulons limiter le nombre de couches ajoutées pour réduire le bruit qui pourrait être crée. 

On sait que l'encodage en angle et en amplitude consiste en deux paradigmes différents : réduire le nombre de qubits mais augmenter la profondeur, ou augmenter le nombre de qubits pour réduire la profondeur. Ici, on peut trouver un compromis avec le data reuploading, et c'est une piste que nous gardons en tête. 
Pour un problème de classification du style n-sphère, on pourrait évaluer la précision de différents modèles en profondeur avec du data reuploading (jusqu'à 1 qubit), ou en largeur avec n qubits et étudier les résultats de structures différentes.


### Tâches
* Faire un système qui, pour n features et m qubits, génère le circuit Qiskit.
* Faire un algo d'entraînement et d'inférence.
* Génération des datasets.
* Faire des courbes précision vs largeur, précision vs profondeur. Pour des problèmes de taille n.

### Travail préparatoire 
* Trouver un algorithme de QML utilisant n qubits qui nous servira de base
* Trouver un article qui réussit à réduire le nombre de bits dans un circuit et étudier les stratégies employées
* Avoir un meilleur visuel sur comment optimiser le nombre de shots et voir si cela peut s'insérer dans notre approche