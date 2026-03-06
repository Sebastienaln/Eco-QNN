# Projet 4


## Analyse

L'article qu'on a lu : Forte expressivité possible avc très peu de qubit (même un seul qubit)
Mais moins de qubit dans le circuit entraine plus de profondeur donc plus de bruit et de decoherence. Soulèeve la problématique de trouver un compromis entre largeur et profondeur. Projet: Comment réaliser des QNN "eco" en minimisant le nombre de qubit, laprofondeur et le bruit, ainsi que le nombre de shots (pas exploré dans notre premier article mais interessant aussi) 

Pour les barren plateaux : dans un circuit quantique idéal, ce phénomène est souvent dû à une trop grande intrication ou à une trop grande profondeur.
Cependant, cet article démontre que le bruit seul suffit à créer ces plateaux, indépendamment de la structure du circuit.

## Idée
On sait que l'encodage en angle et en amplitude consiste en deux paradigmes différents : réduire le nombre de qubits mais augmenter la profondeur, ou augmenter le nombre de qubits pour réduire la profondeur. Ici, on peut trouver un compromis avec le data reuploading.
Pour un problème de classification du style n-sphère, on pourrait évaluer la précision de différents modèles en profondeur avec du data reuploading (jusqu'à 1 qubit), ou en largeur avec n qubits !


### Tâches
* Ecrire un circuit quantique de clustring avec n quibits
* A partir de ce circuit, etablir des circuits quantique avec différents nombres de qubit, profondeurs, nombre de shots.
* Etudier influence de profondeur et largeur sur la précision: Faire des courbes précision vs largeur, précision vs profondeur. Pour des problèmes de taille n.
* Vérifier si le bruit reste acceptable



### Travail préparatoire 
* Trouver un algorithme de QML utilisant n qubits qui nous servira de base
* Trouver un article qui réussit à réduire le nombre de bits dans un circuit et étudier les stratégies employées, avec la perspective de les appliquer à notre circuit
* Avoir une meilleur idée sur comment optimiser le nombre de shots et voir si cela peut s'insérer dans notre approche (ce n'était pas dans les articles qu'on a lu jusqu'à maitenant)