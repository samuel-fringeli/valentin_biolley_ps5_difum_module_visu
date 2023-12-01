# PS5_DIFUM_VISU

> Original author of the project: Biolley Valentin

Ce guide a pour but de démontrer l'utilisation du module de visualisation PS5_DIFUM_VISU qui se trouve sous la forme d'un notebook.

Veuillez suivre le readme.md d'installation du projet.

### Structure:

Ce notebook utilise les encodeurs Bert pour le text et Swin pour l'image. 

Différents labels utilisés pour les tests se trouvent dans le dossier "file"
ainsi que des images de tests se trouvent dans le dossier "images_fire_not_fire"

Ensuite ce notebook met à disposition pleins d'outils permettant de manipuler les vecteurs d'embeddings. La documentation des entrées sorties de chaqu'une de ces fonctions se trouvent dans le notebook.

Voici quelques méthodes utiles pour la générer les données nécessaires à la visualisation :

- get_txt_embedding
- read_file_label
- read_dir_image
- get_img_embedding_swin

### Utilisation:

Pour pouvoir visualiser des vecteurs d'embeddings, vous n'avez besoin que des fonctions suivantes:

- **create_data_set_for_vis** qui permet de créer un dataset avec toutes les données nécessaires pour la génération des graphiques. 
- **visualise_embedding** qui permet de générer les graphiques.

2 exemples de visualisation se trouvent à la fin du notebook.