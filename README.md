# Atelier de Deep Learning Génératif

## Présentation

Cet atelier consiste à découvrir et implémenter plusieurs modèles de Deep Learning génératif avec Python et PyTorch.

L'atelier est organisé en deux parties :

- **Partie 1 :** Autoencoder (AE) et Variational Autoencoder (VAE).
- **Partie 2 :** Generative Adversarial Networks (GAN).

Les expérimentations sont réalisées dans des notebooks Kaggle.

---

## Partie 1 : Autoencoder et Variational Autoencoder

### Dataset

Nous utilisons le jeu de données MNIST, constitué d'images de chiffres manuscrits en niveaux de gris.

- Dimensions des images : 28 × 28 pixels.
- Nombre de classes : 10 (de 0 à 9).
- Images d'entraînement : 60 000.
- Images de test : 10 000.

### 1. Autoencoder (AE)

L'objectif est de compresser les images dans un espace latent de dimension 32, puis de les reconstruire à partir de cette représentation.

Les principales étapes réalisées :

- Chargement et exploration du dataset MNIST.
- Construction de l'architecture de l'Autoencoder.
- Entraînement du modèle.
- Visualisation des pertes.
- Reconstruction des images.
- Extraction des représentations latentes.
- Réduction de dimension avec PCA.

### 2. Variational Autoencoder (VAE)

Le VAE utilise une représentation latente probabiliste. Son encodeur estime une moyenne (mu) et une log-variance (logvar), puis effectue un échantillonnage latent.

Les principales étapes réalisées :

- Construction de l'architecture VAE.
- Implémentation du reparameterization trick.
- Définition de la fonction de perte : Reconstruction Loss + KL Divergence.
- Entraînement pendant 60 epochs avec Adam.
- Évaluation sur les données de test.
- Comparaison des images originales et reconstruites.
- Génération de nouvelles images à partir de vecteurs latents aléatoires.
- Visualisation de l'espace latent avec PCA.

### Comparaison AE / VAE

| Critère | AE | VAE |
|---|---|---|
| Représentation latente | Déterministe | Probabiliste |
| Dimension latente | 32 | 32 |
| Reconstruction | Oui | Oui |
| Génération aléatoire | Pas directement adaptée | Oui |
| Fonction de perte | Reconstruction | Reconstruction + KL |

### Résultats PCA

| Modèle | Variance expliquée par 2 composantes |
|---|---:|
| AE | 28,06 % |
| VAE | 20,34 % |

Ces résultats permettent de comparer la répartition de la variance dans les représentations latentes des deux modèles.

---

## Partie 2 : Generative Adversarial Network (GAN)

### Dataset : Abstract Art Gallery

Pour cette partie, nous utilisons le dataset **Abstract Art Gallery**, disponible sur Kaggle. Il contient 2 872 peintures abstraites.

Caractéristiques du dataset :

- Nombre total d'images : 2 872.
- Format : JPG.
- Mode couleur : RGB (3 canaux).
- Dimensions originales : variables.
- Dimensions après prétraitement : 64 × 64 pixels.

### 1. Exploration et prétraitement des données

Les principales étapes réalisées :

- Exploration de la structure du dataset.
- Vérification des dimensions et du format des images.
- Visualisation de plusieurs peintures abstraites.
- Redimensionnement des images en 64 × 64 pixels.
- Conversion des images en tenseurs PyTorch.
- Normalisation des pixels entre -1 et 1 afin de correspondre à la fonction d'activation Tanh du Generator.

### 2. Construction de l'architecture GAN

Le GAN est composé de deux réseaux neuronaux qui s'entraînent en compétition.

**Generator :**

- Reçoit un vecteur de bruit aléatoire de dimension 100.
- Utilise des couches ConvTranspose2d pour augmenter progressivement la résolution.
- Génère une image RGB de dimensions 64 × 64 pixels.
- Utilise la fonction d'activation Tanh en sortie.

**Discriminator :**

- Reçoit une image RGB de dimensions 64 × 64 pixels.
- Utilise des couches convolutives pour extraire les caractéristiques de l'image.
- Produit un score permettant de distinguer les images réelles des images générées.

### 3. Configuration de l'entraînement

| Paramètre | Valeur |
|---|---|
| Dimension latente | 100 |
| Taille des images | 64 × 64 |
| Taille du batch | 64 |
| Nombre d'epochs | 100 |
| Optimiseur | Adam |
| Learning rate | 0.0002 |
| Betas | (0.5, 0.999) |
| Fonction de perte | BCEWithLogitsLoss |
| Accélération | GPU (CUDA) |

### 4. Entraînement du GAN

L'entraînement repose sur deux opérations répétées pour chaque batch :

- **Entraînement du Discriminator :** apprendre à distinguer les images réelles des images générées.
- **Entraînement du Generator :** produire des images capables de tromper le Discriminator.

Les pertes des deux réseaux sont enregistrées afin de suivre leur évolution pendant les 100 epochs.

### 5. Génération et évaluation des résultats

Après l'entraînement, plusieurs images abstraites sont générées à partir de nouveaux vecteurs de bruit aléatoires.

Les résultats sont évalués à travers :

- La visualisation des images générées.
- L'évolution des pertes du Generator et du Discriminator.
- La comparaison entre les peintures originales et les peintures générées.
- L'analyse des couleurs, des formes, des textures et de la qualité visuelle.

### 6. Résultats et observations

Après 100 epochs d'entraînement, le Generator parvient à apprendre certaines caractéristiques des peintures abstraites, notamment les couleurs, les formes et les compositions.

Les images générées présentent des résultats globalement satisfaisants, même si elles restent légèrement floues et manquent encore de précision dans certains détails par rapport aux images originales.

L'évolution des pertes montre également une diminution importante de la perte du Discriminator, tandis que celle du Generator augmente légèrement entre les 50 premières et les 50 dernières epochs.

Ces observations mettent en évidence les difficultés liées à l'équilibre entre les deux réseaux pendant l'entraînement d'un GAN.

### 7. Perspectives d'amélioration

Plusieurs améliorations pourraient être envisagées :

- Augmenter le nombre d'epochs.
- Ajuster les hyperparamètres d'entraînement.
- Améliorer l'architecture du Generator et du Discriminator.
- Utiliser des techniques de régularisation pour stabiliser l'entraînement.
- Augmenter la résolution des images générées.

---

## Technologies utilisées

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Kaggle Notebooks
- CUDA

---

## Structure du projet

```text
Atelier_Deep_Learning_Generatif/
│
├── README.md
│
├── 01_ae_vae_mnist.ipynb
│
└── 02_gan_abstract_art.ipynb
```

---

## Conclusion générale

Cet atelier nous a permis de découvrir et de mettre en pratique trois approches importantes du Deep Learning génératif : l'Autoencoder, le Variational Autoencoder et le Generative Adversarial Network.

À travers ces expérimentations, nous avons étudié différentes méthodes d'apprentissage des représentations, de reconstruction et de génération de nouvelles images.

La comparaison de ces modèles nous a permis de mieux comprendre leurs mécanismes, leurs objectifs et leurs limites, ainsi que les défis liés à la génération d'images avec les réseaux de neurones.
