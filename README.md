# Atelier de Deep Learning Génératif

## Présentation

 Consiste à découvrir et implémenter plusieurs modèles de Deep Learning génératif avec Python et PyTorch.

L'atelier est organisé en deux parties :

- **Partie 1 :** Autoencoder (AE) et Variational Autoencoder (VAE).
- **Partie 2 :** Generative Adversarial Networks (GAN).

Les expérimentations sont réalisées dans des notebooks Kaggle.

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

## Technologies utilisées

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebooks

## Structure du projet

```text
Atelier_Deep_Learning_Generatif/
│
├── README.md
│
├── 01_ae_vae_mnist.ipynb
│  
│
└── Partie_2_GAN
```

## Partie 2 : GAN

Cette partie sera consacrée à l'implémentation et à l'entraînement d'un Generative Adversarial Network (GAN), en utilisant PyTorch et le dataset Abstract Art Gallery.
