# villa-X Actor Model - Hugging Face Hub Integration Guide

Ce guide explique comment utiliser le modèle Actor de villa-X avec l'intégration Hugging Face Hub, en particulier pour les VLM open source comme Molmo et InternVL.

## 🎯 Vue d'ensemble

Le modèle Actor est la partie du système villa-X qui prédit les actions robotiques à partir d'observations visuelles. Cette implémentation intègre `PyTorchModelHubMixin` pour une distribution facile sur Hugging Face Hub.

### Caractéristiques principales

- **Intégration VLM flexible** : Support pour Molmo, InternVL et autres VLM
- **Actions latentes** : Peut utiliser les actions latentes du LAM (Latent Action Model)
- **Horizon d'action** : Prédit plusieurs actions futures pour un contrôle robotique fluide
- **Hugging Face Ready** : Méthodes `from_pretrained()` et `push_to_hub()` intégrées

## 🚀 Installation rapide

```bash
# Cloner le repository
git clone https://github.com/microsoft/villa-x.git
cd villa-x

# Installer les dépendances
uv sync

# Installer les dépendances Hugging Face si pas déjà présentes
pip install huggingface-hub safetensors
```

## 📖 Guide d'utilisation

### 1. Configuration de base

```python
from lam.actor_model import ActorModel, ActorModelConfig

# Configuration pour Molmo
config = ActorModelConfig(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    use_latent_actions=True,
    action_dim=7,  # 6DoF + gripper
    action_horizon=10,
)

# Créer le modèle
actor_model = ActorModel(config)
```

### 2. Entraînement avec différents VLM

```python
from lam.actor_model import setup_actor_model_for_training

# Pour Molmo
molmo_model = setup_actor_model_for_training(
    vlm_model_name="microsoft/molmo-7B-D-0924",
    action_dim=7,
    action_horizon=10,
)

# Pour InternVL
internvl_model = setup_actor_model_for_training(
    vlm_model_name="OpenGVLab/InternVL2-4B",
    action_dim=7,
    action_horizon=10,
)
```

### 3. Intégration avec le modèle LAM

```python
# Charger le modèle Actor avec LAM pré-entraîné
actor_model = ActorModel.from_pretrained_lam(
    lam_model_path="path/to/lam/checkpoint",
    actor_config=config
)

# Extraire les actions latentes depuis une vidéo
video_clips = torch.randn(1, 8, 3, 224, 224)  # [batch, frames, channels, height, width]
latent_actions = actor_model.extract_latent_actions(video_clips)
```

### 4. Prédiction d'actions

```python
# Caractéristiques visuelles du VLM (à extraire avec votre VLM)
visual_features = torch.randn(1, 10, 4096)  # [batch, sequence, features]

# Prédire les actions
actions = actor_model.predict_actions(
    visual_features=visual_features,
    latent_actions=latent_actions,  # Optionnel
)

print(f"Actions prédites: {actions.shape}")  # [1, 10, 7]
```

## 🤗 Publication sur Hugging Face Hub

### Script d'entraînement et publication

```bash
# Entraîner et publier le modèle
python scripts/train_actor_model.py \
    --vlm_model microsoft/molmo-7B-D-0924 \
    --epochs 100 \
    --upload_to_hub \
    --repo_id microsoft/villa-x-actor-molmo
```

### Publication manuelle

```python
from lam.actor_model import prepare_for_hub_upload

# Après l'entraînement
prepare_for_hub_upload(
    model=actor_model,
    repo_id="microsoft/villa-x-actor-molmo",
    commit_message="Upload Actor Model trained on Molmo VLM"
)
```

### Chargement depuis le Hub

```python
# Charger un modèle publié
actor_model = ActorModel.from_pretrained("microsoft/villa-x-actor-molmo")
```

## 🔧 Configuration détaillée

### ActorModelConfig

```python
config = ActorModelConfig(
    # Configuration VLM
    vlm_model_name="microsoft/molmo-7B-D-0924",
    vlm_hidden_size=4096,
    
    # Configuration des actions
    action_dim=7,
    action_horizon=10,
    action_chunk_size=100,
    
    # Architecture
    hidden_size=512,
    num_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    dropout_prob=0.1,
    
    # Actions latentes
    use_latent_actions=True,
    latent_action_dim=512,
    
    # Entraînement
    loss_type="mse",  # "mse", "l1", "smooth_l1"
    action_loss_weight=1.0,
)
```

## 📝 Exemples d'utilisation

### Exemple avec Molmo

```python
from examples.actor_model_demo import ActorModelDemo

# Démonstration avec Molmo
demo = ActorModelDemo("microsoft/molmo-7B-D-0924")
demo.demonstrate_inference()
```

### Exemple avec InternVL

```python
# Démonstration avec InternVL
demo = ActorModelDemo("OpenGVLab/InternVL2-4B")
demo.demonstrate_inference()
```

## 🔄 Flux de travail recommandé

### Pour l'entraînement

1. **Préparer les données** : Format avec images/vidéos et actions cibles
2. **Choisir le VLM** : Molmo ou InternVL selon vos besoins
3. **Configurer le modèle** : Adapter les dimensions et paramètres
4. **Entraîner** : Utiliser le script fourni ou votre propre boucle
5. **Publier** : Upload sur Hugging Face Hub

### Pour l'inférence

1. **Charger le modèle** : Depuis Hub ou checkpoint local
2. **Extraire les features** : Utiliser votre VLM pour les features visuelles
3. **Prédire** : Obtenir les actions robotiques
4. **Exécuter** : Envoyer les actions au robot

## 🚧 Notes importantes

### Contraintes de licence

Comme mentionné dans l'issue GitHub #12, le modèle pré-entraîné basé sur PaliGemma ne peut pas être publié pour des raisons de licence. Cette implémentation vise les VLM avec des licences plus permissives :

- **Molmo** : Licence Apache 2.0
- **InternVL** : Licence MIT

### Dépendances

Assurez-vous d'avoir les bonnes versions :

```bash
pip install torch>=2.0 transformers>=4.50.0 huggingface-hub safetensors
```

### Performance

- Le modèle Actor est optimisé pour l'inférence temps réel
- L'utilisation de GPU est recommandée
- Les actions latentes du LAM améliorent significativement les performances

## 📚 Ressources supplémentaires

- [Paper villa-X](https://arxiv.org/abs/2507.23682)
- [Page du projet](https://aka.ms/villa-x)
- [Modèles Hugging Face](https://huggingface.co/microsoft/villa-x)
- [Guide Hugging Face Hub](https://huggingface.co/docs/hub/models-uploading)

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment aider :

1. **Nouveaux VLM** : Ajouter le support pour d'autres modèles
2. **Améliorations** : Optimisations de performance ou nouvelles fonctionnalités
3. **Documentation** : Améliorer les guides et exemples
4. **Tests** : Ajouter des tests pour différentes configurations

## 📄 Citation

Si vous utilisez ce travail, merci de citer :

```bibtex
@article{chen2025villa0x0,
  title   = {villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models},
  author  = {Xiaoyu Chen and Hangxing Wei and Pushi Zhang and Chuheng Zhang and Kaixin Wang and Yanjiang Guo and Rushuai Yang and Yucen Wang and Xinquan Xiao and Li Zhao and Jianyu Chen and Jiang Bian},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2507.23682}
}
```

---

**Note** : Cette implémentation est prête pour l'entraînement avec Molmo et InternVL. Une fois l'entraînement terminé, les modèles pourront être partagés sur Hugging Face Hub conformément à la réponse de @kaixin96 dans l'issue #12.
