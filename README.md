# Podcast Brief

Générateur de briefs pour podcast data/IA/MLOps. Scanne l'actu de la semaine et produit des briefs structurés pour chaque segment.

## Stack

- **Backend** : FastAPI + uv
- **Frontend** : HTML/CSS/JS mobile-first (max 480px)
- **IA** : Claude Sonnet (primaire) → Mistral Large (fallback)
- **Recherche** : Hacker News API (gratuit, sans clé — fallback Mistral uniquement)
- **Hébergement** : Render

---

## Setup local

```bash
# 1. Copier et remplir les variables d'environnement
cp .env.example .env
# Éditer .env avec tes clés

# 2. Installer les dépendances
uv sync

# 3. Lancer en mode dev
uv run uvicorn main:app --reload
```

App disponible sur `http://localhost:8000`

### Où obtenir les clés

| Variable | Source |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai) |

---

## Déploiement Render

```bash
# 1. Pusher le repo sur GitHub
git push origin main
```

Ensuite sur [render.com](https://render.com) :

1. **New Web Service** → connecter le repo GitHub
2. Le `render.yaml` configure tout automatiquement
3. **Environment** → ajouter les 2 variables :
   - `ANTHROPIC_API_KEY`
   - `MISTRAL_API_KEY`
4. **Deploy** → l'app est dispo sur `https://podcast-brief.onrender.com`

Chaque `git push` redéploie automatiquement.

---

## Architecture

```
POST /api/scan   → scan actu + sélection des 5 news
POST /api/brief  → génération du brief pour une news

Logique fallback :
  1. Claude Sonnet (web_search natif)
  2. Si 401/429 → Mistral Large + Hacker News API (gratuit, sans clé)
  3. Si tout échoue → message d'erreur clair
```
