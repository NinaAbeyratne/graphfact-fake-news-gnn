# Project Setup Log

## Project Name

GraphFact: Fake News Detection using NLP and Graph Neural Networks

## Repository Setup

A GitHub repository was created for the project:

```text
graphfact-fake-news-gnn
```

The repository was initialized with:

README file
Python .gitignore
MIT License

## Local Repository Setup

The repository was cloned locally using:

git clone https://github.com/NinaAbeyratne/graphfact-fake-news-gnn.git
cd graphfact-fake-news-gnn

## Folder Structure Setup

The following folder structure was created:

configs/
data/
  raw/
  processed/
  graph/
notebooks/
src/
  preprocessing/
  embeddings/
  graph/
  models/
  training/
  inference/
api/
frontend/
artifacts/
tests/
reports/

## Purpose of Main Folders

Folder	Purpose
configs/	Store project configuration files
data/raw/	Store original raw datasets
data/processed/	Store cleaned/preprocessed datasets
data/graph/	Store graph objects and edge lists
notebooks/	Store Google Colab notebooks
src/preprocessing/	Text cleaning and dataset preparation scripts
src/embeddings/	Transformer embedding generation scripts
src/graph/	Graph construction scripts
src/models/	GNN and baseline model definitions
src/training/	Training and evaluation scripts
src/inference/	Prediction/inference utilities
api/	FastAPI backend
frontend/	Streamlit/Gradio frontend
artifacts/	Saved models, embeddings, metrics, and outputs
tests/	Unit and integration tests
reports/	Documentation, analysis, and experiment reports

## Files Created
configs/config.yaml
requirements.txt
api/main.py
frontend/app.py
reports/system_design.md

Placeholder .gitkeep files were added to keep empty data and artifact folders visible in GitHub.

## Mac System File Handling

Mac-generated .DS_Store files were ignored by adding the following line to .gitignore:

.DS_Store

## Git Commands Used
git add .
git commit -m "chore: setup project folder structure"
git push origin main

## Issues Faced

During the first setup attempt, the local branch and GitHub branch diverged because some files were already committed on GitHub.

The issue was resolved by starting with a fresh clone of the GitHub repository and then recreating the project folder structure cleanly.

## Current Status

The project repository is now initialized with a clean folder structure and is ready for:

dataset acquisition,
data exploration,
preprocessing,
NLP embedding generation,
graph construction,
GNN training,
API and frontend development.