#!/usr/bin/env bash
set -euo pipefail

BASE="/home/marmatt/Documents/projects/BRAF/myWork/workflowJanuary2026"

jupyter nbconvert --to notebook --execute "$BASE/Workflow1.ipynb"   --output "Workflow1.ran.ipynb"   --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute "$BASE/Workflow2.ipynb" --output "Workflow2.ran.ipynb" --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute "$BASE/Workflow3.ipynb" --output "Workflow3.ran.ipynb" --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute "$BASE/Workflow3.ipynb" --output "Workflow4.ran.ipynb" --ExecutePreprocessor.timeout=-1