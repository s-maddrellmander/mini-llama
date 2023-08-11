export PYTHONPATH=:~/Projects/mini-llama
isort --profile black *.py tests/*
black .
pytest --cov=. --cov-report term --cov-report html