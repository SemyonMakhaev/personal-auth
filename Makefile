deps:
	python3 -m pip install -r requirements.txt

limiter:
	python3 main.py limiter -m assets/limiter.h5 -p assets/limiter.png -s

modifier:
	python3 main.py modifier -m assets/modifier.h5 -p assets/modifier.png -s

recommender:
	python3 main.py recommender -m assets/recommender.h5 -p assets/recommender.png -s

lint:
	pylint main.py
	pylint models/*
	pylint utils/*

help:
	python3 main.py -h
