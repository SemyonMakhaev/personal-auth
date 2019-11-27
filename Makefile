deps:
	python3 -m pip install -r requirements.txt
	chmod +x main.py
	chmod +x evaluator.py

limiter:
	python3 main.py limiter -m assets/limiter.h5 -p assets/limiter.png -s

modifier:
	python3 main.py modifier -m assets/modifier.h5 -p assets/modifier.png -s

recommender:
	python3 main.py recommender -m assets/recommender.h5 -p assets/recommender.png -s

lint:
	pylint evaluator.py
	pylint main.py
	pylint models/*
	pylint test_suite/*
	pylint utils/*

help:
	python3 main.py -h
