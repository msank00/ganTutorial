clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-pyc
	rm -rf asset/test_image/.ipynb_checkpoints
	rm asset/test_image/*.jpg
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	rm -rf src/.ipynb_checkpoints

format:
	isort -rc .
	black -l 79 .