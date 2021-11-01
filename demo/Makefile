##############################################################################################
# PROJECT-SPECIFIC PARAMETERS                                                                #
##############################################################################################


PROJECT_NAME = treelstm-classifier
PYTHON ?= python3.7
SOURCE_FOLDER = tlc


##############################################################################################
# BASE COMMANDS                                                                       #
##############################################################################################


.PHONY: env-create
env-create:
	$(PYTHON) -m venv .venv --prompt $(PROJECT_NAME)
	make env-update
	#
	# Don't forget to activate the environment before proceeding! You can run:
	# source .venv/bin/activate


.PHONY: env-update
env-update:
	bash -c "\
		. .venv/bin/activate; \
		which python; \
		which pip; \
		$(PYTHON) -m pip install --upgrade -r requirements.txt; \
		$(PYTHON) -m pip freeze; \
	"


.PHONY: env-delete
env-delete:
	rm -rf .venv


.PHONY: update
update:
	$(PYTHON) -m pip install --upgrade -r requirements.txt


.PHONY: check
check: reformat radon lint


.PHONY: clean
clean:
	rm -f .gitinfo
	rm -rf build dist *.egg-info
	find $(SOURCE_FOLDER) -name __pycache__ | xargs rm -rf
	find $(SOURCE_FOLDER) -name '*.pyc' -delete
	rm -rf reports .coverage
	rm -rf docs/build docs/source
	rm -rf .*cache


.PHONY: reformat
reformat:
	isort $(SOURCE_FOLDER) tests
	black $(SOURCE_FOLDER) tests


.PHONY: lint
lint:
	$(PYTHON) -m pycodestyle . --exclude '.venv,setup.py,docs/*,scripts/*'
	pylint $(SOURCE_FOLDER)
	pylint --disable=missing-docstring,no-self-use tests/*
	isort --check-only $(SOURCE_FOLDER) tests
	black --check $(SOURCE_FOLDER) tests
	mypy $(SOURCE_FOLDER)


.PHONY: test tests
test tests:
	$(PYTHON) -m pytest tests/


.PHONY: radon
radon:
	radon cc $(SOURCE_FOLDER) --min c
	xenon --max-absolute C --max-modules C --max-average A $(SOURCE_FOLDER)/


##############################################################################################
# PROJECT-SPECIFIC COMMANDS                                                                  #
##############################################################################################


.PHONY: type-check
type-check:
	mypy $(SOURCE_FOLDER) --disallow-untyped-defs


.PHONY: download-requirements
download-requirements: download-spacy download-spacy-english
	rm -rf .word_embeddings
	mkdir -p .word_embeddings/embeddings_dict
	wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
	unzip crawl-300d-2M.vec.zip -d .
	python scripts/fasttext_to_pickle_script.py
	rm -rf crawl-300d-2M.vec.zip crawl-300d-2M.vec

.PHONY: download-spacy
download-spacy:
	python -m spacy download xx_ent_wiki_sm

.PHONY: download-spacy-english
download-spacy-english:
	python -m spacy download en_core_web_sm


.PHONY: zip
zip:
	zip -r source_code.zip \
		dataset \
		Dockerfile \
		gin_configs \
		LICENSE \
		Makefile \
		NOTICE \
		pyproject.toml \
		README.md \
		requirements.txt \
		scripts \
		setup.cfg \
		tests \
		tlc \
		train_freedom.sh \
		vocabularies \
		hyperparameter_grids.zip
