# Build, test, and lint psipy.
#
# Run 'make help' to get a list of known commands.
#
# Dependencies:
#   flake8

all: clean lint test build

help:
	@echo ""
	@echo "    clean"
	@echo "        Recursively remove python artifacts and *~."
	@echo "    test-code"
	@echo "        Run pytest"
	@echo "    test-coverage"
	@echo "        Same as test-code but also producing an html coverage report."
	@echo ""

build:
	@echo "Not implemented yet."

lint-code:
	flake8 psipy

lint-examples:
	flake8 examples/*.py

lint-tests:
	flake8 tests

lint: lint-code lint-examples lint-tests

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -type d -empty -delete

clean_all: clean
	find . -name '*coverage*' -exec rm -rf {} +
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +
	rm -rf .eggs .pytest_cache coverage_html

doc:
	make -C docs all

doc_remove:
	make -C docs rm

editable_install:
	uv pip install -e ".[dev]"

test: clean
	python3 -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
	pytest
	flake8 psipy
	mypy psipy

test_venv:
	uv venv .test_venv
	. .test_venv/bin/activate
	uv pip install -e ".[dev]"

dist/psipy.%:
	# *nix: %/$* == so
	#  win: %/$* == pyd
	./tools/pypack psipy --output dist --no-tests --cleanup

lock:
	uv pip compile -o requirements.txt requirements.in
	uv pip compile -o requirements-dev.txt requirements-dev.in -c requirements.txt

sync:
	uv pip sync requirements.txt

sync-dev:
	uv pip sync requirements-dev.txt

## DOCKER
DOCKER_SUFFIX?=
DOCKER_IMAGE?=psipy.azurecr.io/psipy
DOCKER_TAG?=latest
GPU?=0

ifneq ($(GPU),1)
	DOCKER_BASE = ubuntu:18.04
	DOCKER_SUFFIX =
else
	DOCKER_BASE = nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
	DOCKER_SUFFIX = -gpu
endif

docker/pull:
	docker pull $(DOCKER_IMAGE):$(DOCKER_TAG)$(DOCKER_SUFFIX)
	docker pull $(DOCKER_IMAGE):$(DOCKER_TAG)$(DOCKER_SUFFIX)-base

docker/build:
	echo "Building $(DOCKER_TAG)-base"
	DOCKER_BUILDKIT=1 docker build . \
	  --tag $(DOCKER_IMAGE):$(DOCKER_TAG)$(DOCKER_SUFFIX)-base \
	  --file tools/docker/Dockerfile.base \
	  --build-arg BUILDKIT_INLINE_CACHE=1 \
	  --build-arg BASE_IMAGE=$(DOCKER_BASE) \
	  --cache-from $(DOCKER_IMAGE):latest$(DOCKER_SUFFIX)-base
	echo "Building $(DOCKER_TAG)"
	DOCKER_BUILDKIT=1 docker build . \
	  --tag $(DOCKER_IMAGE):$(DOCKER_TAG)$(DOCKER_SUFFIX) \
	  --file tools/docker/Dockerfile \
	  --build-arg BUILDKIT_INLINE_CACHE=1 \
	  --build-arg BASE_IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)$(DOCKER_SUFFIX)-base \
	  --cache-from $(DOCKER_IMAGE):latest$(DOCKER_SUFFIX)
