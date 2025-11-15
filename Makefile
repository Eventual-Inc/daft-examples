.PHONY: uv-env

venv:
	uv venv .venv
	uv pip install daft openai pillow numpy ipykernel ipywidgets 

