"""Canonical Daft model usage patterns.

Each model lives in its own package with a consistent layout:

- ``model.py``    — the ``@daft.cls`` UDF, result schema, and pure helpers.
                    Importable anywhere; never imports ``modal``.
- ``modal_app.py`` — the Modal deployment shell (image, volumes, entrypoints).
- ``README.md``   — backend choice and run commands.

End-to-end pipelines that compose these models live in ``pipelines/``.
"""
