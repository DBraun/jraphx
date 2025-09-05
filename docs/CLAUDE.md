# Building Documentation

To build the documentation:

1. [Build and install](https://github.com/DBraun/jraphx) JraphX from source.
1. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```
   pip install sphinx-rtd-theme
   ```
1. Generate the documentation file via:
   ```
   cd docs
   make html
   ```
1. Launch an HTTP server
    ```
   python -m http.server -d build/html
   ```

The documentation is now available to view by opening `http://localhost:8000`.
