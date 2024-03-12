# SeaSight

SeaSight is a deep learning project developed as the final assignment for DS 6050 (Deep Learning) in the University of Virginia's Master of Science in Data Science program. This project leverages neural network architectures to classify a wide range of sea vessel images.

## Setup

Follow the guide below to setup your development environment.

### Prerequisites

Before setting up the project, ensure you have Poetry installed. If you do not have Poetry installed, follow the installation instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

### Development Environment

1. **Clone the Repository**

    First, clone the project.

    ```bash
    git clone https://github.com/twt6xy/SeaSight.git
    cd SeaSight
    ```

2. **Install Dependencies**
    To install all dependencies listed in `pyproject.toml` with Poetry, execute this command in your project's root directory, ensuring it adheres to versions in `poetry.lock`:

    ```bash
    poetry install
    ```

3. **Activating the Environment**
    Poetry creates a virtual environment for your project to manage dependencies separately from your global Python installation. To activate this environment, use the command:

    ```bash
    poetry shell
    ```

4. **[Optional] Setting Up Pre-commit Hooks**
    Pre-commit hooks automatically enforce code quality and style standards before committing changes. To set them up, run:

    ```bash
    poetry run pre-commit install
    ```

    Here's what each hook does:

    - **`black`**: Formats Python code for consistent style.

    - **`isort`**: Sorts import statements alphabetically.

    - **`ruff`**: Lints Python code for syntax and stylistic issues.

    - **`pytest`**: Runs tests to ensure code functionality.

    <br>
    After installation, these hooks auto-run on staged files, blocking commits with detected issues until fixed. Verify installation with:

    ```bash
    poetry run pre-commit run
    ```
