# MLflow, ML experiments and model registry
- ### __ML experiment__: the process of building an ML model; The whole process in which a Data Scientist creates and optimizes a model.
- ### **Experiment run**: each trial in an ML experiment; Each run is within an ML experiment.
- ### **Run artifact**: any file associated with an ML run: Examples include the model itself, package versions...etc; Each Artifact is tied to an Experiment.
- ### **Experiment metadata**: metadata tied to each experiment.

<br />

# - Experiment Tracking: 

- ### The process of keeping track of all the relevant information from an ML experiment, which includes:
    - Source code.
    - Environment.
    - Data (different versions).
    - Model (different architectures).
    - Hyperparameters.
    - Metrics.

<br />

- ### Why is Experiment Tracking so important?
    1. Reproducibility
    2. Organization
    3. Optimization

>> __NOTE:__ Tracking experiments in spreadsheets is not enough: Error Prone, No standard format and Visibility & Collaboration.

<br />

# MLflow:
- ### "An Open source platform for the machine learning lifecycle"

- ### It's a Python package with four main modules:
    - Tracking
    - Models
    - Model registry
    - Projects.

<br />

## 1. Tracking:
- ### The MLFlow Tracking module allows you to organize your experiments into runs and to keep track of:

    1. Parameters.
    2. Metrics.
    3. Metadata.
    4. Artifacts.
    5. Models.

<br />

## 2. Models:
- ### It is a directory where the model is saved along with a few related files denoting its properties, associated information and environment dependencies. Generally a model is served by a variety of downstream tools for serving in real time through REST API or in batch mode. And, the format or flavour of the saved model is decided based on which downstream tool is going to use for model serving. For example mlflow Sklearn library allows loading the model back as a scikit-learn pipeline object while mlflow sagemaker tool wants the model in python_function format. mlflow provides a powerful option for defining required flavours in MLmodel file.

- ### A typical model directory contains the following files:
    - MLmodel - a YAML file describing model flavours, time created, run_id if the model was created in experiment tracking, signature denoting input and output details, input example, version of databricks runtime (if used) and mlflow version.
    - model.pkl - saved model pickle file.
    - conda.yaml - environment specifications for conda environment manager.
    - python_env.yaml - environment specification for virtualenv environment manager.
    - requirements.txt - list of pip installed libraries for dependencies.

<br />

## 3. Model Registry:
- ### Enterprises conduct a lot of experiments and move the selected models to production. Having said that a lot of models are created and saved in mlflow Models. Some of them are for new requirements and rest as updated models for same requirements. We needed a versioning and stage transitioning system for the models, that is fulfilled by mlflow Model Registry.

- ### Model Registry serves as a collaborative hub where teams share models and work together from experimentation to testing and production. It provides a set of APIs as well as a UI to manage the entire life cycle of an mlflow model.

### - Model Registry concepts to manage life cycle of mlflow model:
- Model - An mlflow model logged with one of the flavours mlflow.<model_flavour>.log_model().
- Registered model - An mlflow model registered on Model Registry. It has a unique name, contains versions, transitional stages, model lineage and other associated metadata.
- Model Version - Version of the registered model.
- Model Stage - Each distinct model version can be associated with one stage at a time. Stages supported are Staging, Production and Archived.
- Annotations and descriptions - Add useful information such as descriptions, data used, methodology etc. to the registered model.

![1](images/Screenshot%202023-03-26%20152941.png)

<br />

## 4. Projects
- ### It is a directory or a Git repo containing code files following a convention so that users or tools can run the project using its entry point(s). If a project contains multiple algorithms that can be run separately, in that multiple entry points are mentioned in MLProject file.

- ### Properties of a project:
    - Name - Name of the project.
    + Entry Points - Typically a .py or .sh file to run the entire project or some specific functionality, say an algorithm. List of entry points are mentioned in MLProject file
    - Environment - Specifications such as library dependencies for the software environment for the code to run. Supported environments - conda environments, virtualenv environments, docker environments.

