# Experiment 7: Continuous Integration Using GitHub Actions

## Aim
To implement a Continuous Integration (CI) pipeline using GitHub Actions to automate the machine learning build process, model training, testing, validation, and build report generation.

## Objectives
1. To configure GitHub Actions for continuous integration.
2. To automate the machine learning build process.
3. To implement automated testing using Pytest.
4. To validate machine learning training scripts.
5. To generate and store build reports.
6. To document the CI pipeline.

## Software and Technologies Used
- Python 3.13
- Git and GitHub
- GitHub Actions
- Pandas
- Scikit-learn
- Pytest
- YAML
- VS Code

## Implementation

### 1. Project Setup
Created an MLOps experiment folder and installed the required Python libraries.

### 2. Machine Learning Model Training
Developed a Linear Regression model to predict student marks based on study hours using Scikit-learn.

### 3. Automated Testing
Created test cases using Pytest to verify:
- Successful model training.
- Model prediction functionality.
- Model accuracy validation.

### 4. GitHub Actions Configuration
Created a CI workflow using a YAML configuration file inside `.github/workflows/ci.yml`.

### 5. Automated Build and Testing
Configured GitHub Actions to automatically:
- Checkout the repository.
- Set up Python.
- Install dependencies.
- Execute the model training script.
- Run automated test cases.
- Generate build reports.
- Upload reports as workflow artifacts.

### 6. Build Report Generation
Generated XML test reports and text-based build reports to record the CI execution results.

## Result
Successfully implemented a Continuous Integration pipeline using GitHub Actions. The machine learning training script and all three automated test cases executed successfully. The pipeline generated and uploaded build reports.

## Conclusion
The experiment demonstrates how Continuous Integration automates machine learning project validation, reduces manual testing efforts, and improves software development reliability.