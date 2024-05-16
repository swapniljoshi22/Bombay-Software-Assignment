# Bombay-Software-Assignment

###Image Classification Web Application
This is a Flask web application for image classification using a trained Support Vector Machine (SVM) model.

Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your_username/your_repository.git
Navigate to the project directory:

bash
Copy code
cd your_repository
Create a virtual environment (optional but recommended):

Copy code
python -m venv venv
Activate the virtual environment:

On Windows:
Copy code
venv\Scripts\activate
On macOS and Linux:
bash
Copy code
source venv/bin/activate
Install the required Python libraries using the provided environment file:

Copy code
pip install -r environment_assignment.yml
Usage
Run the Flask application:

Copy code
python app_assignment.py
Open a web browser and go to http://localhost:5000.

Upload an image using the provided interface.

Files
app_assignment.py: The Flask application script.
svm_model.pkl: Trained SVM model saved as a pickle file.
environment_assignment.yml: Environment file containing the required Python libraries.
Dependencies
Flask
NumPy
OpenCV
scikit-learn
