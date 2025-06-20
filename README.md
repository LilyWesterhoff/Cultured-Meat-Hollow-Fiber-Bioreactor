# Modelling Continuous Cell Harvest from a Hollow Fiber Bioreactor

This Streamlit application models "best-case" scenario continuous harvest of suspended cells in a hollow-fiber bioreactor, providing an interactive interface that allows users to specify bioreactor and cell parameters, solve for pressure, velocity, and concentration profiles, and generate visualizations.

## Features

- Interactive parameter adjustment for bioreactor and cell parameters
- Real-time calculation of pressure, velocity, and concentration profiles
- Visualization of results 

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/LilyWesterhoff/Cultured-Meat-Hollow-Fiber-Bioreactor.git
   cd porous_media_model
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app:
```
streamlit run solve_pmm_model.py
```

The app will be available at http://localhost:8501
Steps 2-3 can be automatically completed by running the setup shell script: zsh setup_venv.sh 

## License

[MIT License](LICENSE)