# ffsimquery service
Users can maximize the potential of their fantasy team by simulating round-by-round selections in a 10-team 2023 NFL Snake Draft!

We've run nearly 100,000 (for now) simulated fantasy drafts and combined the results with predictive machine learning algorithms to provide estimates for which roster choices will help your team the most!

## Download the tool!
Before you begin, make sure you have the following installed!

- Python 3.9
- pip
- git

### Installation steps

1.  **Clone the repository**  
    ```
    git clone https://github.com/kevniu95/ffwrapped.git
    ```
    Navigate to the project directory
    ```
    cd ffwrapped
    ```
2.  **Set up a virtual environment**
    ```
    python3 -m venv ffwrapped_env
    ```
    Activate the virtual environment

    - **Windows**
      ```
      ffwrapped_env\Scripts\activate
      ```
    - **Mac**
      ```
      source ffwrapped_env/bin/activate
      ```
3.  **Install dependencies**
      ```
      pip install -r requirements.txt
      ```
4.  **Run the application**
      ```
      python -m src.modules.simQuery.cli
      ```
    Be sure to execute this from the root of project directory ```path/to/ffwrapped```


## Example
Oh wow, you were chosen as the 9th team in the first round of a 10-team HPPR league- I was last year too! Let's see how we can use the ffsimquery tool to help us out!

1. 