@echo off
REM ---------------------------------------------------
REM Batch file to run your LSTM Streamlit app on GPU
REM ---------------------------------------------------

REM Step 1: Change directory to your project folder
echo Changing directory to project folder...
cd /d "C:\Users\saipr\OneDrive\Desktop\MAIN UPDATED PROJECT"
echo.

REM Step 2: Activate your Conda environment
echo Activating Conda environment...
call C:\Users\saipr\miniconda3\Scripts\activate tf_gpu
echo.

REM Step 3: Check GPU availability in TensorFlow
echo Checking GPU availability...
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
if %errorlevel% neq 0 (
    echo Error: No GPU detected by TensorFlow! Check your setup.
    pause
    exit /b
)
echo GPU detected! Proceeding...
echo.

REM Step 4: Run the Streamlit app with GPU
echo Starting the Streamlit app...
streamlit run Main3.py --server.fileWatcherType none

REM Keep the window open after completion
pause
