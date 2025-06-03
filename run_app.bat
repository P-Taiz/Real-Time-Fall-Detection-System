    @echo off
    rem This batch file is used to launch the Streamlit application

    rem Change to the directory containing this batch file
    cd /d "%~dp0"

    rem Activate virtual environment (if any)
    call venv\Scripts\activate

    rem Run Streamlit
    streamlit run app.py

    rem Keep the console window open until the user presses a key
    echo.
    echo Enter to close
    pause >nul
