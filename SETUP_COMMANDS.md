# Setup And Run Commands

## 1. Open the project folder

```powershell
cd "c:\Users\KIIT0001\OneDrive\Desktop\datasci\my_proj\miniProject"
```

## 2. Check Python

```powershell
py --version
```

If `py` does not work, try:

```powershell
python --version
```

## 3. Create a virtual environment

```powershell
py -m venv .venv
```

If that fails, try:

```powershell
python -m venv .venv
```

## 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 6. Install all requirements

```powershell
pip install -r requirements.txt
```

## 7. Add your Gemini API key

Open the `.env` file and replace the placeholder value:

```env
GEMINI_API_KEY=your_real_gemini_api_key_here
```

## 8. Run the Streamlit app

```powershell
streamlit run app.py
```

## 9. If `streamlit` is not recognized

```powershell
python -m streamlit run app.py
```

## 10. Optional verification commands

Check installed packages:

```powershell
pip list
```

Check Streamlit:

```powershell
python -m streamlit --version
```

## 11. Optional reinstall flow if something breaks

```powershell
deactivate
```

```powershell
Remove-Item -Recurse -Force .venv
```

```powershell
py -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

```powershell
python -m streamlit run app.py
```
