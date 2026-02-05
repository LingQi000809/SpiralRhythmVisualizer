# Backend

## libraries
I'm using arm64 python3.12.10.

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
pip freeze > requirements.txt
```

## Spin up backend

```
python3 -m uvicorn audio_analysis:app --reload
```

# Frontend
Using vite for ReactJS.

## libraries
```
npm install axios
```