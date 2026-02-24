FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "-m", "streamlit", "run", "Stock agent app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

Commit it directly on GitHub (no laptop needed ✅)

---

### Step 5 — Deploy on Cloud Run
1. Go to **Cloud Run** → Click **Create Service**
2. Select **"Continuously deploy from a repository"**
3. Click **Set up with Cloud Build**
4. Connect your **GitHub account**
5. Select your repo `Stock-Agent-App`
6. Branch: `main`
7. Build type: **Dockerfile**
8. Click **Save**

---

### Step 6 — Configure the Service
- **Port**: `8080`
- **Allow unauthenticated requests**: ✅ Yes (so anyone can access it)
- **Memory**: 1GB minimum (your app needs it for ARIMA)
- Click **Create**

---

### Step 7 — Get Your Live URL
After 2-3 minutes you get a URL like:
```
https://stock-agent-app-xxxxxxxx-uc.a.run.app
