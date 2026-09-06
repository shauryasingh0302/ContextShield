# ContextShield 🛡️

ContextShield is a web application designed to check social-media posts (image/meme + caption) before publication to identify potential risks and suggest safer alternatives.

> **Note**: This project is built step-by-step for a student-level final year project.

---

## Project Structure

```text
ContextShield/
├── backend/            # FastAPI backend service
│   ├── main.py         # Application entry point and API endpoints
│   └── requirements.txt# Python dependencies
├── frontend/           # Next.js frontend application (TypeScript + Tailwind CSS)
│   ├── src/
│   │   └── app/        # Next.js App Router (pages and layouts)
│   ├── package.json
│   └── ...
└── README.md           # Project documentation and setup guide
```

---

## Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: v18 or higher (LTS recommended)
- **npm**: v9 or higher

---

## Getting Started

### 1. Run the Backend (FastAPI)

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. Check that the backend is running by opening:
   - **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health) (returns `{"status":"ok"}`)
   - **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Run the Frontend (Next.js)

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

4. Open your browser and go to:
   - [http://localhost:3000](http://localhost:3000)

You should see the ContextShield landing page:
> **ContextShield**  
> *"Check your social-media post before publishing."*

---

## API Endpoints (Current)

| Method | Endpoint | Description | Expected Response |
|---|---|---|---|
| `GET` | `/health` | Health check endpoint | `{"status": "ok"}` |
