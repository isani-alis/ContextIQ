#!/bin/bash
# Start ContextIQ — backend + frontend

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting ContextIQ backend..."
cd "$PROJECT_DIR"
venv/bin/uvicorn backend.api.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "Starting ContextIQ frontend..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "ContextIQ running:"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
