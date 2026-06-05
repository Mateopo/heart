
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agente import agente_executor, obtener_historial

# ── La aplicación ─────────────────────────────────────────────────────
app = FastAPI(
    title="Agente Cardiológico",
    description="API con predicción de riesgo cardíaco y memoria persistente en Redis",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Modelos de datos ──────────────────────────────────────────────────
class Consulta(BaseModel):
    session_id: str | None = None
    mensaje:    str

class Respuesta(BaseModel):
    session_id:          str
    respuesta:           str
    mensajes_en_memoria: int

# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def raiz():
    return {"estado": "activo", "agente": "Cardiológico"}

@app.post("/consultar", response_model=Respuesta)
def consultar(consulta: Consulta):
    session_id = consulta.session_id or str(uuid.uuid4())

    resultado = agente_executor.invoke({
        "input":      consulta.mensaje,
        "session_id": session_id
    })

    return Respuesta(
        session_id=          session_id,
        respuesta=           resultado["respuesta"],
        mensajes_en_memoria= resultado["mensajes_en_memoria"]
    )

@app.delete("/limpiar/{session_id}")
def limpiar(session_id: str):
    obtener_historial(session_id).clear()
    return {"mensaje": f"Historial '{session_id}' eliminado"}
