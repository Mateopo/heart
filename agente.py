
import os
import numpy as np
import pandas as pd
import joblib
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.tools import tool

# ── Cargar modelos serializados ───────────────────────────────────────
loaded_model_rf       = joblib.load("random_forest_tuned_model.joblib")
loaded_scaler_rf      = joblib.load("scaler_rf.joblib")
loaded_label_encoders = joblib.load("label_encoders.joblib")
loaded_mediana_chol   = joblib.load("mediana_chol.joblib")
loaded_feature_names  = joblib.load("feature_names.joblib")

VENTANA = 10

# ── Memoria Redis ─────────────────────────────────────────────────────
def obtener_historial(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ["REDIS_URL"],
        ttl=3600
    )

# ── Tool ──────────────────────────────────────────────────────────────
@tool
def predecir_enfermedad_cardiaca(
    age: int, sex: str, chestpaintype: str, restingbp: int,
    cholesterol: int, fastingbs: int, restingecg: str, maxhr: int,
    exerciseangina: str, oldpeak: float, st_slope: str,
) -> str:
    """Predice el riesgo de enfermedad cardíaca usando el modelo Random Forest."""

    raw = pd.DataFrame({
        'age': [age], 'sex': [sex], 'chestpaintype': [chestpaintype],
        'restingbp': [restingbp], 'cholesterol': [cholesterol],
        'fastingbs': [fastingbs], 'restingecg': [restingecg],
        'maxhr': [maxhr], 'exerciseangina': [exerciseangina],
        'oldpeak': [oldpeak], 'st_slope': [st_slope],
    })

    raw['cholesterol'] = raw['cholesterol'].replace(0, np.nan).fillna(loaded_mediana_chol)

    for col in ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope']:
        try:
            raw[col] = loaded_label_encoders[col].transform(raw[col])
        except ValueError as e:
            return f"Error: valor inválido en '{col}'. Detalle: {e}"

    raw = raw.drop(columns=[c for c in ['restingbp', 'restingecg'] if c in raw.columns])
    raw = raw.reindex(columns=loaded_feature_names, fill_value=0)
    num_cols = raw.select_dtypes(include=np.number).columns
    raw[num_cols] = loaded_scaler_rf.transform(raw[num_cols])

    pred  = loaded_model_rf.predict(raw)[0]
    proba = loaded_model_rf.predict_proba(raw)[0]
    etiqueta = "Enfermedad cardíaca" if int(pred) == 1 else "Sin enfermedad cardíaca"
    return (
        f"Predicción: {etiqueta} (clase={int(pred)}). "
        f"Probabilidades — Sin enfermedad: {proba[0]:.2f}, Con enfermedad: {proba[1]:.2f}."
    )

# ── LLM y tools ───────────────────────────────────────────────────────
llm_agente = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=300,
)

tools         = [predecir_enfermedad_cardiaca]
tools_by_name = {t.name: t for t in tools}
llm_con_tools = llm_agente.bind_tools(tools)

SYSTEM_CONTENT = """Eres un asistente médico especializado en cardiología, diseñado para
ayudar a usuarios a interpretar predicciones de riesgo cardíaco y a entender
exámenes y conceptos relacionados.

Tienes acceso a una herramienta llamada `predecir_enfermedad_cardiaca` que
ejecuta un modelo Random Forest entrenado sobre el dataset Heart Failure.

Reglas:
1. Si el usuario te entrega datos clínicos de un paciente y quiere una predicción, usa la herramienta.
   Si falta algún campo obligatorio, pídelo amablemente antes de llamarla.
2. Si la pregunta es informativa (qué es el colesterol, cómo se interpreta un
   oldpeak, consejos de prevención, etc.) responde directamente sin usar la herramienta.
3. Después de una predicción, explica el resultado en lenguaje claro y agrega
   2-3 recomendaciones generales de salud cardiovascular.
4. Siempre aclara que NO sustituyes la consulta con un profesional de la salud.
5. Responde en español, con tono cercano y empático."""

# ── Agente con memoria ────────────────────────────────────────────────
class AgenteCardio:
    def __init__(self, llm, tools_by_name, system_content, max_iter=4, verbose=True):
        self.llm            = llm
        self.tools_by_name  = tools_by_name
        self.system_content = system_content
        self.max_iter       = max_iter
        self.verbose        = verbose

    def invoke(self, payload):
        pregunta   = payload["input"]
        session_id = payload.get("session_id", "default")

        historial_redis  = obtener_historial(session_id)
        mensajes_previos = historial_redis.messages[-VENTANA:]

        mensajes = [
            SystemMessage(content=self.system_content),
            *mensajes_previos,
            HumanMessage(content=pregunta),
        ]

        respuesta_final = None

        for _ in range(self.max_iter):
            respuesta = self.llm.invoke(mensajes)
            mensajes.append(respuesta)

            if not getattr(respuesta, "tool_calls", None):
                respuesta_final = respuesta.content
                break

            for call in respuesta.tool_calls:
                tool_fn   = self.tools_by_name[call["name"]]
                if self.verbose:
                    print(f"→ Tool: {call['name']}({call['args']})")
                resultado = tool_fn.invoke(call["args"])
                if self.verbose:
                    print(f"  ← {resultado}\n")
                mensajes.append(
                    ToolMessage(content=str(resultado), tool_call_id=call["id"])
                )

        if respuesta_final is None:
            respuesta_final = "Se alcanzó el máximo de iteraciones sin respuesta final."

        historial_redis.add_user_message(pregunta)
        historial_redis.add_ai_message(respuesta_final)

        return {
            "respuesta":           respuesta_final,
            "session_id":          session_id,
            "mensajes_en_memoria": len(historial_redis.messages)
        }


agente_executor = AgenteCardio(
    llm=llm_con_tools,
    tools_by_name=tools_by_name,
    system_content=SYSTEM_CONTENT,
    max_iter=4,
    verbose=False
)

print("✓ agente.py cargado correctamente")
