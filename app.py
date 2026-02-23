import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(page_title="Medical RAG AI Assistant", layout="wide")

st.title("Medical RAG AI Assistant")
st.caption("Ask a medical question.")

#Backend health check
backend_ok = True
try:
    h = requests.get(HEALTH_URL, timeout=3)
    backend_ok = h.ok
except Exception:
    backend_ok = False

if not backend_ok:
    st.warning("Backend is not reachable.")

#Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("sources"):
            with st.expander("Sources (file + page)"):
                for s in sorted(set(msg["sources"])):
                    st.write(f"- {s}")

# User input
query = st.chat_input("Ask a medical question...")

if query:
    # store user msg
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical documents..."):
            try:
                response = requests.post(API_URL, json={"question": query}, timeout=90)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer returned from backend.")
                sources = data.get("sources", [])
                
                confidence = float(data.get("confidence", 0.0))
                normalized_query = data.get("normalized_query", query)

            except requests.exceptions.RequestException as e:
                answer = f"Backend error: {e}"
                sources = []
                confidence = 0.0
                normalized_query = query

        # Answer
        st.write(answer)

        # Query normalization hint
        if normalized_query.strip().lower() != query.strip().lower():
            st.info(f"Normalized query: {normalized_query}")

        # Sources
        if sources:
            with st.expander("Sources (file + page)"):
                for s in sorted(set(sources)):
                    st.write(f"- {s}")

        

    # store assistant msg
    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence, 
    }
    st.session_state.messages.append(assistant_msg)


# streamlit run app.py