import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="centered",
)

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "config_id" not in st.session_state:
    st.session_state.config_id = None


# ----------------------------
# Sidebar: API configuration
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Connect any OpenAI-compatible API.")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your API key",
        help="Your key is used for API requests and is not displayed.",
    )

    base_url = st.text_input(
        "API Endpoint",
        value="https://api.openai.com/v1",
        help="The API base URL, usually ending in /v1.",
    )

    model = st.text_input(
        "Model",
        value="gpt-4o-mini",
        help="Enter a model ID supported by your provider.",
    )

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Answer clearly and accurately.",
        height=120,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
    )

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("API credentials are held in Streamlit session state only "
               "if you choose to store them there. This app does not "
               "save them to a file or database.")


# ----------------------------
# Main interface
# ----------------------------
st.title("💬 AI Chatbot")
st.caption("Your configurable AI assistant")

if not api_key.strip():
    st.info("Enter your API key in the sidebar to start chatting.")
    st.stop()

if not base_url.strip() or not model.strip():
    st.warning("Please enter both an API endpoint and a model name.")
    st.stop()

# Remove accidental whitespace and identify config changes.
base_url = base_url.strip().rstrip("/")
model = model.strip()

# Reset the conversation when the provider configuration changes.
# The API key is included in the comparison but never displayed.
import hashlib

config_id = hashlib.sha256(
    f"{api_key}\n{base_url}\n{model}".encode()
).hexdigest()

if st.session_state.config_id != config_id:
    st.session_state.messages = []
    st.session_state.config_id = config_id

# Create the OpenAI-compatible client.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=2,
)

# Display conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input.
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the request.
    api_messages = [
        {"role": "system", "content": system_prompt},
        *st.session_state.messages,
    ]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )

            def response_chunks():
                for chunk in stream:
                    if not chunk.choices:
                        continue

                    content = chunk.choices[0].delta.content

                    if isinstance(content, str):
                        yield content

            response = st.write_stream(response_chunks())

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        except Exception as exc:
            # Avoid showing credentials or request headers to the user.
            st.error(
                "The API request failed. Check your API key, endpoint, "
                "model name, provider availability, and API limits."
            )
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")

            # Remove the failed user turn so it can be retried.
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
                and st.session_state.messages[-1]["content"] == prompt
            ):
                st.session_state.messages.pop()