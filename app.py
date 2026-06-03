import streamlit as st
from services.recommendation_service import get_recommendation

st.set_page_config(
    page_title="Laptop AI · Rekomendasi Cerdas",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS via st.markdown (lebih reliable dari file load) ──────
def load_css():
    try:
        with open("assets/style.css") as f:
            css = f.read()
    except FileNotFoundError:
        css = ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# ── Session State ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:2.8rem;margin-bottom:10px'>💻</div>",
        unsafe_allow_html=True
    )
    st.markdown("## Laptop AI")
    st.markdown(
        "<p style='color:#8B92A5;font-size:0.83rem;line-height:1.6;margin-bottom:0'>"
        "Asisten rekomendasi laptop berbasis AI, dirancang khusus untuk mahasiswa Kendari."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#4A5068;margin-bottom:6px'>Teknologi</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px'>"
        "<span style='background:#1E2330;color:#4F8EF7;border:1px solid rgba(79,142,247,0.25);"
        "border-radius:6px;font-size:0.68rem;padding:2px 9px;font-weight:500'>RAG</span>"
        "<span style='background:#1E2330;color:#4F8EF7;border:1px solid rgba(79,142,247,0.25);"
        "border-radius:6px;font-size:0.68rem;padding:2px 9px;font-weight:500'>Gemini 2.5 Flash</span>"
        "<span style='background:#1E2330;color:#4F8EF7;border:1px solid rgba(79,142,247,0.25);"
        "border-radius:6px;font-size:0.68rem;padding:2px 9px;font-weight:500'>ChromaDB</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#4A5068;margin-bottom:8px'>Coba tanya</p>",
        unsafe_allow_html=True
    )

    examples = [
        ("💰", "Budget 6 juta untuk kuliah"),
        ("🎮", "Laptop gaming under 10 juta"),
        ("🎨", "Desain grafis ringan & portabel"),
        ("💻", "Coding & programming terbaik"),
    ]
    for icon, text in examples:
        if st.button(f"{icon}  {text}", key=f"ex_{text}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": text})
            with st.spinner("🔍 Mencari rekomendasi..."):
                answer = get_recommendation(text)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    st.markdown("---")

    msg_count = len(st.session_state.messages)
    rec_count = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    st.markdown(
        f"<div style='display:flex;gap:8px;margin-bottom:4px'>"
        f"<div style='flex:1;background:#1E2330;border-radius:8px;padding:8px;text-align:center'>"
        f"<div style='font-family:'Sora',sans-serif;font-size:1.15rem;font-weight:800;color:#4F8EF7'>{msg_count}</div>"
        f"<div style='font-size:0.62rem;color:#4A5068'>Pesan</div></div>"
        f"<div style='flex:1;background:#1E2330;border-radius:8px;padding:8px;text-align:center'>"
        f"<div style='font-family:'Sora',sans-serif;font-size:1.15rem;font-weight:800;color:#7ECBA1'>{rec_count}</div>"
        f"<div style='font-size:0.62rem;color:#4A5068'>Rekomendasi</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if st.button("🗑  Hapus Riwayat Chat", use_container_width=True, key="delete"):
        st.session_state.messages = []
        st.rerun()

# ── Header ───────────────────────────────────────────────────
st.title("💻 Chatbot Rekomendasi Laptop Kendari")
st.caption("Temukan laptop terbaik sesuai jurusan, kebutuhan, dan budget kamu — powered by AI.")
st.markdown("---")

# ── Empty State ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;max-width:500px;margin:0 auto">
            <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
            <h3 style="font-family:'Sora',sans-serif;font-size:1.25rem;font-weight:800;
                color:#EEF0F5;margin-bottom:0.5rem">Halo! Saya siap membantu</h3>
            <p style="color:#8B92A5;font-size:0.88rem;line-height:1.7;margin-bottom:1.5rem">
                Ceritakan kebutuhan kamu — jurusan, budget, dan aktivitas utama —
                lalu saya akan rekomendasikan laptop terbaik di Kendari.
            </p>
            <div style="background:#1E2330;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1rem 1.25rem;text-align:left">
                <p style="color:#4A5068;font-size:0.68rem;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px">Contoh pertanyaan</p>
                <p style="color:#8B92A5;font-size:0.84rem;margin-bottom:6px">
                    💬 <em>"Saya mahasiswa Teknik, budget 8 juta, butuh laptop kuat"</em></p>
                <p style="color:#8B92A5;font-size:0.84rem;margin-bottom:6px">
                    💬 <em>"Laptop terbaik untuk desain grafis di bawah 12 juta"</em></p>
                <p style="color:#8B92A5;font-size:0.84rem;margin:0">
                    💬 <em>"Laptop tipis untuk mahasiswa yang sering mobile"</em></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Chat History ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ───────────────────────────────────────────────
prompt = st.chat_input("Ceritakan kebutuhan laptopmu, misal: mahasiswa Informatika budget 8 juta...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari laptop terbaik untuk Anda..."):
            answer = get_recommendation(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Sistem Rekomendasi Laptop Berbasis RAG")