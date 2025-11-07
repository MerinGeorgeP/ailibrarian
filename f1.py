"""
AI Librarian Frontend (with Authentication)
-------------------------------------------
✅ User signup/login (JWT-based)
✅ Upload PDFs (user-specific)
✅ Search, Delete, Summarize (protected by token)
"""

import streamlit as st
import requests

# ----------------------------
# CONFIG
# ----------------------------
BACKEND_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="AI Librarian", page_icon="📚", layout="wide")

# ----------------------------
# SESSION STATE
# ----------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None


# ----------------------------
# AUTH FUNCTIONS
# ----------------------------
def signup(username, password):
    data = {"username": username, "password": password}
    res = requests.post(f"{BACKEND_URL}/signup", data=data)
    return res


def login(username, password):
    data = {"username": username, "password": password}
    res = requests.post(f"{BACKEND_URL}/login", data=data)
    if res.status_code == 200:
        token = res.json()["access_token"]
        return token
    return None


def authorized_request(method, endpoint, **kwargs):
    """Attach token to requests automatically"""
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    if "headers" in kwargs:
        kwargs["headers"].update(headers)
    else:
        kwargs["headers"] = headers
    return requests.request(method, f"{BACKEND_URL}{endpoint}", **kwargs)


# ----------------------------
# AUTH UI
# ----------------------------
def login_ui():
    st.title("🔐 AI Librarian Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            token = login(username, password)
            if token:
                st.session_state.access_token = token
                st.session_state.username = username
                st.success(f"✅ Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Sign Up")
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create Account"):
            res = signup(new_user, new_pass)
            if res.status_code == 200:
                st.success("🎉 Account created! Please log in.")
            else:
                try:
                   error_msg = res.json().get("detail", "Signup failed")
                except ValueError:
                    error_msg = res.text or "Signup failed"
                st.error(error_msg)
   



# ----------------------------
# LOGGED-IN DASHBOARD
# ----------------------------
def dashboard():
    st.sidebar.success(f"👋 Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.access_token = None
        st.session_state.username = None
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["Upload PDF", "Search PDFs", "Manage PDFs", "Summarize PDF"])

    # =============== Upload ===============
    with tab1:
        st.header("📤 Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file and st.button("Upload"):
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                res = authorized_request("POST", "/upload", files=files)
                if res.status_code == 200:
                    st.success("✅ Uploaded successfully!")
                else:
                    st.error(res.text)

    # =============== Search ===============
    with tab2:
        st.header("🔍 Search My PDFs")
        query = st.text_input("Enter a keyword or question")
        top_k = st.slider("Results to show", 1, 10, 5)
        if st.button("Search"):
            with st.spinner("Searching..."):
                res = authorized_request("POST", "/search", json={"query": query, "top_k": top_k})
                if res.status_code == 200:
                    results = res.json()
                    if not results:
                        st.info("No matches found.")
                    else:
                        for r in results:
                            st.markdown(f"### 📄 {r['original_filename']}")
                            st.caption(r.get("excerpt", "")[:600] + "...")
                            st.markdown(f"[⬇️ Download]({BACKEND_URL}{r['download_url']})")
                            st.divider()
                else:
                    st.error(res.text)

    # =============== Manage PDFs ===============
    with tab3:
        st.header("🗂️ Manage My PDFs")
        res = authorized_request("GET", "/pdfs")
        if res.status_code == 200:
            pdfs = res.json()
            if not pdfs:
                st.info("No PDFs uploaded yet.")
            else:
                for pdf in pdfs:
                    col1, col2, col3 = st.columns([5, 2, 1])
                    with col1:
                        st.markdown(f"### 📄 {pdf['original_filename']}")
                        st.caption(pdf.get("excerpt", "")[:500] + "...")
                    with col2:
                        st.markdown(f"[⬇️ Download]({BACKEND_URL}/download/{pdf['filename']})")
                    with col3:
                        if st.button("🗑️ Delete", key=pdf["id"]):
                            del_res = authorized_request("DELETE", f"/delete/{pdf['id']}")
                            if del_res.status_code == 200:
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error(del_res.text)
        else:
            st.error(res.text)

    # =============== Summarize ===============
    with tab4:
        st.header("🧠 Summarize My PDFs")
        res = authorized_request("GET", "/pdfs")
        if res.status_code == 200:
            pdfs = res.json()
            if not pdfs:
                st.info("No PDFs available.")
            else:
                selected_pdf = st.selectbox(
                    "Choose a PDF", [p["original_filename"] for p in pdfs]
                )
                pdf_id = next((p["id"] for p in pdfs if p["original_filename"] == selected_pdf), None)
                if st.button("Generate Summary"):
                    with st.spinner("Summarizing..."):
                        sum_res = authorized_request("GET", f"/summarize/{pdf_id}")
                        if sum_res.status_code == 200:
                            summary = sum_res.json().get("summary")
                            st.subheader("📘 Summary:")
                            st.write(summary)
                        else:
                            st.error(sum_res.text)
        else:
            st.error("Failed to load PDFs.")


# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if st.session_state.access_token:
    dashboard()
else:
    login_ui()
