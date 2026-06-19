"""
login.py
----------
Authentication UI module.

Implements every screen in the auth flow as a Streamlit render function:

    render_auth_page()       -> dispatcher, called by app.py when no user
                                 is logged in. Reads st.session_state.auth_view.
        - "login"
        - "register"
        - "otp"               (post-login OTP verification)
        - "forgot"            (request a password-reset OTP)
        - "forgot_otp"        (verify the reset OTP)
        - "reset_password"    (set a new password)

Session-state contract (all keys are created lazily, never assumed to exist):

    st.session_state.user                 -> logged-in user dict, or None
    st.session_state.auth_view            -> which auth screen is active
    st.session_state.pending_otp_user     -> user dict awaiting login-OTP
    st.session_state.pending_reset_user   -> user dict awaiting reset-OTP
    st.session_state.last_otp_banner      -> dev-mode OTP banner text (demo aid)

   # FUTURE SCOPE: add OAuth / SSO (Google Workspace, Okta, Azure AD) as an
   # additional identity provider alongside username+password+OTP.
"""

import streamlit as st

import config
import database
import otp_service
import security


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _set_view(view: str):
    st.session_state.auth_view = view
    st.rerun()


def _centered(width_ratio=(1, 1.4, 1)):
    """Return the centered middle column used by every auth screen."""
    left, mid, right = st.columns(width_ratio)
    return mid


def _password_policy():
    return database.get_setting("password_policy") or config.DEFAULT_PASSWORD_POLICY


def _otp_settings():
    return database.get_setting("otp_settings") or config.DEFAULT_OTP_SETTINGS


def _logo_header(subtitle: str):
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:18px;">
            <div style="font-size:2rem; font-weight:800;
                        background:linear-gradient(95deg,#0369A1,#0EA5E9 45%,#22D3EE);
                        -webkit-background-clip:text; background-clip:text; color:transparent;">
                🛡️ {config.APP_NAME}
            </div>
            <div style="color:var(--ss-muted,#64748B); font-size:0.95rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _otp_dev_banner(send_result: dict):
    """Surface the OTP code directly when running without real SMTP (dev mode)."""
    if send_result.get("dev_mode"):
        st.info(
            f"📧 **Dev mode (no SMTP configured):** your one-time code is "
            f"**{send_result['otp']}** — in production this is emailed instead.",
            icon="🔧",
        )
        if send_result.get("error"):
            st.caption(f"(SMTP send failed: {send_result['error']} — falling back to dev mode.)")
    else:
        st.success("📧 A one-time verification code has been sent to your email.")


# ---------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------
def _render_login():
    mid = _centered()
    with mid:
        _logo_header("Sign in to your Security Operations Center")
        with st.container(border=True):
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="e.g. admin")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = database.authenticate(username, password)
                    if not user:
                        st.error("Invalid credentials or inactive account.")
                        database.log_action(None, "login_failed", f"username={username}")
                    else:
                        send_result = otp_service.send_otp(user, purpose="login")
                        st.session_state.pending_otp_user = user
                        st.session_state.last_otp_banner = send_result
                        database.log_action(user["id"], "login_otp_sent")
                        _set_view("otp")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create an account", use_container_width=True):
                    _set_view("register")
            with c2:
                if st.button("Forgot password?", use_container_width=True):
                    _set_view("forgot")

        st.caption(
            f"Demo admin account — username **{config.SEED_ADMIN_USERNAME}**, "
            f"password **{config.SEED_ADMIN_PASSWORD}**"
        )


def _render_register():
    mid = _centered()
    with mid:
        _logo_header("Create your analyst account")
        policy = _password_policy()
        with st.container(border=True):
            with st.form("register_form"):
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                username = st.text_input("Username")
                col_a, col_b = st.columns(2)
                with col_a:
                    password = st.text_input("Password", type="password")
                with col_b:
                    confirm = st.text_input("Confirm Password", type="password")
                st.caption(
                    f"Password must be at least {policy.get('min_length', 8)} characters and include "
                    "uppercase, lowercase, a number, and a special character."
                )
                submitted = st.form_submit_button("Register", use_container_width=True, type="primary")

            if submitted:
                errors = []
                if not all([full_name, email, username, password, confirm]):
                    errors.append("All fields are required.")
                if email and not security.is_valid_email(email):
                    errors.append("Please enter a valid email address.")
                if password != confirm:
                    errors.append("Passwords do not match.")
                if password:
                    errors.extend(security.validate_password_policy(password, policy))
                if username and database.get_user_by_username(username):
                    errors.append("That username is already taken.")
                if email and database.get_user_by_email(email):
                    errors.append("An account with that email already exists.")

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    user = database.create_user(full_name, email, username, password, role="user")
                    database.log_action(user["id"], "user_registered")
                    st.success("Account created! You can now log in.")
                    if st.button("Go to Login", type="primary"):
                        _set_view("login")

            if st.button("← Back to Login", use_container_width=True):
                _set_view("login")


def _render_otp(purpose="login"):
    is_login = purpose == "login"
    user = st.session_state.get("pending_otp_user") if is_login else st.session_state.get("pending_reset_user")

    mid = _centered()
    with mid:
        _logo_header("Two-factor verification")

        if not user:
            st.warning("Your session expired. Please log in again.")
            if st.button("← Back to Login"):
                _set_view("login")
            return

        with st.container(border=True):
            st.markdown(f"Verifying **{user['email']}**")
            banner = st.session_state.get("last_otp_banner")
            if banner:
                _otp_dev_banner(banner)

            otp_settings = _otp_settings()
            code = st.text_input(
                f"Enter the {otp_settings.get('otp_length', 6)}-digit code",
                max_chars=otp_settings.get("otp_length", 6),
            )

            c1, c2 = st.columns(2)
            with c1:
                verify_clicked = st.button("Verify", type="primary", use_container_width=True)
            with c2:
                resend_clicked = st.button("Resend Code", use_container_width=True)

            if resend_clicked:
                result = otp_service.send_otp(user, purpose=purpose)
                st.session_state.last_otp_banner = result
                st.rerun()

            if verify_clicked:
                ok, message = otp_service.verify_otp(user, code, purpose=purpose)
                if ok:
                    if is_login:
                        st.session_state.user = user
                        st.session_state.pending_otp_user = None
                        st.session_state.last_otp_banner = None
                        database.log_action(user["id"], "login_success")
                        st.session_state.route = "app"
                        st.rerun()
                    else:
                        st.session_state.pending_reset_user = user
                        database.log_action(user["id"], "reset_otp_verified")
                        _set_view("reset_password")
                else:
                    st.error(message)

            if st.button("← Back to Login", use_container_width=True):
                st.session_state.pending_otp_user = None
                st.session_state.pending_reset_user = None
                _set_view("login")


def _render_forgot():
    mid = _centered()
    with mid:
        _logo_header("Reset your password")
        with st.container(border=True):
            with st.form("forgot_form"):
                email = st.text_input("Account email")
                submitted = st.form_submit_button("Send Reset Code", type="primary", use_container_width=True)

            if submitted:
                user = database.get_user_by_email(email) if email else None
                if user:
                    result = otp_service.send_otp(user, purpose="reset")
                    st.session_state.pending_reset_user = user
                    st.session_state.last_otp_banner = result
                    database.log_action(user["id"], "reset_otp_sent")
                    _set_view("forgot_otp")
                else:
                    # Generic message -> avoid leaking which emails are registered.
                    st.info("If that email is registered, a reset code has been sent.")

            if st.button("← Back to Login", use_container_width=True):
                _set_view("login")


def _render_reset_password():
    mid = _centered()
    with mid:
        _logo_header("Choose a new password")
        user = st.session_state.get("pending_reset_user")
        if not user:
            st.warning("Reset session expired. Please start again.")
            if st.button("← Back to Login"):
                _set_view("login")
            return

        policy = _password_policy()
        with st.container(border=True):
            with st.form("reset_password_form"):
                new_password = st.text_input("New Password", type="password")
                confirm = st.text_input("Confirm New Password", type="password")
                submitted = st.form_submit_button("Update Password", type="primary", use_container_width=True)

            if submitted:
                errors = []
                if new_password != confirm:
                    errors.append("Passwords do not match.")
                errors.extend(security.validate_password_policy(new_password, policy))
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    database.update_password(user["id"], new_password)
                    database.log_action(user["id"], "password_reset")
                    st.session_state.pending_reset_user = None
                    st.success("Password updated! Please log in with your new password.")
                    if st.button("Go to Login", type="primary"):
                        _set_view("login")


# ---------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------
def render_auth_page():
    view = st.session_state.get("auth_view", "login")
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    if view == "login":
        _render_login()
    elif view == "register":
        _render_register()
    elif view == "otp":
        _render_otp(purpose="login")
    elif view == "forgot":
        _render_forgot()
    elif view == "forgot_otp":
        _render_otp(purpose="reset")
    elif view == "reset_password":
        _render_reset_password()
    else:
        st.session_state.auth_view = "login"
        st.rerun()
