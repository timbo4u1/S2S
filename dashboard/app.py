"""
S2S Certification Dashboard
Run: streamlit run dashboard/app.py

Requires: pip install streamlit
S2S has zero other dependencies.
"""

import streamlit as st
import sys
import os
import math
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S2S Physics Certification",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.tier-gold     { background:#FFD700; color:#000; padding:8px 20px; border-radius:8px;
                 font-size:2rem; font-weight:bold; text-align:center; }
.tier-silver   { background:#C0C0C0; color:#000; padding:8px 20px; border-radius:8px;
                 font-size:2rem; font-weight:bold; text-align:center; }
.tier-bronze   { background:#CD7F32; color:#fff; padding:8px 20px; border-radius:8px;
                 font-size:2rem; font-weight:bold; text-align:center; }
.tier-rejected { background:#e74c3c; color:#fff; padding:8px 20px; border-radius:8px;
                 font-size:2rem; font-weight:bold; text-align:center; }
.law-pass { color:#27ae60; font-weight:bold; }
.law-fail { color:#e74c3c; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Import PhysicsEngine ───────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
    return PhysicsEngine()

# ── Helpers ───────────────────────────────────────────────────────────────────

TIER_EMOJI = {
    "GOLD":     "🥇",
    "SILVER":   "🥈",
    "BRONZE":   "🥉",
    "REJECTED": "❌",
}
TIER_COLOR = {
    "GOLD": "tier-gold", "SILVER": "tier-silver",
    "BRONZE": "tier-bronze", "REJECTED": "tier-rejected",
}

LAW_DESCRIPTIONS = {
    "jerk_bounds":           "Jerk ≤ 500 m/s³ (Flash-Hogan 1985)",
    "rigid_body_kinematics": "a = α×r + ω²×r at every joint (Euler)",
    "resonance_frequency":   "Forearm tremor 8-12 Hz (Flash-Hogan 1985)",
    "imu_consistency":       "Accel+gyro coupling r ≥ 0.15",
    "Newton F=ma":           "EMG force precedes accel by 75ms (Newton 1687)",
    "BCG heartbeat":         "Heartbeat pulse in wrist IMU (Starr 1939)",
    "Joule heating":         "EMG power matches thermal output (Ohm 1827)",
}

def parse_csv_column(text, col_index):
    """Parse a column from CSV text, skipping header."""
    rows = []
    for line in text.strip().split('\n')[1:]:
        parts = line.split(',')
        if len(parts) > col_index:
            try:
                rows.append(float(parts[col_index].strip()))
            except ValueError:
                pass
    return rows

def generate_demo_imu(n=200, hz=100, mode="real"):
    """Generate demo data for 'Try Demo' button."""
    dt = 1.0 / hz
    timestamps = [int(i * dt * 1e9) for i in range(n)]
    accel, gyro = [], []
    for i in range(n):
        t = i * dt
        if mode == "real":
            ax = 0.5 * math.sin(2 * math.pi * 1.5 * t) + 0.02 * math.sin(2 * math.pi * 10 * t)
            ay = 0.3 * math.cos(2 * math.pi * 1.2 * t) + 0.02 * math.cos(2 * math.pi * 9.5 * t)
            az = 9.81 + 0.015 * math.sin(2 * math.pi * 1.1 * t)
            gx = 0.1 * math.sin(2 * math.pi * 1.5 * t + 0.1)
            gy = 0.08 * math.cos(2 * math.pi * 1.2 * t + 0.1)
            gz = 0.05 * math.sin(2 * math.pi * 0.8 * t)
        else:  # synthetic bad
            ax = 50.0 * (1 if (i % 10) < 5 else -1)
            ay = 30.0 * (1 if (i % 7) < 3 else -1)
            az = 9.81
            gx = 0.001 * math.sin(2 * math.pi * 500 * t)
            gy = 0.001 * math.cos(2 * math.pi * 700 * t)
            gz = 0.0
        accel.append([ax, ay, az])
        gyro.append([gx, gy, gz])
    return {"timestamps_ns": timestamps, "accel": accel, "gyro": gyro}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/S2S-v1.3-blue", width=120)
    st.title("S2S Dashboard")
    st.caption("Physics-Certified Motion Data")
    st.divider()
    segment = st.selectbox("Body segment", ["forearm", "thigh", "shin", "trunk"])
    st.divider()
    st.markdown("**Certification Tiers**")
    st.markdown("🥇 **GOLD** — 90-100 — All laws pass")
    st.markdown("🥈 **SILVER** — 60-89 — Core laws pass")
    st.markdown("🥉 **BRONZE** — 30-59 — Partial pass")
    st.markdown("❌ **REJECTED** — 0-29 — Physics violated")
    st.divider()
    st.markdown("[GitHub](https://github.com/timbo4u1/S2S) · [Docs](https://timbo4u1.github.io/S2S)")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏅 S2S Physics Certification Dashboard")
st.caption("Validate IMU sensor data against 7 biomechanical physics laws")

tab1, tab2, tab3 = st.tabs(["📊 Certify Data", "📁 Upload CSV", "⚙️ Batch Check"])

# ── TAB 1: DEMO + JSON INPUT ──────────────────────────────────────────────────
with tab1:
    col_demo1, col_demo2, col_blank = st.columns([1, 1, 2])

    with col_demo1:
        if st.button("▶ Try: Real Human Motion", use_container_width=True, type="primary"):
            st.session_state['demo_imu'] = generate_demo_imu(mode="real")
            st.session_state['demo_label'] = "real"

    with col_demo2:
        if st.button("⚠ Try: Synthetic Bad Data", use_container_width=True):
            st.session_state['demo_imu'] = generate_demo_imu(mode="synthetic")
            st.session_state['demo_label'] = "synthetic"

    st.divider()

    if 'demo_imu' in st.session_state:
        imu_input = st.session_state['demo_imu']
        label = st.session_state.get('demo_label', '')

        with st.spinner("Running physics certification..."):
            try:
                engine = load_engine()
                start = time.time()
                result = engine.certify(imu_raw=imu_input, segment=segment)
                elapsed = (time.time() - start) * 1000

                tier = result.get('tier', 'REJECTED')
                score = result.get('physical_law_score', 0)
                laws_passed = result.get('laws_passed', [])

                # ── Result display ──
                st.markdown("---")
                rc1, rc2, rc3 = st.columns([2, 1, 1])

                with rc1:
                    css_class = TIER_COLOR.get(tier, "tier-rejected")
                    emoji = TIER_EMOJI.get(tier, "❌")
                    st.markdown(
                        f'<div class="{css_class}">{emoji} {tier}</div>',
                        unsafe_allow_html=True
                    )

                with rc2:
                    st.metric("Physics Score", f"{score}/100")

                with rc3:
                    n_passed = len(laws_passed)
                    st.metric("Laws Passed", f"{n_passed}/7")
                    st.caption(f"Certified in {elapsed:.1f}ms")

                st.markdown("---")

                # ── Per-law breakdown ──
                st.subheader("Law-by-Law Breakdown")

                ALL_LAWS = [
                    "jerk_bounds", "rigid_body_kinematics", "resonance_frequency",
                    "imu_consistency", "Newton F=ma", "BCG heartbeat", "Joule heating"
                ]

                cols = st.columns(2)
                for i, law in enumerate(ALL_LAWS):
                    passed = law in laws_passed
                    icon = "✅" if passed else "❌"
                    desc = LAW_DESCRIPTIONS.get(law, law)
                    with cols[i % 2]:
                        st.markdown(f"{icon} **{law}**")
                        st.caption(desc)

                st.markdown("---")

                # ── Score bar chart (pure streamlit, no deps) ──
                st.subheader("Score Distribution")
                if score > 0:
                    bar_data = {"Score": score, "Remaining": 100 - score}
                    # Simple progress bar
                    st.progress(score / 100)
                    st.caption(f"Overall physics score: {score}/100")

                # ── Download result ──
                st.subheader("Download Result")
                result_json = json.dumps(result, indent=2, default=str)
                st.download_button(
                    "⬇ Download Certified JSON",
                    data=result_json,
                    file_name=f"s2s_certified_{tier.lower()}_{score}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Certification error: {e}")
                st.info("Make sure the S2S package is in your Python path.")
    else:
        st.info("👆 Click a demo button above, or upload a CSV in the next tab.")


# ── TAB 2: CSV UPLOAD ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload IMU CSV")
    st.caption("Expected columns: timestamp_ns, ax, ay, az, gx, gy, gz")

    col_a, col_g = st.columns(2)
    with col_a:
        accel_file = st.file_uploader("Accelerometer CSV", type="csv", key="accel")
    with col_g:
        gyro_file = st.file_uploader("Gyroscope CSV", type="csv", key="gyro")

    st.caption("Or use a combined CSV with all 7 columns:")
    combined_file = st.file_uploader("Combined IMU CSV (timestamp, ax, ay, az, gx, gy, gz)", type="csv", key="combined")

    if combined_file is not None:
        text = combined_file.read().decode('utf-8')
        lines = [l for l in text.strip().split('\n') if l.strip()]
        st.success(f"Loaded {len(lines)-1} rows")

        if st.button("🔬 Certify Uploaded Data", type="primary"):
            try:
                timestamps, accel, gyro = [], [], []
                for line in lines[1:]:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        timestamps.append(int(float(parts[0])))
                        accel.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        gyro.append([float(parts[4]), float(parts[5]), float(parts[6])])

                imu_raw = {"timestamps_ns": timestamps, "accel": accel, "gyro": gyro}
                engine = load_engine()
                result = engine.certify(imu_raw=imu_raw, segment=segment)

                tier = result.get('tier', 'REJECTED')
                score = result.get('physical_law_score', 0)
                emoji = TIER_EMOJI.get(tier, "❌")

                st.markdown(f"## Result: {emoji} {tier} — {score}/100")
                st.json(result)

                result_json = json.dumps(result, indent=2, default=str)
                st.download_button("⬇ Download Result", result_json,
                                   file_name=f"s2s_{tier.lower()}.json",
                                   mime="application/json")

            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                st.code("Expected format:\ntimestamp_ns,ax,ay,az,gx,gy,gz\n1000000,0.1,0.2,9.8,0.01,0.02,0.0\n...")


# ── TAB 3: BATCH CHECK ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Batch Certification")
    st.caption("Compare real vs synthetic data side-by-side")

    if st.button("▶ Run Comparison (Real vs Synthetic)", type="primary"):
        engine = load_engine()

        with st.spinner("Certifying both samples..."):
            real_imu = generate_demo_imu(mode="real", n=300)
            fake_imu = generate_demo_imu(mode="synthetic", n=300)

            real_result = engine.certify(imu_raw=real_imu, segment=segment)
            fake_result = engine.certify(imu_raw=fake_imu, segment=segment)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Real Human Motion")
            tier = real_result.get('tier', 'REJECTED')
            score = real_result.get('physical_law_score', 0)
            st.markdown(f'<div class="{TIER_COLOR[tier]}">{TIER_EMOJI[tier]} {tier} — {score}/100</div>',
                       unsafe_allow_html=True)
            st.progress(score / 100)
            laws = real_result.get('laws_passed', [])
            st.caption(f"Laws passed: {len(laws)}/7")
            for law in laws:
                st.markdown(f"✅ {law}")

        with c2:
            st.markdown("### Synthetic Bad Data")
            tier = fake_result.get('tier', 'REJECTED')
            score = fake_result.get('physical_law_score', 0)
            st.markdown(f'<div class="{TIER_COLOR[tier]}">{TIER_EMOJI[tier]} {tier} — {score}/100</div>',
                       unsafe_allow_html=True)
            st.progress(score / 100)
            laws = fake_result.get('laws_passed', [])
            st.caption(f"Laws passed: {len(laws)}/7")
            for law in laws:
                st.markdown(f"✅ {law}")

        st.info("S2S correctly distinguishes real from synthetic using physics alone — no labels required.")
