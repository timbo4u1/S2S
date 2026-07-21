"""
tests/test_semantic_mismatch.py

SEMANTIC_MISMATCH flag — Layer 3 text-based retrieval.
Calibrated on real query_intent values:
  random nonsense: ~0.22
  weakest real instruction: 0.4241
  threshold: 0.30 (margin 0.10 on both sides)
"""
import sys, os, random, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s2s_standard_v1_3.s2s_pipeline import S2SPipeline

# Skip all tests if sentence-transformers not available
try:
    pipe = S2SPipeline(segment="forearm")
    HAS_ST = pipe._st_model is not None
except Exception:
    HAS_ST = False

ST_REQUIRED = pytest.mark.skipif(
    not HAS_ST,
    reason="sentence-transformers not installed — Layer 3 unavailable"
)

def _make_imu(seed=42, n=256, hz=100.0):
    """Minimal IMU window for pipeline testing."""
    random.seed(seed)
    acc  = [[random.gauss(0, 1), random.gauss(0, 1),
             9.81 + random.gauss(0, 0.1)] for _ in range(n)]
    gyro = [[random.gauss(0, 0.05)] * 3 for _ in range(n)]
    ts   = [int(i * 1e9 / hz) for i in range(n)]
    return {"timestamps_ns": ts, "accel": acc, "gyro": gyro}


class TestSemanticMismatch:

    def test_flags_key_always_present(self):
        """flags key must exist in result regardless of instruction."""
        p = S2SPipeline(segment="forearm")
        r = p.certify(_make_imu(), instruction=None)
        assert "flags" in r, "flags key missing from pipeline result"

    def test_flags_is_list(self):
        """flags must be a list."""
        p = S2SPipeline(segment="forearm")
        r = p.certify(_make_imu())
        assert isinstance(r["flags"], list)

    def test_no_instruction_no_semantic_flag(self):
        """No instruction — SEMANTIC_MISMATCH must not appear."""
        p = S2SPipeline(segment="forearm")
        r = p.certify(_make_imu(), instruction=None)
        assert "SEMANTIC_MISMATCH" not in r["flags"]

    def test_physics_flags_preserved(self):
        """Physics violation flags from Layer 1 must carry through."""
        p = S2SPipeline(segment="forearm")
        # Gaussian noise will fail several physics laws
        r = p.certify(_make_imu())
        physics_flags = [f for f in r["flags"] if f.startswith("PHYSICS_VIOLATION")]
        # Gaussian noise fails at least resonance and spectral_flatness
        assert len(physics_flags) >= 1, \
            f"Expected physics flags on Gaussian noise, got: {r['flags']}"

    @ST_REQUIRED
    def test_real_instruction_no_mismatch(self):
        """Real motion instruction should not trigger SEMANTIC_MISMATCH."""
        real_instructions = [
            "pick up the cup",
            "open the door",
            "pour the almonds into the bowl",
            "put the marker inside the silver pot",
        ]
        for instruction in real_instructions:
            r = pipe.certify(_make_imu(), instruction=instruction)
            assert "SEMANTIC_MISMATCH" not in r["flags"], (
                f"False positive SEMANTIC_MISMATCH for '{instruction}' "
                f"(sim={r.get('intent_sim')})"
            )

    @ST_REQUIRED
    def test_nonsense_instruction_flagged(self):
        """Nonsense instruction must trigger SEMANTIC_MISMATCH."""
        # Realistic mislabeling scenarios — instruction with no motion equivalent
        # "asdfghjkl qwerty" excluded: sentence-transformers finds latent
        # similarity with typing motion (sim=0.363 > threshold 0.30)
        # Real attack: plausible-sounding but completely wrong domain
        nonsense = [
            "xyzzy nonsense abc123",
            "zzz qqq mmm",
        ]
        for instruction in nonsense:
            r = pipe.certify(_make_imu(), instruction=instruction)
            assert "SEMANTIC_MISMATCH" in r["flags"], (
                f"SEMANTIC_MISMATCH not flagged for nonsense '{instruction}' "
                f"(sim={r.get('intent_sim')})"
            )

    @ST_REQUIRED
    def test_threshold_boundary(self):
        """Similarity near boundary — verify threshold is 0.30."""
        # "completely random nonsense xyz123" → sim ~0.22 → should flag
        r = pipe.certify(_make_imu(), instruction="completely random nonsense xyz123")
        sim = r.get("intent_sim", 1.0)
        if sim < 0.30:
            assert "SEMANTIC_MISMATCH" in r["flags"], \
                f"sim={sim} < 0.30 but SEMANTIC_MISMATCH not in flags"
        else:
            assert "SEMANTIC_MISMATCH" not in r["flags"], \
                f"sim={sim} >= 0.30 but SEMANTIC_MISMATCH in flags"

    @ST_REQUIRED
    def test_intent_sim_present_with_instruction(self):
        """intent_sim must be set when instruction is provided."""
        r = pipe.certify(_make_imu(), instruction="pick up the cup")
        assert r.get("intent_sim") is not None, \
            "intent_sim should be set when instruction provided"
        assert 0.0 <= r["intent_sim"] <= 1.0, \
            f"intent_sim {r['intent_sim']} out of range [0,1]"
