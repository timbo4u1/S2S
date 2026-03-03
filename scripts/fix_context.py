with open('CONTEXT.md') as f:
    t = f.read()

t = t.replace('**Version:** v1.3', '**Version:** v1.4')
t = t.replace('zero external dependencies', 'cryptography dep for signing; zero deps in physics core')
t = t.replace('1. **Domain classifier accuracy: 65.9%**', '1. **Domain classifier accuracy: 76.6%** (FINE_MOTOR mode) — standard 5-domain: 65.9%')
t = t.replace('2. **No test suite** — zero pytest coverage currently', '2. **Test suite exists** — tests/test_physics_laws.py, 21 tests ✅')
t = t.replace('4. **No ML integration layer** — `s2s_ml_interface` module designed but not yet implemented', '4. **ML integration layer built** — s2s_standard_v1_3/s2s_ml_interface.py ✅')
t = t.replace('5. **Single commit on GitHub** — 19 local commits not yet pushed (git push pending)', '5. **29 commits on GitHub** ✅')
t = t.replace('6. **No CI/CD** — no GitHub Actions workflow yet', '6. **CI green** — .github/workflows/ci.yml, passing ✅')
t = t.replace('7. **No PyPI package** — must clone repo manually', '7. **PyPI packaging ready** — pyproject.toml committed, not yet published')
t = t.replace('- 2026-03-03: Full roadmap created. 13 GitHub issues defined. Priority: push commits first.',
              '- 2026-03-03: v1.4 complete. CI green. 29 commits. 76.6% accuracy. Dashboard working. 2 stars.')

with open('CONTEXT.md', 'w') as f:
    f.write(t)
print('CONTEXT.md updated to v1.4')
