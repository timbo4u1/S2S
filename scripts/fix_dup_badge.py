with open('README.md') as f: t = f.read()
# Remove the duplicate S2S CI badge (keep only first)
t = t.replace(
    '[![S2S CI](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml)\n[![S2S CI]',
    '[![S2S CI]'
)
with open('README.md','w') as f: f.write(t)
print('Fixed')
