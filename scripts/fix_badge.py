with open('README.md') as f:
    t = f.read()

badge = '[![S2S CI](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml/badge.svg)](https://github.com/timbo4u1/S2S/actions/workflows/ci.yml)\n'

if 'S2S CI' not in t:
    t = t.replace('[![License: BSL-1.1]', badge + '[![License: BSL-1.1]')
    with open('README.md', 'w') as f:
        f.write(t)
    print('Badge added')
else:
    print('Badge already present')
