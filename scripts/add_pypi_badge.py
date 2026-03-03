with open('README.md') as f: t = f.read()
badge = '[![PyPI](https://img.shields.io/pypi/v/s2s-certify)](https://pypi.org/project/s2s-certify/)\n'
t = t.replace('[![S2S CI]', badge + '[![S2S CI]')
with open('README.md','w') as f: f.write(t)
print('Badge added')
