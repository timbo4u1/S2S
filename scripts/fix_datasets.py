with open('README.md') as f:
    t = f.read()
t = t.replace('| Berkeley MHAD | 12 | 11 | 100 | ✅ |', '| Berkeley MHAD | 12 | 11 | 100 | 🔄 planned |')
t = t.replace('| MoVi          | 90 | 20 | 120 | ✅ |', '| MoVi          | 90 | 20 | 120 | 🔄 planned |')
with open('README.md', 'w') as f:
    f.write(t)
print('done')
