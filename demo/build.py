import re
core = open('core.js').read().split("if (typeof module")[0]
model = open('model.json').read()
import base64
html = open('template.html').read().replace('/*CORE*/', core).replace('/*MODEL*/', model).replace('/*FLOW*/', 'data:image/jpeg;base64,' + base64.b64encode(open('flow.jpg','rb').read()).decode())
open('index.html', 'w').write(html); print(len(html))
