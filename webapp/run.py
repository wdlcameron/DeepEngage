#!/usr/bin/env python
from flaskexample import app
#app.run(host='0.0.0.0', debug = True)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.run(host='0.0.0.0', debug=True)