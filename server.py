from flask import Flask

import routes

# import config if present
try: import config
except: pass

app = Flask(__name__,
            static_folder='public/',
            template_folder='templates')

app.config.update(dict(
    DEBUG = False,
))

routes.register_routes(app)