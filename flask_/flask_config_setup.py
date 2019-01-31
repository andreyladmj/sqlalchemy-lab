from flask import Flask
app = Flask(__name__)
app.config.from_object('prod_settings.Config')
print(app.config)
'''<Config {'SESSION_COOKIE_HTTPONLY': True, 'LOGGER_NAME': '__main__',
         'APPLICATION_ROOT': None, 'MAX_CONTENT_LENGTH': None,
         'PRESERVE_CONTEXT_ON_EXCEPTION': None,
         'LOGGER_HANDLER_POLICY': 'always',
         'SESSION_COOKIE_DOMAIN': None, 'SECRET_KEY': None,
         'EXPLAIN_TEMPLATE_LOADING': False,
         'TRAP_BAD_REQUEST_ERRORS': False,
         'SESSION_REFRESH_EACH_REQUEST': True,
         'TEMPLATES_AUTO_RELOAD': None,
         'JSONIFY_PRETTYPRINT_REGULAR': True,
         'SESSION_COOKIE_PATH': None,
         'SQLURI': 'postgres://tarek:xxx@localhost/db',
         'JSON_SORT_KEYS': True, 'PROPAGATE_EXCEPTIONS': None,
         'JSON_AS_ASCII': True, 'PREFERRED_URL_SCHEME': 'http',
         'TESTING': False, 'TRAP_HTTP_EXCEPTIONS': False,
         'SERVER_NAME': None, 'USE_X_SENDFILE': False,
         'SESSION_COOKIE_NAME': 'session', 'DEBUG': False,
         'JSONIFY_MIMETYPE': 'application/json',
         'PERMANENT_SESSION_LIFETIME': datetime.timedelta(31),
         'SESSION_COOKIE_SECURE': False,
         'SEND_FILE_MAX_AGE_DEFAULT': datetime.timedelta(0, 43200)}> '''