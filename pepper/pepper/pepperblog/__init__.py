from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flaskext.markdown import Markdown



app=Flask(__name__)

mkd=Markdown(app)
app.config['SECRET_KEY']="9a82397956cabbaa1dc4c7a1706c4a1"
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///site.db"
db=SQLAlchemy(app)
bcrypt=Bcrypt(app)
login_manager=LoginManager(app)
login_manager.login_view="login"
login_manager.login_message_category="danger"

from pepperblog import routes


##check the database
#from pepperblog import db
#from pepperblog.models import User
#User.query.all()