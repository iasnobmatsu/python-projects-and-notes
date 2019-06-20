from pepperblog import db, login_manager
from datetime import datetime
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as serializer


#decorator for login manager to work
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String(20), unique=True, nullable=False)
    email=db.Column(db.String(120), unique=True, nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default='default.jpg')
    password=db.Column(db.String(60), nullable=False)
    posts=db.relationship("Post", backref='author',lazy=True)



    def reset_token(self):
        s=serializer(app.config['SECRET_KEY'], 3600)
        return s.dumps({'user_id':self.id}).decode('utf-8')

    @staticmethod
    def verify_token(token):
        s=serializer(app.config['SECRET_KEY'])
        try:
            user_id=s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return "User ('{}', '{}', '{}')".format(self.username, self.email, self.image_file)

class Post(db.Model):
     id=db.Column(db.Integer, primary_key=True)
     title=db.Column(db.String(100), nullable=False)
     ismarkdown=db.Column(db.Integer, default=0)
     date_posted=db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
     content=db.Column(db.Text,nullable=False)
     user_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
   
  

     def __repr__(self):
        return "Post ('{}', '{}')".format(self.title, self.date_posted)

class Like(db.Model):
    __tablename__='like'    
    id=db.Column(db.Integer, primary_key=True)
    userid=db.Column(db.Integer)
    postid=db.Column(db.Integer)

    def __repr__(self):
        return "Like ('{}', '{}')".format(self.userid, self.postid)



    