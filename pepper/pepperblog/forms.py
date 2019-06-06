from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from pepperblog.models import User
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError

class RegistrationForm(FlaskForm):
    username=StringField("Username", validators=[DataRequired(), Length(min=5,max=20)])
    email=StringField("Email",validators=[DataRequired(),Email()])
    password= PasswordField("Password", validators=[DataRequired(), Length(min=5,max=20)])
    confirm=PasswordField("Confirm Password", validators=[DataRequired(), EqualTo('password')])
    submit=SubmitField("Sign Up")

    def validate_username(self, username):
        user=User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("Username entered already exist.")
            
    def validate_email(self, email):
        email=User.query.filter_by(email=email.data).first()
        if email:
            raise ValidationError("Email entered already used.")
    
class LoginForm(FlaskForm):
    email=StringField("Email",validators=[DataRequired(),Email()])
    password= PasswordField("Password", validators=[DataRequired(), Length(min=5,max=20)])
    remember=BooleanField("stay signed in")
    submit=SubmitField("Login")
    
class UpdateAccountForm(FlaskForm):
    username=StringField("Username", validators=[DataRequired(), Length(min=5,max=20)])
    email=StringField("Email",validators=[DataRequired(),Email()])
    picture=FileField("Profile", validators=[FileAllowed(["jpg","png","svg"])])
    # confirm=PasswordField("Confirm Password", validators=[DataRequired(), EqualTo('password')])
    submit=SubmitField("Update Profile")

    def validate_username(self, username):
        if username.data!=current_user.username:
            user=User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError("Username entered already exist.")
            
    def validate_email(self, email):
        if email.data!=current_user.email:
            email=User.query.filter_by(email=email.data).first()
            if email:
                raise ValidationError("Email entered already used.")


class NewPostForm(FlaskForm):
    title=StringField("Title", validators=[DataRequired()])
    article=TextAreaField("Content", validators=[DataRequired()])
    submit=SubmitField("Post")
