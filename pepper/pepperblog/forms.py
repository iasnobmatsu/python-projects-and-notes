from flask_wtf import FlaskForm
from pepperblog.models import User
from wtforms import StringField, PasswordField, SubmitField, BooleanField
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
    
